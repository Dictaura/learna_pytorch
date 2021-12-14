import torch

from network.learna_net import Learna_Net
from torch import nn
from collections import namedtuple
from torch.autograd import Variable
from torch import no_grad, clamp
import os
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import torch.nn.functional as F

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

def fetch_items(list, index):
    return [list[i] for i in index]


class Agent(nn.Module):
    def __init__(
            self, net_param,
            batch_size, k_epoch, eps_clip, gamma, action_space, buffer_vol, device, lr_decay
    ):
        super(Agent, self).__init__()
        self.net = Learna_Net(
            net_param.in_size,
            net_param.emb_num, net_param.emb_out_size,
            net_param.conv1_in_size, net_param.conv1_out_size, net_param.conv1_kernel_size, net_param.conv1_stride,
            net_param.conv2_in_size, net_param.conv2_out_size, net_param.conv2_kernel_size, net_param.conv2_stride,
            net_param.lstm_in_size, net_param.lstm_hidden_size, net_param.lstm_num_layers,
            net_param.fc1_a_in_size, net_param.fc1_a_out_size, net_param.fc2_a_in_size, net_param.fc2_a_out_size,
            net_param.fc1_c_in_size, net_param.fc1_c_out_size, net_param.fc2_c_in_size, net_param.fc2_c_out_size,
        )

        # self.criticNet = Learna_Net(
        #     critic_param.in_size, critic_param.out_size,
        #     critic_param.emb_num, critic_param.emb_out_size,
        #     critic_param.conv1_in_size, critic_param.conv1_out_size, critic_param.conv1_kernel_size,
        #     critic_param.conv1_stride,
        #     critic_param.conv2_in_size, critic_param.conv2_out_size, critic_param.conv2_kernel_size,
        #     critic_param.conv2_stride,
        #     critic_param.lstm_in_size, critic_param.lstm_hidden_size, critic_param.lstm_num_layers,
        #     critic_param.fc1_in_size, critic_param.fc1_out_size,
        #     critic_param.fc2_in_size,
        # )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=net_param.lr)
        # self.optimizer_c = torch.optim.Adam(self.criticNet.parameters(), lr=critic_param.lr)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)
        # self.scheduler_c = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_c, lr_decay)

        self.batch_size = batch_size
        self.k_epoch = k_epoch
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.action_space = action_space
        self.buffer_vol = buffer_vol
        self.device = device
        self.buffer = []
        self.buffer_cnt = 0

    def forward(self, states, actions):
        # value = self.criticNet(states)
        # action_probs = self.actorNet(states)
        action_probs, value = self.net(states)
        action_probs = torch.softmax(action_probs, dim=1)

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_log_probs, value, dist_entropy

    def work(self, state):
        with no_grad():
            # action_probs = self.actorNet(state)
            action_probs, _ = self.net(state)
            action_probs = torch.softmax(action_probs, dim=1)

        action = torch.multinomial(action_probs, 1)

        dist = Categorical(action_probs)
        action_log_prob = dist.log_prob(action)

        return action.detach().item(), action_log_prob.detach().item()

    def storeTransition(self, transition):
        self.buffer.append(transition)

    def clean_buffer(self):
        self.buffer_cnt += 1
        if self.buffer_cnt % self.buffer_vol == 0:
            del self.buffer[:]
            self.buffer_cnt = 0

    def trainStep(self):
        state_list = []
        next_state_list = []
        reward_list = []
        action_list = []
        Gt_list = []
        old_a_log_p_list = []
        done_list = []

        state_list = [t.state for t in self.buffer]
        action_list = [t.action for t in self.buffer]
        reward_list = [t.reward for t in self.buffer]
        done_list = [t.done for t in self.buffer]
        old_a_log_p_list = [t.a_log_prob for t in self.buffer]

        R = 0
        for r, done in zip(reward_list[::-1], done_list[::-1]):
            if done:
                R = 0
            R = r + R * self.gamma
            Gt_list.insert(0, R)

        Gts = torch.tensor(Gt_list).to(self.device)
        states = torch.cat(state_list, dim=0).to(self.device)
        actions = torch.tensor(action_list).to(self.device)
        old_a_log_ps = torch.tensor(old_a_log_p_list).to(self.device)

        for i in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(len(state_list))), self.batch_size, False):
                Gt_index = Gts[index]
                state_index = states[index]
                action_index = actions[index]
                old_a_log_p_index = old_a_log_ps[index]

                state_index = state_index.long().to(self.device)

                a_log_p, v, dist_entropy = self.forward(state_index, action_index)
                advantage = Gt_index - v.detach().view(-1,)

                ratio = torch.exp(a_log_p - old_a_log_p_index.detach())
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage

                loss_a = -torch.min(surr1, surr2).mean()
                loss_c = F.mse_loss(Gt_index, v.view(-1))

                loss_all = loss_a + 0.5 * loss_c - 0.01 * dist_entropy.mean()

                # self.optimizer_a.zero_grad()
                # self.optimizer_c.zero_grad()
                self.optimizer.zero_grad()

                loss_all.backward()

                # self.optimizer_a.step()
                # self.optimizer_c.step()
                self.optimizer.step()
            print("Loss_A: {}, Loss_c: {}".format(loss_a.item(), loss_c.item()))

        # self.scheduler_a.step()
        # self.optimizer_c.step()
        self.scheduler.step()

        return loss_a.item(), loss_c.item()




