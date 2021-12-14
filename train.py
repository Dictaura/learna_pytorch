from datetime import datetime

import RNA
import torch
from utils.config import actor_param, critic_param, device, action_space
from rl_lib.ppo import Agent, Transition
from rl_lib.environment import RNA_ENV
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

def main():
    ########################### global parameters ###########################
    # url
    root = os.path.dirname(os.path.realpath(__file__))
    data_dir = root + '/data/raw/rfam_learn/train/1.rna'
    # time
    start_time = datetime.now().replace(microsecond=0)
    start_str = start_time.strftime("%Y_%m_%d_%H_%M_%S")

    ########################### environment ###########################
    f = open(data_dir)
    iter_f = iter(f)
    dataset = []
    for line_ in iter_f:
        line = line_.replace('\n', '')
        dataset.append(line)

    window_size = 11
    env = RNA_ENV(dataset[0], window_size, action_space)

    ########################### agent ###########################
    batch_size = 64
    k_epoch = 6
    eps_clip = 0.1
    gamma = 0.9
    buffer_vol = 3
    lr_decay = 0.999

    agent = Agent(
        actor_param, critic_param, batch_size, k_epoch, eps_clip, gamma, action_space, buffer_vol, device, lr_decay

    )

    ########################### setting ###########################
    max_round = 3000
    update_freq = 1
    log_freq = 1
    save_freq = 20

    ########################### log ###########################
    log_main_dir = root + '/logs/'
    if not os.path.exists(log_main_dir):
        os.makedirs(log_main_dir)

    log_dir = log_main_dir + '/' + start_str + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # tensorboard
    tensorboard_dir = log_dir + '/tensor/'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir, comment=start_str)
    writer.add_text('time', start_str)

    # log
    log_f_dir = log_dir + '/log/'
    if not os.path.exists(log_f_dir):
        os.makedirs(log_f_dir)
    log_f_dir += 'log.csv'
    log_f = open(log_f_dir, "w+")
    log_f.write("episode, reward, distance, sequence\n")

    ########################### training process ###########################
    print("Started training at (GMT) : ", start_time)
    for ep in range(1, max_round+1):
        print("=====================================" + str(ep) + "==================================================")
        done = False
        state = env.reset()

        while not done:
            state = torch.tensor(state).long().unsqueeze(0).to(device)
            action, a_log_prob = agent.work(state)
            next_state, reward, done = env.step(action)

            trainsition = Transition(state, action, a_log_prob, reward, next_state, done)
            agent.storeTransition(trainsition)

            state = next_state

        if ep % log_freq == 0:
            seq = ''.join(env.seq_base_list)
            real_dotB = RNA.fold(seq)[0]
            distance = RNA.hamming_distance(env.dotB, real_dotB)
            reward = agent.buffer[-1].reward
            log_f.write('{}, {}, {}, {}\n'.format(ep, reward, distance, seq))

            writer.add_scalar('reward', reward, ep)
            writer.add_scalar('distance', distance, ep)

        if ep % update_freq == 0:
            loss_a, loss_c = agent.trainStep()
            loss_a = abs(loss_a)

            writer.add_scalar('loss_a', loss_a, ep)
            writer.add_scalar('loss_c', loss_c, ep)

            for tag_, value in agent.actorNet.named_parameters():
                tag_ = 'a.' + tag_.replace('.', '/')
                writer.add_histogram(tag_, value, ep)
                writer.add_histogram(tag_ + '/grad', value.grad.data.cpu().numpy(), ep)

            for tag_, value in agent.criticNet.named_parameters():
                tag_ = 'c.' + tag_.replace('.', '/')
                writer.add_histogram(tag_, value, ep)
                writer.add_histogram(tag_ + '/grad', value.grad.data.cpu().numpy(), ep)
            writer.add_histogram('lr_a', agent.optimizer_a.state_dict()['param_groups'][0]['lr'], ep)
            writer.add_histogram('lr_c', agent.optimizer_c.state_dict()['param_groups'][0]['lr'], ep)

            agent.clean_buffer()

    ########################### finished ###########################
    log_f.close()

    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")








if __name__ == '__main__':
    main()