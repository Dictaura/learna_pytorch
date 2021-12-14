import torch
import argparse

parser = argparse.ArgumentParser()

device = torch.device("cpu")

action_space = 4

# actor net
actor_dict = {
    'in_size': 1, 'out_size': action_space,
    'emb_num': 4, 'emb_out_size': 4,
    'conv1_in_size': 4, 'conv1_out_size': 16, 'conv1_kernel_size': 3, 'conv1_stride': 1,
    'conv2_in_size': 16, 'conv2_out_size': 8, 'conv2_kernel_size': 2, 'conv2_stride': 1,
    'lstm_in_size': 8, 'lstm_hidden_size': 8, 'lstm_num_layers': 2,
    'fc1_in_size' : 8, 'fc1_out_size': 8,
    'fc2_in_size': 4
}

for key, value in actor_dict.items():
    parser.add_argument(key, action='store_const', const=value)

actor_param = parser.parse_args()

# critic net
critic_dict = {
    'in_size': 1, 'out_size': action_space,
    'emb_num': 4, 'emb_out_size': 4,
    'conv1_in_size': 4, 'conv1_out_size': 16, 'conv1_kernel_size': 3, 'conv1_stride': 1,
    'conv2_in_size': 16, 'conv2_out_size': 8, 'conv2_kernel_size': 2, 'conv2_stride': 1,
    'lstm_in_size': 8, 'lstm_hidden_size': 8, 'lstm_num_layers': 2,
    'fc1_in_size': 8, 'fc1_out_size': 4,
    'fc2_in_size': 1
}

for key, value in critic_dict.items():
    parser.add_argument(key, action='store_const', const=value)

critic_param = parser.parse_args()