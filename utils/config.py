import torch
import argparse

parser = argparse.ArgumentParser()

device = torch.device("cpu")

action_space = 4

# net
net_dict = {
    'in_size': 1,
    'emb_num': 4, 'emb_out_size': 4,
    'conv1_in_size': 4, 'conv1_out_size': 16, 'conv1_kernel_size': 3, 'conv1_stride': 1,
    'conv2_in_size': 16, 'conv2_out_size': 8, 'conv2_kernel_size': 2, 'conv2_stride': 1,
    'lstm_in_size': 8, 'lstm_hidden_size': 8, 'lstm_num_layers': 2,
    'fc1_a_in_size': 8, 'fc1_a_out_size': 8, 'fc2_a_in_size': 8, 'fc2_a_out_size': action_space,
    'fc1_c_in_size': 8, 'fc1_c_out_size': 4, 'fc2_c_in_size': 4, 'fc2_c_out_size': 1,
    'lr': 0.00001
}

for key, value in net_dict.items():
    parser.add_argument(key, action='store_const', const=value)

net_param = parser.parse_args()
