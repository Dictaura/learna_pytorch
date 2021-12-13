import torch
from torch import nn as nn
import torch.functional as F


class Learna_Net(nn.Module):
    def __init__(
            self, in_size, out_size,
            emb_num, emb_out_size,
            conv1_in_size, conv1_out_size, conv1_kernel_size, conv1_stride, conv1_activate,
            conv2_in_size, conv2_out_size, conv2_kernel_size, conv2_stride, conv2_activate,
            lstm_in_size, lstm_hidden_size, lstm_num_layers,
            dense_in_size, dense_out_size
    ):
        super(Learna_Net, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.emb = nn.Embedding(emb_num, emb_out_size)
        self.conv1 = nn.Conv1d(conv1_in_size, conv1_out_size, conv1_kernel_size, conv1_stride, conv1_activate)
        self.conv2 = nn.Conv1d(conv2_in_size, conv2_out_size, conv2_kernel_size, conv2_stride, conv2_activate)

        self.lstm = nn.LSTM(lstm_in_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers)

        self.dense = nn.Linear(dense_in_size, dense_out_size)
