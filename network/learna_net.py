import torch
from torch import nn as nn
import torch.nn.functional as F


class Learna_Net(nn.Module):
    def __init__(
            self, in_size, out_size,
            emb_num, emb_out_size,
            conv1_in_size, conv1_out_size, conv1_kernel_size, conv1_stride,
            conv2_in_size, conv2_out_size, conv2_kernel_size, conv2_stride,
            lstm_in_size, lstm_hidden_size, lstm_num_layers,
            fc1_in_size, fc1_out_size,
            fc2_in_size
    ):
        super(Learna_Net, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.emb = nn.Embedding(emb_num, emb_out_size)
        self.conv1 = nn.Conv1d(conv1_in_size, conv1_out_size, conv1_kernel_size, conv1_stride)
        self.conv2 = nn.Conv1d(conv2_in_size, conv2_out_size, conv2_kernel_size, conv2_stride)

        self.lstm = nn.LSTM(lstm_in_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers)

        self.fc1 = nn.Linear(fc1_in_size, fc1_out_size)
        self.fc2 = nn.Linear(fc2_in_size, out_size)

    def forward(self, x):
        x_in  = x
        x_emb = self.emb(x_in)
        x_emb = torch.transpose(x_emb, -2, -1)
        x_conv1 = F.relu(self.conv1(x_emb))
        x_conv2 = F.relu(self.conv2(x_conv1))
        x_conv2 = torch.transpose(x_conv2, -2, -1)
        x_conv2 = torch.transpose(x_conv2, 0, 1)
        x_lstm, _ = self.lstm(x_conv2)
        x_lstm = x_lstm[-1::]
        x_lstm = torch.transpose(x_lstm, 0, 1).squeeze(1)
        x_fc1 = F.relu(self.fc1(x_lstm))
        x_fc2 = F.relu(self.fc2(x_fc1))

        return x_fc2