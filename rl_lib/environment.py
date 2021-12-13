import RNA
import torch
import gym
from utils.rna_lib import struct_dotB2Edge, struct_dotB2Code, base_list, base_pair_dict_4, base_pair_dict_6


class RNA_ENV(gym):
    def __init__(self, dotB, windw_size, action_space=4):
        super(RNA_ENV, self).__init__()
        self.dotB = dotB
        self.window_size = windw_size
        self.padding_size = windw_size // 2
        self.h_edges = struct_dotB2Edge(self.dotB)
        self.scope = []
        self.l = len(self.dotB)
        self.sight_center = 0
        self.struct_code = struct_dotB2Code(self.dotB, self.padding_size)
        if action_space == 4:
            self.base_pair_dict = base_pair_dict_4
        else:
            self.base_pair_dict = base_pair_dict_6
        self.seq_base_list = ['A'] * self.l

    def renew(self):
        self.seq_base_list = [' '] * self.l
        self.sight_center = 0
        self.struct_code = struct_dotB2Code(self.dotB, self.padding_size)
        sight_start = self.sight_center
        sight_end = self.sight_center + self.window_size
        scope = self.struct_code[sight_start:sight_end]
        return scope

    def observe(self):
        while self.seq_base_list[self.sight_center] != ' ':
            self.sight_center += 1
            if self.sight_center == self.l:
                return -1
        sight_start = self.sight_center
        sight_end = self.sight_center + self.window_size
        scope = self.struct_code[sight_start:sight_end]
        return scope

    def step(self, action):
        base = base_list[action]
        place = self.sight_center
        self.seq_base_list[place] = base
        if self.dotB[self.sight_center] != '.':
            base_pair = self.base_pair_dict[base]
            pair_index = torch.where(self.h_edges[0] == place)
            place_pair = self.h_edges[pair_index][1]
            self.seq_base_list[place_pair] = base_pair

        scope = self.observe()

        reward = 0
        if scope == -1:
            seq = ''.join(self.seq_base_list)
            dotB_real = RNA.fold(seq)[0]
            distance = RNA.hamming_distance(self.dotB, dotB_real)
            reward = 1 - distance / self.l

        return scope, reward

