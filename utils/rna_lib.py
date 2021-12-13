import math
import RNA
import random
import numpy as np
import torch


#############################################################
# 全局常量
#############################################################
base_color_dict = {'A': 'y', 'U': 'b', 'G': 'r', 'C': 'g'}
base_list = ['A', 'U', 'C', 'G']
onehot_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
base_pair_dict_6 = {'A': ['U'], 'U': ['A', 'G'], 'G': ['U', 'C'], 'C': ['G'] }
base_pair_dict_4 = {'A': ['U'], 'U': ['A'], 'G': ['C'], 'C': ['G']}
base_pair_list_6 = [['A', 'U'], ['U', 'A'], ['U', 'G'], ['G', 'U'], ['G', 'C'], ['C', 'G']]
base_pair_list_4 = [['A', 'U'], ['U', 'A'], ['C', 'G'], ['G', 'C']]
onehot_pair_list_4 = [
    [[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 0, 0], [1, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]], [[0, 0, 0, 1], [0, 0, 1, 0]]
]
onehot_pair_list_6 = [
    [[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 0, 0], [1, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 0, 1], [0, 1, 0, 0]],
    [[0, 0, 0, 1], [0, 0, 1, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]
]

dotB_code_dist = {
    '.': 1, '(': 2, ')': 3
}

#############################################################
# 数据结构和工具
#############################################################

class Stack(object):
    """栈"""
    def __init__(self):
         self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items == []

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items)-1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)


def dice(max, min=0):
    """
    骰子工具
    :param max: 最大数+1
    :param min: 最小数，默认为0
    :return: 随机采样数
    """
    return random.randrange(min, max, 1)

#############################################################
# 结构转换
#############################################################

def struct_dotB2Edge(dotB):
    l = len(dotB)
    # 初始化
    u = []
    v = []
    # for i in range(l - 1):
    #     u += [i, i + 1]
    #     v += [i + 1, i]
    str_list = list(dotB)
    stack = Stack()
    for i in range(l):
        if (str_list[i] == '('):
            stack.push(i)
        elif (str_list[i] == ')'):
            last_place = stack.pop()
            u += [i, last_place]
            v += [last_place, i]

    edges = torch.tensor(np.array([u, v])).t()
    return edges


def struct_dotB2Code(dotB, padding_size):
    code = []
    for c in dotB:
        code.append(dotB_code_dist[c])
    code = [0] * padding_size + code + [0] * padding_size
    return code