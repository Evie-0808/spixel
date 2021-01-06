import numpy as np
import sys
import cv2
import torch
from loss2 import compute_fp
N_ACTIONS=9
class State():
    def __init__(self, size, move_range):
        # image实际上就是状态
        self.oriimage = None
        self.p = np.zeros(size, dtype=np.float32)
        self.move_range = move_range
        self.current_fp = None

    def reset(self, x, fp):
        self.oriimage = x
        self.current_fp = fp

    def step(self, act, prev_fp):
        # act: b, w, h
        b, c, h, w = self.p.shape
        neutral = (self.move_range - 1)/2
        move = act.astype(np.float32)
        choose_grid = np.ones(self.p.shape[-2: ], self.p.dtype)
        zeors = np.zeros(self.p.shape[-2:], self.p.dtype)
        background = np.zeros((b, N_ACTIONS, self.p.shape[-2], self.p.shape[-1]), self.p.dtype)
        # 遍历每一个batch
        for i in range(0, b):
            # actions: 动作i选择第i个
            for g in range(N_ACTIONS):
                background[i][g] = np.where(act[i]==g, choose_grid, zeors)
        self.p = background
        # 更新状态
        self.current_fp = compute_fp(torch.Tensor(background).cuda(), torch.Tensor(prev_fp).cuda()).detach().cpu().numpy()

    def to_probin(self, act):

        b, c, h, w = self.p.shape
        neutral = (self.move_range - 1) / 2
        move = act.astype(np.float32)
        choose_grid = np.ones(self.p.shape[-2:], self.p.dtype)
        zeors = np.zeros(self.p.shape[-2:], self.p.dtype)
        background = np.zeros((b, N_ACTIONS, self.p.shape[-2], self.p.shape[-1]), self.p.dtype)
        # 遍历每一个batch
        for i in range(0, b):
            # actions: 动作i选择第i个
            for g in range(N_ACTIONS):
                background[i][g] = np.where(act[i] == g, choose_grid, zeors)

        return background

    def set_prev_loss(self, loss):
        self.old_loss = loss

