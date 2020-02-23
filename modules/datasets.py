import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class StateGridSet(Dataset):
    def __init__(self, target, use_coords=False,
                 batch_size=10, random_spawn=True,
                 pad=50):
        self.target = F.pad(target, (0, pad, 0, pad))
        self.use_coords = use_coords
        self.batch_size = batch_size
        self.random_spawn = random_spawn

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        state_grid = torch.zeros((16, self.target.shape[-2],
                                  self.target.shape[-1]),
                                 requires_grad=False,
                                 device=self.target.device)
        if self.random_spawn:
            center = random.randint(int(0.2 * (self.target.shape[-2] - 1)),
                                    int(0.8 * (self.target.shape[-2] - 1)))
        else:
            center = state_grid.shape[2] // 2
        state_grid[4:, center, center] += torch.ones_like(state_grid[4:,
                                                                     center,
                                                                     center])
        state_grid[4, center, center] = torch.ones_like(state_grid[4,
                                                                   center,
                                                                   center])

        if self.use_coords:
            xv, yv = torch.meshgrid([torch.linspace(-1, 1, steps=128),
                                     torch.linspace(-1, 1, steps=128)])
            state_grid[-1] = xv
            state_grid[-2] = yv

        return state_grid, self.target
