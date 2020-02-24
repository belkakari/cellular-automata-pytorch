import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from modules.utils import load_emoji

class StateGridSet(Dataset):
    def __init__(self, emoji='🦎', use_coords=False,
                 batch_size=10, random_spawn=True,
                 pad=50, target_size=128):
        # TODO: refactor padding/resize

        self.transform = [transforms.Pad(pad),
                          transforms.Resize(target_size),
                          transforms.ToTensor(),
                          transforms.Normalize((.5, .5, .5, .5),
                                               (.5, .5, .5, .5))]
        self.transform = transforms.Compose(self.transform)

        self.target = self.transform(load_emoji(emoji=emoji))

        self.use_coords = use_coords
        self.batch_size = batch_size
        self.random_spawn = random_spawn

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        state_grid = torch.zeros((16, self.target.shape[-2],
                                 self.target.shape[-1]),
                                 requires_grad=False)
        if self.random_spawn:
            center = random.randint(int(0.2 * (self.target.shape[-2] - 1)),
                                    int(0.8 * (self.target.shape[-2] - 1)))
        else:
            center = state_grid.shape[2] // 2
        state_grid[3:, center, center] = 1.

        if self.use_coords:
            xv, yv = torch.meshgrid([torch.linspace(-1, 1, steps=128),
                                     torch.linspace(-1, 1, steps=128)])
            state_grid[-1] = xv
            state_grid[-2] = yv

        return state_grid, self.target
