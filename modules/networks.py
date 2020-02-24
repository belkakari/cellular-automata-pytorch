import torch
import torch.nn as nn
import torch.nn.functional as F


class Perception(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        sobel_x = (torch.tensor([[-1, 0, +1],
                                 [-2, 0, +2],
                                 [-1, 0, +1]]).float() / 8)[None, ...]
        sobel_y = (torch.tensor([[-1, -2, -1],
                                 [0, 0, 0],
                                 [+1, +2, +1]]).float() / 8)[None, ...]
        idt = torch.tensor([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]).float()[None, ...]

        self.kernel = torch.stack([idt,
                                   sobel_x,
                                   sobel_y],
                                  dim=0).repeat(channels,
                                                channels, 1, 1)
        self.perception = nn.Conv2d(channels, channels * 3,
                                    kernel_size=3, bias=False,
                                    padding=1)
        self.perception.weight = nn.Parameter(self.kernel)

    def forward(self, state_grid):
        return self.perception(state_grid)


class Policy(nn.Module):
    def __init__(self, state_dim=16, interm_dim=128,
                 use_embedding=True, kernel=1, padding=0):
        super().__init__()
        dim = state_dim * 3
        if use_embedding:
            dim += 1
        self.conv1 = nn.Conv2d(dim, interm_dim, kernel, padding=padding)
        self.conv2 = nn.Conv2d(interm_dim, state_dim, kernel, padding=padding)
        nn.init.constant_(self.conv2.weight, 0.)
        nn.init.constant_(self.conv2.bias, 0.)

    def forward(self, state):
        interm = self.conv1(state)
        interm = nn.ReLU()(interm)
        return torch.tanh(self.conv2(interm))


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Sequential(*[nn.Conv2d(4, 16, 3),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 16, 3),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 1, 3)])

    def forward(self, img):
        return nn.AdaptiveAvgPool2d((1, 1))(self.embedder(img))
