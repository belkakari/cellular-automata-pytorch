import torch
import torch.nn as nn
import torch.nn.functional as F


class Perception(nn.Module):
    def __init__(self, channels=16, norm_kernel=False):
        super().__init__()
        self.channels = channels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]) / 8
        sobel_y = torch.tensor([[1.0, 2.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, -2.0, -1.0]]) / 8
        identity = torch.tensor([[0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0]])

        self.kernel =  torch.stack((identity, sobel_x, sobel_y)).repeat(channels, 1, 1).unsqueeze(1)
        if norm_kernel:
            self.kernel /= channels

    def forward(self, state_grid):
        return F.conv2d(state_grid, 
                        self.kernel.to(state_grid.device), 
                        groups=self.channels,
                        padding=1)  # thanks https://github.com/PWhiddy/Growing-Neural-Cellular-Automata-Pytorch?files=1 for the group parameter


class Policy(nn.Module):
    def __init__(self, state_dim=16, interm_dim=128,
                 use_embedding=True, kernel=1, padding=0,
                 bias=False):
        super().__init__()
        dim = state_dim * 3
        if use_embedding:
            dim += 1
        self.conv1 = nn.Conv2d(dim, interm_dim, kernel, padding=padding)
        self.conv2 = nn.Conv2d(interm_dim, state_dim, kernel, padding=padding,
                               bias=bias)
        nn.init.constant_(self.conv2.weight, 0.)
        if bias:
            nn.init.constant_(self.conv2.bias, 0.)

    def forward(self, state):
        interm = self.conv1(state)
        interm = torch.relu(interm)
        return self.conv2(interm)


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
