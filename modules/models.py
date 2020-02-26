import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import utils


class AbstractCAModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_input(self, state_grid, target):
        self.state_grid = state_grid
        self.target = target

    def forward(self):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError


class SimpleCA(AbstractCAModel):
    def __init__(self, perception, policy, config,
                 logger=None, grad_clip=3):
        super().__init__()
        self.perception = perception
        self.policy = policy
        self.config = config
        self.use_coords = config['model']['use_coords']
        self.stochastic_prob = config['model']['stochastic_prob']
        self.optim = torch.optim.Adam(list(self.policy.parameters()),
                                      lr=config['optim']['lr'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim,
                                                config['optim']['milestones'],
                                                gamma=config['optim']['gamma'])
        self.loss_fn = nn.MSELoss()
        self.logger = logger
        self.grad_clip = grad_clip

    def forward(self):
        alive_pre = utils.alive_mask(self.state_grid, thr=0.1)
        perception_grid = self.perception(self.state_grid)
        ds_grid = self.policy(perception_grid)
        mask = utils.stochastic_update_mask(ds_grid,
                                            prob=self.stochastic_prob)
        self.state_grid = self.state_grid + ds_grid * mask
        alive_post = utils.alive_mask(self.state_grid, thr=0.1)
        final_mask = (alive_post.bool() & alive_pre.bool()).float()
        self.state_grid = self.state_grid * final_mask

        if self.use_coords:
            xgrid = torch.linspace(-1, 1, steps=self.target.shape[-1])
            ygrid = torch.linspace(-1, 1, steps=self.target.shape[-2])
            xv, yv = torch.meshgrid([xgrid, ygrid])
            self.state_grid[:, -1, ...] = xv[None, :, :]
            self.state_grid[:, -2, ...] = yv[None, :, :]

        return final_mask


    def optimize_parameters(self):
        loss_value = self.loss_fn(self.target[:, :4, ...],
                                  self.state_grid[:, :4, ...])
        self.optim.zero_grad()
        loss_value.backward()
        if self.logger:
            norm = []
            for p in self.policy.parameters():
                param_norm = p.grad.data.norm(2)
                norm.append(param_norm.item())
            self.logger.debug(f'norm before clipping, {norm}')

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                       max_norm=self.grad_clip)

        if self.logger:
            norm = []
            for p in self.policy.parameters():
                param_norm = p.grad.data.norm(2)
                norm.append(param_norm.item())
            self.logger.debug(f'norm after clipping, {norm}')
        self.optim.step()
        self.scheduler.step()
        return loss_value
