# -*- coding: utf-8 -*-

import argparse
import os
import random
import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.datasets import StateGridSet
from modules.networks import Perception, Policy
from modules.utils import alive_mask, load_emoji, stochastic_update_mask, test

parser = argparse.ArgumentParser(description='Train neural cellular automata')
parser.add_argument('-c', '--config', type=str,
                    help='path to config .yaml')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

config_path = args.config

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = config['device']
stochastic_prob = config['stochastic_prob']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
output_folder = config['output_folder']

img = load_emoji(emoji='🦎')
img = transforms.ToTensor()(img)
img = transforms.Normalize(tuple(0.5 for _ in range(img.shape[0])),
                           tuple(0.5 for _ in range(img.shape[0])))(img)
img = img.to(device)

policy = Policy(use_embedding=False, kernel=1, padding=0).to(device)
perception = Perception(channels=16).to(device)
optim = torch.optim.Adam(list(policy.parameters()) +
                         list(perception.parameters()), lr=2e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 100, gamma=0.7)
loss_fn = nn.MSELoss()

dset = StateGridSet(img, use_coords=True,
                    batch_size=batch_size,
                    random_spawn=False)
dset_test = StateGridSet(img, use_coords=True,
                         batch_size=1,
                         random_spawn=False)
dloader = DataLoader(dset, batch_size=batch_size)
dloader_test = DataLoader(dset, batch_size=1)

xv, yv = torch.meshgrid([torch.linspace(-1, 1, steps=img.shape[-1]),
                         torch.linspace(-1, 1, steps=img.shape[-2])])

for epoch in range(num_epochs):
    n_steps = random.randint(100, 150)
    split_rate = random.randint(30, 40)
    for state_grid, target in dloader:
        for k in range(n_steps):
            alive_pre = alive_mask((state_grid + 1.) / 2., thr=0.1)
            perception_grid = perception(state_grid)
            ds_grid = policy(perception_grid)
            mask = stochastic_update_mask(ds_grid,
                                          prob=stochastic_prob)
            state_grid = state_grid + ds_grid * mask
            alive_post = alive_mask((state_grid + 1.) / 2., thr=0.1)
            final_mask = (alive_post.bool() & alive_pre.bool()).float()
            state_grid = state_grid * final_mask

            if dset.use_coords:
                state_grid[:, -1, ...] = xv[None, :, :]
                state_grid[:, -2, ...] = yv[None, :, :]

            if k % split_rate == 0:
                loss_value = loss_fn(target[:, :4, ...],
                                     state_grid[:, :4, ...])
                optim.zero_grad()
                loss_value.backward()
                optim.step()
                state_grid = state_grid.detach()

    if k % split_rate == 0:
        pass
    else:
        loss_value = loss_fn(target[:, :4, ...], state_grid[:, :4, ...])
        optim.zero_grad()
        loss_value.backward()
        optim.step()
    scheduler.step()

    logging.info(f'{loss_value.item():.2f}, {n_steps} steps, {split_rate} split rate, {epoch} epoch')

    if epoch % 50 == 0:
        output_path = os.path.join(output_folder, f'{epoch}/')
        logging.info(f'writing gif to {output_path}')
        os.makedirs(output_path, exist_ok=True)
        test(policy, perception, dloader_test,
             output_path, num_steps=150,
             stochastic_prob=stochastic_prob)
