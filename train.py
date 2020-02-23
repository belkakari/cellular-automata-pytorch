# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.datasets import StateGridSet
from modules.networks import Perception, Policy
from modules.utils import (alive_mask, get_timestamp, load_emoji, setup_logger,
                           stochastic_update_mask, test)

parser = argparse.ArgumentParser(description='Train neural cellular automata')
parser.add_argument('-c', '--config', type=str,
                    help='path to config .yaml')
args = parser.parse_args()
config_path = args.config

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = config['device']
stochastic_prob = config['stochastic_prob']
batch_size = config['train']['batch_size']
num_epochs = config['train']['num_epochs']
n_steps_interval = config['n_steps_interval']
split_rate_interval = config['split_rate_interval']
test_frequency = config['test_frequency']
use_coords = config['use_coords']
random_spawn = config['random_spawn']

if (config['experiment_name'] == 'time') or \
   ('experiment_name' not in config.keys()):
    start_time = get_timestamp()
    experiment_name = start_time
else:
    experiment_name = config['experiment_name']

output_folder = os.path.join(config['output_folder'], experiment_name)
os.makedirs(output_folder, exist_ok=True)
shutil.copy(config_path, os.path.join(output_folder, 'config.yaml'))

setup_logger('base', output_folder,
             level=logging.INFO, screen=True, tofile=True)

logger = logging.getLogger('base')

img = load_emoji(emoji='🦎')
img = transforms.ToTensor()(img)
img = transforms.Normalize(tuple(0.5 for _ in range(img.shape[0])),
                           tuple(0.5 for _ in range(img.shape[0])))(img)
img = img.to(device)

policy = Policy(use_embedding=False, kernel=1, padding=0).to(device)
perception = Perception(channels=16).to(device)
optim = torch.optim.Adam(list(policy.parameters()) +
                         list(perception.parameters()), lr=config['optim']['lr'])
scheduler = lr_scheduler.MultiStepLR(optim,
                                     config['optim']['milestones'],
                                     gamma=config['optim']['gamma'])
loss_fn = nn.MSELoss()

dset = StateGridSet(img, use_coords=use_coords,
                    batch_size=batch_size,
                    random_spawn=random_spawn)
dset_test = StateGridSet(img, use_coords=use_coords,
                         batch_size=1,
                         random_spawn=False)
dloader = DataLoader(dset, batch_size=batch_size)
dloader_test = DataLoader(dset, batch_size=1)

xv, yv = torch.meshgrid([torch.linspace(-1, 1, steps=img.shape[-1]),
                         torch.linspace(-1, 1, steps=img.shape[-2])])

for epoch in range(num_epochs):
    n_steps = random.randint(*n_steps_interval)
    split_rate = random.randint(*split_rate_interval)
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

    logger.info(f'{loss_value.item():.2f}, {n_steps} steps, {split_rate} split rate, {epoch} epoch')

    if epoch % test_frequency == 0:
        output_path = os.path.join(output_folder, f'{epoch}/')
        logger.info(f'writing gif to {output_path}')
        os.makedirs(output_path, exist_ok=True)
        test(policy, perception, dloader_test,
             output_path, num_steps=150,
             stochastic_prob=stochastic_prob)
