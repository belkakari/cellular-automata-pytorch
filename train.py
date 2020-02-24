# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import shutil
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.datasets import StateGridSet
from modules.models import SimpleCA
from modules.networks import Perception, Policy
from modules.utils import get_timestamp, set_random_seed, setup_logger

parser = argparse.ArgumentParser(description='Train neural cellular automata')
parser.add_argument('-c', '--config', type=str,
                    help='path to config .yaml')
args = parser.parse_args()
config_path = args.config

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = config['device']
batch_size = config['train']['batch_size']
num_epochs = config['train']['num_epochs']
n_steps_interval = config['n_steps_interval']
split_rate_interval = config['split_rate_interval']
test_frequency = config['test_frequency']
use_coords = config['model']['use_coords']
random_spawn = config['model']['random_spawn']
norm_kernel = config['model']['norm_kernel']
interm_dim = config['model']['interm_dim']
bias = config['model']['bias']

set_random_seed(10)

if (config['experiment_name'] == 'time') or \
   ('experiment_name' not in config.keys()):
    start_time = get_timestamp()
    experiment_name = start_time
else:
    experiment_name = config['experiment_name']

output_folder = os.path.join(config['output_folder'], experiment_name)
os.makedirs(output_folder, exist_ok=True)
shutil.copy(config_path, os.path.join(output_folder, 'config.yaml'))

logging_level = logging.DEBUG if config['logging_level'] == 'DEBUG' else logging.INFO
setup_logger('base', output_folder,
             level=logging_level, screen=True, tofile=True)

logger = logging.getLogger('base')

perception = Perception(channels=16,
                        norm_kernel=norm_kernel).to(device)
policy = Policy(use_embedding=False, kernel=1, padding=0,
                interm_dim=interm_dim, bias=bias).to(device)

model = SimpleCA(perception, policy, config, logger=logger,
                 grad_clip=config['optim']['grad_clip'])

dset = StateGridSet(emoji=config['emoji'], use_coords=use_coords,
                    batch_size=batch_size,
                    random_spawn=random_spawn,
                    pad=50, target_size=128)
dset_test = StateGridSet(emoji=config['emoji'], use_coords=use_coords,
                         batch_size=1,
                         random_spawn=False,
                         pad=50, target_size=128)
dloader = DataLoader(dset, batch_size=batch_size)
dloader_test = DataLoader(dset, batch_size=1)

xv, yv = torch.meshgrid([torch.linspace(-1, 1, steps=dset.target.shape[-1]),
                         torch.linspace(-1, 1, steps=dset.target.shape[-2])])

with torch.autograd.detect_anomaly():
    for epoch in range(num_epochs):
        n_steps = random.randint(*n_steps_interval)
        split_rate = None
        if split_rate_interval:
            split_rate = random.randint(*split_rate_interval)
        for state_grid, target in dloader:
            state_grid, target = state_grid.to(device), target.to(device)
            model.get_input(state_grid, target)
            for k in range(n_steps):
                final_mask = model.forward()
                if split_rate and (k % split_rate == 0):  # truncated bptt
                    loss_value = model.optimize_parameters()
                    state_grid = model.state_grid.detach()
                    model.get_input(state_grid, target)

        if split_rate and (k % split_rate == 0):
            pass
        else:
            loss_value = model.optimize_parameters()

        logger.info(f'{loss_value.item():.2f}, {n_steps} steps, {split_rate} split rate, {epoch} epoch')

        if epoch % test_frequency == 0:
            output_path = os.path.join(output_folder, f'{epoch}/')
            logger.info(f'writing gif to {output_path}')
            os.makedirs(output_path, exist_ok=True)
            topil = transforms.ToPILImage()
            with torch.no_grad():
                for k, (state_grid, target) in enumerate(dloader_test):
                    state_grid, target = state_grid.to(device), target.to(device)
                    topil((target[0].cpu() + 1.) / 2.).save(os.path.join(output_folder,
                                                             f'target.png'))
                    imgs = []
                    masks = []
                    model.get_input(state_grid, target)
                    for _ in range(150):
                        final_mask = model.forward()
                        imgs.append(topil((model.state_grid[0, :4, ...].cpu() + 1.) / 2.))
                        masks.append(topil(final_mask[0, :, ...].cpu()))
                    imgs[0].save(os.path.join(output_path, f'{k}.gif'),
                                 save_all=True, append_images=imgs[1:])
                    masks[0].save(os.path.join(output_path, f'{k}_mask.gif'),
                                  save_all=True, append_images=masks[1:])
