import io
import os
import logging
from time import gmtime, strftime

import requests
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


def stochastic_update_mask(ds_grid, prob=0.5):
    # Generate mask for zero out a random fraction of the updates.
    bern = torch.distributions.Bernoulli(prob)
    rand_mask = bern.sample((ds_grid.shape[2] * ds_grid.shape[3],))
    rand_mask = rand_mask.view(ds_grid.shape[2:]).float()
    return rand_mask.to(ds_grid.device)[None, None]


def alive_mask(state_grid, thr=0.1):
    # Take the alpha channel as the measure of “life”.
    alive = (state_grid[:, 4, :, :].clamp(0, 1) > thr).float().unsqueeze(1)
    return alive


def load_image(url, max_size=128):
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), Image.ANTIALIAS)
    img = np.float32(img) / 255.
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = f'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u{code}.png'
    return load_image(url)


def test(policy, perception, dloader_test,
         output_path, num_steps=150, stochastic_prob=0.1):
    imgs = []
    topil = transforms.ToPILImage()
    with torch.no_grad():
        for k, (state_grid, _) in enumerate(dloader_test):
            for _ in range(num_steps):
                imgs.append(topil(state_grid[0, :4, ...].cpu()))
                alive_pre = alive_mask((state_grid + 1.) / 2., thr=0.1)
                perception_grid = perception(state_grid)
                ds_grid = policy(perception_grid)
                mask = stochastic_update_mask(ds_grid,
                                              prob=stochastic_prob)
                state_grid = state_grid + ds_grid * mask
                alive_post = alive_mask((state_grid + 1.) / 2., thr=0.1)
                final_mask = (alive_post.bool() & alive_pre.bool()).float()
                state_grid = state_grid * final_mask
            imgs.append(topil(state_grid[0, :4, ...].cpu()))
            imgs[0].save(os.path.join(output_path, f'{k}.gif'),
                         save_all=True, append_images=imgs[1:])


def get_timestamp():
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())


def setup_logger(logger_name, root, level=logging.INFO,
                 screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, f'_{get_timestamp()}.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
