import io
import logging
import os
import random
from time import gmtime, strftime

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def stochastic_update_mask(ds_grid, prob=0.5):
    # Generate mask for zero out a random fraction of the updates.
    bern = torch.distributions.Bernoulli(prob)
    rand_mask = bern.sample((ds_grid.shape[2] * ds_grid.shape[3],))
    rand_mask = rand_mask.view(ds_grid.shape[2:]).float()
    return rand_mask.to(ds_grid.device)[None, None]


def alive_mask(state_grid, thr=0.1):
    # Take the alpha channel as the measure of “life”.
    alpha = state_grid[:, [3], :, :].clamp(0, 1)
    alive = (nn.MaxPool2d(3, stride=1, padding=1)(alpha) > thr).float()#.unsqueeze(1)
    return alive


def load_image(url, max_size=128):
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    #img.thumbnail((max_size, max_size), Image.ANTIALIAS)
    img = np.float32(img) / 255.
    ## premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = f'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u{code}.png'
    return load_image(url)


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
