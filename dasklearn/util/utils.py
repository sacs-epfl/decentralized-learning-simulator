import os
import random

import numpy as np

import torch


def conditional_value(var, nul, default):
    """
    Set the value to default if nul.

    Parameters
    ----------
    var : any
        The value
    nul : any
        The null value. Assigns default if var == nul
    default : any
        The default value

    Returns
    -------
    type(var)
        The final value

    """
    if var != nul:
        return var
    else:
        return default


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_random_hex_str(length: int) -> str:
    return ''.join(random.choice('0123456789abcdef') for n in range(length))


def start_profile():
    import yappi
    yappi.start(builtins=True)


def stop_profile(data_dir: str, is_broker=False):
    import yappi
    yappi.stop()
    yappi_stats = yappi.get_func_stats()
    yappi_stats.sort("tsub")
    file_name: str = "yappi.stats" if not is_broker else "yappi_broker.stats"
    yappi_stats.save(os.path.join(data_dir, file_name), type='callgrind')
