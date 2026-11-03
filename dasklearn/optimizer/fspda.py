"""FSPDA-SA: simultaneous descent/ascent, paper (7), Appendix A (35)-(36)."""
import copy
import hashlib
import math

import torch

from dasklearn.optimizer.gt_dsgd import flat


def edge_seed(seed, r, i, j, channel):
    """Identical at both endpoints, independent of event order and process hash."""
    key = f'{seed}:{r}:{min(i, j)}:{max(i, j)}:{channel}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % 2**63


def kept_count(size, ratio):
    return max(1, math.ceil(size * ratio))


def pack(settings, params):
    x = flat(params['model'])
    indices = None
    if settings.keep_ratio < 1:
        seed = edge_seed(settings.seed, params['round'], params['peer'], params['neighbor'], 'coordinates')
        indices = torch.randperm(x.numel(), generator=torch.Generator().manual_seed(seed))[:kept_count(x.numel(), settings.keep_ratio)]
    return [{'indices': indices, 'values': x.clone() if indices is None else x[indices].clone()}]


def gradient(settings, params):
    from dasklearn.functions import compute_gradient
    model = compute_gradient(settings, dict(params))[0]
    grad = torch.cat([g.detach().flatten().cpu() for g in model.gradient])
    grad = grad + settings.learning.weight_decay * flat(params['model'])
    local = copy.deepcopy(model)
    del local.gradient
    return [local, grad]


def finish(settings, params):
    # Positive graph Laplacian: both recurrences use the OLD x, never the
    # locally updated x against old neighbor packets. Symmetric masks imply
    # sum_i laplacian_i = 0 and preserve sum_i dual_i = 0 from zero startup.
    x = flat(params['base'])
    dual = params['dual'] if params['dual'] is not None else torch.zeros_like(x)
    laplacian = torch.zeros_like(x)
    for packet in params['packets']:
        indices = packet['indices']
        if indices is None:
            laplacian.add_(x - packet['values'])
        else:
            laplacian[indices] += x[indices] - packet['values']
    next_x = x - settings.learning.learning_rate * params['gradient']
    next_x = next_x - settings.dual_weight * dual - settings.consensus_stepsize * laplacian
    next_dual = dual + settings.dual_stepsize * laplacian
    model = copy.deepcopy(params['local'])
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            p.copy_(next_x[offset:offset + p.numel()].view_as(p).to(p.device))
            offset += p.numel()
    return [model, next_dual]
