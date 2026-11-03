"""RelaySum model parcels and RelaySGD's startup-normalized update."""
import copy

import torch

from dasklearn.optimizer.gt_dsgd import flat


def forward(settings, params):
    """Add only the previous-round branches other than the destination branch."""
    result = flat(params['local_model'])
    for packet in params['packets']:
        result.add_(packet)
    return [result]


def finish(settings, params):
    count = 1 + sum(params['counts'])
    if not 1 <= count <= settings.participants:
        raise ValueError('RelaySum population count exceeds the tree population')
    result = flat(params['local_model'])
    for packet in params['packets']:
        result.add_(packet)
    if settings.normalization == 'initial':
        result.add_(flat(params['initial_model']), alpha=settings.participants - count)
        result.div_(settings.participants)
    elif settings.normalization == 'counts':
        result.div_(count)
    else:
        raise ValueError('Unknown RelaySGD startup normalization')
    model = copy.deepcopy(params['local_model'])
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            p.copy_(result[offset:offset + p.numel()].view_as(p).to(p.device))
            offset += p.numel()
    return [model]
