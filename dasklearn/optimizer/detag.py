"""DeTAG Algorithms 2/3; normalized gradient buffer as in JMLR 2023."""
import copy

import torch

from dasklearn.optimizer.gt_dsgd import flat, average


def packet(settings, params):
    vector = flat(params['model'])
    tracker = params['tracker']
    return [torch.stack([vector, torch.zeros_like(vector) if tracker is None else tracker])]


def gradient(settings, params):
    from dasklearn.functions import compute_gradient
    local = compute_gradient(settings, dict(params))[0]
    values = torch.cat([g.detach().reshape(-1).cpu() for g in local.gradient])
    values = values + settings.learning.weight_decay * flat(local)
    values = values / settings.phase_length
    accumulator = params['accumulator']
    if accumulator is not None:
        values = accumulator + values
    del local.gradient
    # Parameters remain fixed; only local non-parameter buffers follow forwards.
    return [local, values]


def gossip(settings, params):
    mixed = average(params['packets'], params['weights'])
    return [(1 + settings.eta) * mixed - settings.eta * params['previous']]


def finish(settings, params):
    mixed = params['packet']
    vector = mixed[0] - settings.learning.learning_rate * mixed[1]
    old = params['previous_gradient']
    tracker = mixed[1] + params['gradient'] - (torch.zeros_like(vector) if old is None else old)
    model = copy.deepcopy(params['local_model'])
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            p.copy_(vector[offset:offset + p.numel()].view_as(p).to(p.device))
            offset += p.numel()
    return [model, tracker]
