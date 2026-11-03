"""Decentralized AMSGrad, Chen et al., Algorithm 3 (PMLR 189).

The denominator mixes OLD tracker values. It is deliberately one iteration
behind the local second moment, including when there is only one client.
"""
import copy

import torch

from dasklearn.optimizer.gt_dsgd import flat


def initial_state(x, epsilon):
    return {'m': torch.zeros_like(x), 'v': torch.zeros_like(x),
            'vhat': torch.full_like(x, epsilon), 'u': torch.full_like(x, epsilon)}


def pack(settings, params):
    x = flat(params['model'])
    u = params['state']['u'] if params['state'] is not None else torch.full_like(x, settings.epsilon)
    return [{'x': x.clone(), 'u': u.clone()}]


def gradient(settings, params):
    from dasklearn.functions import compute_gradient
    model = compute_gradient(settings, dict(params))[0]
    grad = torch.cat([g.detach().flatten().cpu() for g in model.gradient])
    grad = grad + settings.learning.weight_decay * flat(params['model'])
    old = params['state'] if params['state'] is not None else initial_state(grad, settings.epsilon)
    m = settings.beta1 * old['m'] + (1 - settings.beta1) * grad
    v = settings.beta2 * old['v'] + (1 - settings.beta2) * grad.square()
    vhat = torch.maximum(old['vhat'], v)
    local = copy.deepcopy(model)
    del local.gradient
    return [local, {'m': m, 'v': v, 'vhat': vhat}]


def finish(settings, params):
    packets, weights = params['packets'], params['weights']
    mixed_x = sum(w * packet['x'] for w, packet in zip(weights, packets))
    mixed_u = sum(w * packet['u'] for w, packet in zip(weights, packets))
    new = params['local_state']
    previous_vhat = (params['state']['vhat'] if params['state'] is not None
                     else torch.full_like(new['vhat'], settings.epsilon))
    # Clamp only the denominator, not the persistent consensus tracker. Epsilon
    # is a floor in squared-gradient units, not an addition outside the sqrt.
    x = mixed_x - settings.learning.learning_rate * new['m'] / mixed_u.clamp_min(settings.epsilon).sqrt()
    state = {**new, 'u': mixed_u + (new['vhat'] - previous_vhat)}
    model = copy.deepcopy(params['local'])
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            p.copy_(x[offset:offset + p.numel()].view_as(p).to(p.device))
            offset += p.numel()
    return [model, state]
