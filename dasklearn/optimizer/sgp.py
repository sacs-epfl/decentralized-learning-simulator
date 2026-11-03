"""SGP Algorithm 1: gradients at z=x/w, updates to x, then PushSum."""
import copy

import torch

from dasklearn.optimizer.gt_dsgd import flat


def prepare(gradient_model, numerator, learning_rate, weight_decay, weight):
    gradient = torch.cat([g.detach().reshape(-1).cpu() for g in gradient_model.gradient])
    gradient = gradient + weight_decay * flat(gradient_model)
    x = flat(gradient_model) if numerator is None else numerator
    # The gradient is not multiplied by the PushSum mass. Updating z with
    # ordinary SGD and multiplying the result by w would be a different rule.
    packet = (x - learning_rate * gradient) * weight
    local = copy.deepcopy(gradient_model)
    del local.gradient
    return local, packet


def gradient(settings, params):
    from dasklearn.functions import compute_gradient
    model = compute_gradient(settings, dict(params))[0]
    return list(prepare(model, params['numerator'], settings.learning.learning_rate,
                        settings.learning.weight_decay, params['weight']))


def mix(settings, params):
    mass = params['mass']
    if not isinstance(mass, float) or not 0 < mass < float('inf'):
        raise ValueError('SGP requires positive finite PushSum mass')
    numerator = torch.zeros_like(params['packets'][0])
    for packet in params['packets']:
        numerator.add_(packet)
    vector = numerator / mass
    model = copy.deepcopy(params['local_model'])
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            p.copy_(vector[offset:offset + p.numel()].view_as(p).to(p.device))
            offset += p.numel()
    return [model, numerator]
