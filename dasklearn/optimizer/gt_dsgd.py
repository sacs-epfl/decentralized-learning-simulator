"""Functional ATC GT-DSGD: Algorithm 1 / Eq. (1) of Xin et al. (2021)."""
import copy

import torch


def flat(model):
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def average(packets, weights):
    result = torch.zeros_like(packets[0])
    for packet, weight in zip(packets, weights):
        result.add_(packet, alpha=weight)
    return result


def prepare(gradient_model, tracker, previous_gradient, weight_decay):
    """Return local buffers, g_k, and the first-phase payload y_k+g_k-g_{k-1}."""
    gradient = torch.cat([g.detach().reshape(-1).cpu() for g in gradient_model.gradient])
    gradient = gradient + weight_decay * flat(gradient_model)
    old_tracker = torch.zeros_like(gradient) if tracker is None else tracker
    old_gradient = torch.zeros_like(gradient) if previous_gradient is None else previous_gradient
    corrected = old_tracker + gradient - old_gradient
    model = copy.deepcopy(gradient_model)
    del model.gradient
    return model, gradient, corrected


def gradient(settings, params):
    from dasklearn.functions import compute_gradient
    model = compute_gradient(settings, dict(params))[0]
    return list(prepare(model, params['tracker'], params['previous_gradient'], settings.learning.weight_decay))


def track(settings, params):
    """First consensus: y_{k+1}=W(y_k+g_k-g_{k-1}); form the proposal."""
    tracker = average(params['packets'], params['weights'])
    proposal = flat(params['local_model']) - settings.learning.learning_rate * tracker
    return [tracker, proposal]


def mix(settings, params):
    """Second consensus: x_{k+1}=W(x_k-lr*y_{k+1}); buffers stay local."""
    vector = average(params['packets'], params['weights'])
    model = copy.deepcopy(params['local_model'])
    with torch.no_grad():
        offset = 0
        for parameter in model.parameters():
            parameter.copy_(vector[offset:offset + parameter.numel()].view_as(parameter).to(parameter.device))
            offset += parameter.numel()
    return [model]
