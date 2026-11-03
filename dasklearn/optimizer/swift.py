"""Functional SWIFT update: average pre-step models, then subtract old gradient."""
import copy

import torch


def pack(settings, params):
    return [torch.cat([p.detach().reshape(-1).cpu() for p in params['model'].parameters()])]


def update(settings, params):
    source = params['gradient_model']
    result = copy.deepcopy(source)
    del result.gradient
    packets = params['packets']
    weights = params['weights']
    self_weight = 1 - sum(weights)
    lr = settings.learning.learning_rate
    decay = settings.learning.weight_decay
    with torch.no_grad():
        offset = 0
        for target, base, gradient in zip(result.parameters(), source.parameters(), source.gradient):
            target.copy_(base * self_weight)
            for packet, weight in zip(packets, weights):
                value = packet[offset:offset + base.numel()].view_as(base).to(target.device)
                target.add_(value, alpha=weight)
            target.add_(gradient.to(target.device), alpha=-lr)
            if decay:
                target.add_(base, alpha=-lr * decay)
            offset += base.numel()
    # Buffers (e.g. BatchNorm statistics) follow this client's gradient pass.
    return [result]
