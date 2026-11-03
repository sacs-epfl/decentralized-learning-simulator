"""Base MoTEF, Algorithm 1: compressed estimates of models and momentum trackers."""
import copy
import math

import torch

from dasklearn.optimizer.choco import decompress
from dasklearn.optimizer.gt_dsgd import flat


def kept_count(size, ratio):
    return max(1, math.ceil(size * ratio))


def compress(vector, compressor, ratio, seed):
    """Contractive compression over the whole parameter vector, without scaling."""
    if compressor == 'identity':
        return {'indices': None, 'values': vector.clone()}
    k = kept_count(vector.numel(), ratio)
    if compressor == 'topk':
        indices = vector.abs().topk(k, sorted=False).indices
    elif compressor == 'randomk':
        indices = torch.randperm(vector.numel(), generator=torch.Generator().manual_seed(seed))[:k]
    else:
        raise ValueError('Unknown MoTEF compressor')
    return {'indices': indices, 'values': vector[indices].clone()}


def seed(settings, params, channel):
    return (settings.seed + 1000003 * params['peer'] + 9176 * params['round'] + channel) % 2**63


def oracle(settings, params):
    from dasklearn.functions import compute_gradient
    model = compute_gradient(settings, dict(params))[0]
    grad = torch.cat([g.detach().flatten().cpu() for g in model.gradient])
    grad = grad + settings.learning.weight_decay * flat(model)
    local = copy.deepcopy(model)
    del local.gradient
    return local, grad


def bootstrap(settings, params):
    model, grad = oracle(settings, params)
    # V0=M0 is essential for mean(V)=mean(M). H0=G0=0 are known to every
    # client without a free exchange of initial client-specific gradients.
    state = {key: torch.zeros_like(grad) for key in ['h', 'g', 'h_memory', 'g_memory']}
    state.update(m=grad.clone(), v=grad.clone())
    return [model, state]


def propose(settings, params):
    state = params['state']
    x = flat(params['model']) + settings.consensus_stepsize * (state['h_memory'] - state['h'])
    x = x - settings.learning.learning_rate * state['v']
    model = copy.deepcopy(params['model'])
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            p.copy_(x[offset:offset + p.numel()].view_as(p).to(p.device))
            offset += p.numel()
    qh = compress(x - state['h'], settings.compressor, settings.keep_ratio, seed(settings, params, 13))
    return [model, qh]


def gradient(settings, params):
    model, grad = oracle(settings, params)
    state = params['state']
    m = (1 - settings.momentum_weight) * state['m'] + settings.momentum_weight * grad
    v = state['v'] + settings.consensus_stepsize * (state['g_memory'] - state['g']) + m - state['m']
    qg = compress(v - state['g'], settings.compressor, settings.keep_ratio, seed(settings, params, 29))
    return [model, {'m': m, 'v': v}, {'model': params['model_packet'], 'tracker': qg}]


def receive(settings, params):
    old = params['state']
    size = old['h'].numel()
    result = dict(params['local_state'])
    for channel, own_key, memory_key in [('model', 'h', 'h_memory'), ('tracker', 'g', 'g_memory')]:
        own_delta = decompress(params['packets'][0][channel], size)
        result[own_key] = old[own_key] + own_delta
        memory = old[memory_key].clone()
        for packet, weight in zip(params['packets'], params['weights']):
            memory.add_(decompress(packet[channel], size), alpha=weight)
        result[memory_key] = memory
    return [result]
