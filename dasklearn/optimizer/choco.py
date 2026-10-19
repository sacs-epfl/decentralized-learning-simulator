"""Functional CHOCO-SGD, ICML 2019 Algorithm 2.

Each worker stores its own public estimate and a weighted sum of public
estimates, requiring two parameter vectors independent of node degree.
Compressor operations apply independently to each trainable parameter tensor.
"""
import copy

import torch


def parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def sizes(model):
    return [p.numel() for p in parameters(model)]


def kept_count(size, ratio):
    return max(1, int(size * ratio))


def payload_bytes(model, compressor, ratio):
    params = parameters(model)
    if compressor == 'identity':
        return sum(p.numel() * p.element_size() for p in params)
    # Explicit int64 coordinates plus values. No float-encoded index shortcuts.
    return sum(kept_count(p.numel(), ratio) * (8 + p.element_size()) for p in params)


def compress(vector, tensor_sizes, compressor, ratio, seed):
    if compressor == 'identity':
        return {'values': vector.clone(), 'indices': None}
    generator = torch.Generator(device='cpu').manual_seed(seed)
    indices = []
    offset = 0
    for size in tensor_sizes:
        count = kept_count(size, ratio)
        values = vector[offset:offset + size]
        if compressor == 'topk':
            selected = (values.abs().argmax().reshape(1) if count == 1 else
                        values.abs().topk(count, sorted=False).indices)
        elif compressor == 'randomk':
            selected = torch.randperm(size, generator=generator)[:count]
        else:
            raise ValueError('Unknown CHOCO compressor: ' + compressor)
        indices.append(selected + offset)
        offset += size
    indices = torch.cat(indices)
    return {'values': vector[indices].clone(), 'indices': indices}


def decompress(packet, size):
    if packet['indices'] is None:
        return packet['values'].clone()
    result = packet['values'].new_zeros(size)
    result[packet['indices']] = packet['values']
    return result


def prepare(model, state, compressor, ratio, gamma, seed):
    """Correct the locally trained model with OLD hats, then compress its delta."""
    result = copy.deepcopy(model)
    params = parameters(result)
    flat = torch.cat([p.detach().reshape(-1).cpu() for p in params])
    own = torch.zeros_like(flat) if state is None else state['own']
    memory = torch.zeros_like(flat) if state is None else state['memory']
    flat = flat + gamma * (memory - own)
    with torch.no_grad():
        offset = 0
        for param in params:
            param.copy_(flat[offset:offset + param.numel()].reshape(param.shape).to(param.device))
            offset += param.numel()
    packet = compress(flat - own, [p.numel() for p in params], compressor, ratio, seed)
    return result, packet


def receive(state, packets, weights, size):
    """First packet is this worker's packet; weights include the diagonal W_ii."""
    own_delta = decompress(packets[0], size)
    own = torch.zeros_like(own_delta) if state is None else state['own']
    memory = torch.zeros_like(own_delta) if state is None else state['memory']
    updated_memory = memory.clone()
    for packet, weight in zip(packets, weights):
        updated_memory.add_(decompress(packet, size), alpha=weight)
    return {'own': own + own_delta, 'memory': updated_memory}
