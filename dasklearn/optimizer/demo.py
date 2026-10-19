"""DeMo's reference Signum update, with immutable per-worker error feedback.

Matches bloc97/DeMo demo.py: delta = decay*delta + lr*grad, remove the
locally transmitted reconstruction, mean *only transmitted* colliding DCT
coefficients, inverse transform, sign update and decoupled weight decay.
Conv kernels are represented as (out_channels, -1) matrices before blocking.
"""
import math
from functools import lru_cache
import torch


def block_shape(shape, chunk):
    shape = tuple(shape) if len(shape) <= 2 else (shape[0], math.prod(shape[1:]))
    if not shape:
        shape = (1,)
    splits = tuple(max(d for d in range(1, min(n, chunk) + 1) if n % d == 0) for n in shape)
    return shape, splits


@lru_cache(maxsize=128)
def basis(n, dtype):
    # Orthonormal DCT-II; rows are spatial positions, columns frequencies.
    positions = torch.arange(n, dtype=torch.float64)[:, None]
    frequencies = torch.arange(n, dtype=torch.float64)[None, :]
    matrix = torch.cos(math.pi / n * (positions + .5) * frequencies) * math.sqrt(2 / n)
    matrix[:, 0] /= math.sqrt(2)
    return matrix.to(dtype)


def encode(tensor, chunk):
    shape, splits = block_shape(tensor.shape, chunk)
    x = tensor.reshape(shape)
    if len(shape) == 1:
        return x.reshape(-1, splits[0]) @ basis(splits[0], x.dtype).to(x.device)
    a, b = splits
    x = x.reshape(shape[0] // a, a, shape[1] // b, b)
    return torch.einsum('yhxw,hb,wd->yxbd', x, basis(a, x.dtype).to(x.device), basis(b, x.dtype).to(x.device))


def decode(coefficients, original_shape, chunk):
    shape, splits = block_shape(original_shape, chunk)
    if len(shape) == 1:
        x = coefficients @ basis(splits[0], coefficients.dtype).to(coefficients.device).T
    else:
        a, b = splits
        x = torch.einsum('yxbd,hb,wd->yhxw', coefficients,
                         basis(a, coefficients.dtype).to(coefficients.device), basis(b, coefficients.dtype).to(coefficients.device))
    return x.reshape(original_shape)


def compress(tensor, chunk, topk):
    coefficients = encode(tensor, chunk)
    flat = coefficients.flatten(-2) if coefficients.ndim == 4 else coefficients
    indices = flat.abs().topk(min(topk, flat.shape[-1]), dim=-1, sorted=False).indices
    return indices, flat.gather(-1, indices)


def decompress(packets, shape, chunk):
    """Reference scatter_reduce(mean, include_self=False), including collisions.

CPU ordered accumulation avoids CUDA atomic nondeterminism. No division by
world size for coefficients absent from some workers (the released code's rule).
"""
    logical, splits = block_shape(shape, chunk)
    prefix = tuple(n // s for n, s in zip(logical, splits))
    count = math.prod(splits)
    values = torch.zeros((*prefix, count), dtype=packets[0][1].dtype)
    counts = torch.zeros_like(values)
    for indices, coefficients in packets:
        indices, coefficients = indices.cpu(), coefficients.cpu()
        values.scatter_add_(-1, indices, coefficients)
        counts.scatter_add_(-1, indices, torch.ones_like(coefficients))
    values /= counts.clamp_min_(1)
    return decode(values.reshape(*prefix, *splits), shape, chunk)


def packet_bytes(shapes, chunk=64, topk=32, value_bytes=4):
    """Actual int64 indices + parameter-dtype values; headers charged separately."""
    size = 0
    for original in shapes:
        shape, splits = block_shape(original, chunk)
        blocks = math.prod(n // s for n, s in zip(shape, splits))
        size += blocks * min(topk, math.prod(splits)) * (8 + value_bytes)
    return size


@torch.no_grad()
def prepare(gradients, previous, lr, decay, chunk, topk):
    residuals, packet = [], []
    for index, gradient in enumerate(gradients):
        delta = gradient.detach().cpu() * lr
        if previous is not None:
            delta = delta + previous[index] * decay
        message = compress(delta, chunk, topk)
        residuals.append(delta - decompress([message], delta.shape, chunk))
        packet.append(message)
    return packet, residuals


@torch.no_grad()
def update(model, packets, lr, weight_decay, chunk):
    import copy
    result = copy.deepcopy(model).cpu()
    for index, parameter in enumerate(result.parameters()):
        if parameter.requires_grad:
            gradient = decompress([packet[index] for packet in packets], parameter.shape, chunk)
            parameter.mul_(1 - lr * weight_decay).add_(gradient.sign(), alpha=-lr)
    return result
