"""Pure chunk-local FedOpt and a specified, measurable moment wire format.

FedAdam follows Reddi et al. Algorithm 2: EMA first/second moments, v0=tau**2,
no bias correction, denominator sqrt(v)+tau. FedAvgM uses unnormalized SGD
momentum (m=beta1*m+g). Local SGD is unchanged; g = base - mean(local models).
Non-parameter state entries retain the original averaged-buffer behavior.
"""
from functools import lru_cache
from math import ceil
import torch


def quantize(value, block_size, unsigned=False):
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Moment encoding requires finite FP32 tensors')
    if unsigned and torch.any(value < 0):
        raise ValueError('Second moments must be nonnegative')
    flat=value.flatten()
    blocks=ceil(flat.numel()/block_size)
    padded=torch.nn.functional.pad(flat,(0,blocks*block_size-flat.numel())).reshape(blocks,block_size)
    bound=255 if unsigned else 127
    scale=padded.abs().amax(1)/bound
    scale=torch.where(scale>0,scale,torch.ones_like(scale))
    values=torch.round(padded/scale[:,None]).clamp(0 if unsigned else -127,bound)
    values=values.flatten()[:flat.numel()].to(torch.uint8 if unsigned else torch.int8)
    return dict(values=values,scales=scale,block_size=block_size)


def dequantize(value):
    if isinstance(value,torch.Tensor):return value
    scales=value['scales'].repeat_interleave(value['block_size'])[:value['values'].numel()]
    return value['values'].float()*scales


@lru_cache(maxsize=8)
def parameter_mask(dataset,model):
    from dasklearn.models import create_model
    with torch.random.fork_rng(devices=[]):
        prototype=create_model(dataset,architecture=model)
    names=set(dict(prototype.named_parameters()))
    return torch.cat([torch.full((v.numel(),),name in names,dtype=torch.bool)
                      for name,v in prototype.state_dict().items()])


def initialize_state(settings,params):
    base=params['base'].float().clone()
    return [dict(x=base,m=torch.zeros_like(base),
        v=torch.full_like(base,settings.server_tau**2) if settings.server_optimizer=='fedadam' else None,
        step=0)]


def update_state(settings,params):
    state=params['state']
    if state['step']+1 != params['round']:
        raise ValueError('Optimizer state must advance once per common logical round')
    average=torch.stack(params['chunks']).mean(0)
    base=state['x']
    gradient=base-average
    previous=dequantize(state['m'])
    if settings.server_optimizer=='fedavgm':
        moment=settings.server_beta1*previous+gradient
        variance=None
        updated=base-settings.server_learning_rate*moment
    elif settings.server_optimizer=='fedadam':
        moment=settings.server_beta1*previous+(1-settings.server_beta1)*gradient
        variance=settings.server_beta2*dequantize(state['v'])+(1-settings.server_beta2)*gradient.square()
        updated=base-settings.server_learning_rate*moment/(variance.sqrt()+settings.server_tau)
    else:
        raise ValueError('Stateful updates require fedavgm or fedadam')
    mask=parameter_mask(settings.dataset,settings.model)
    size=mask.numel()//params['n'];idx=params['chunk_index']
    mask=mask[idx*size:None if idx==params['n']-1 else (idx+1)*size]
    updated=torch.where(mask,updated,average)
    moment=torch.where(mask,moment,torch.zeros_like(moment))
    if variance is not None:variance=torch.where(mask,variance,torch.zeros_like(variance))
    if not torch.isfinite(updated).all():raise ValueError('Nonfinite server optimizer update')
    return [updated,dict(x=updated,m=moment,v=variance,step=params['round'])]


def pack_state(settings,params):
    state=params['state']
    if settings.state_bits==32:return [state]
    return [dict(x=state['x'],m=quantize(state['m'],settings.state_block_size),
        v=quantize(state['v'],settings.state_block_size,unsigned=True) if state['v'] is not None else None,
        step=state['step'])]


def state_payload_bytes(elements,optimizer,bits,block_size):
    """Wire layout: FP32 base, moment values/scales, uint64 logical step.

    Tensor lengths and coding scheme are public configuration/chunk metadata.
    The normal network API separately charges the message header and metadata.
    """
    moments=2 if optimizer=='fedadam' else 1
    per_moment=4*elements if bits==32 else elements+4*ceil(elements/block_size)
    return 4*elements+moments*per_moment+8


def packet_payload_bytes(state):
    total=state['x'].numel()*state['x'].element_size()+8
    for key in ['m','v']:
        value=state[key]
        if value is None:continue
        tensors=[value] if isinstance(value,torch.Tensor) else [value['values'],value['scales']]
        total+=sum(t.numel()*t.element_size() for t in tensors)
    return total
