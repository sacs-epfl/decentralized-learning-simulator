"""Functional ICML 2021 QG-DSGDm and persistent local-momentum controls."""
import copy
import torch


def flat(model):
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def replace(model, vector):
    result=copy.deepcopy(model)
    if hasattr(result,'gradient'):
        del result.gradient
    with torch.no_grad():
        offset=0
        for p in result.parameters():
            p.copy_(vector[offset:offset+p.numel()].reshape_as(p).to(p.device));offset+=p.numel()
    return result


def prepare(gradient_model, previous, lr, beta, weight_decay, nesterov, mode):
    base=flat(gradient_model)
    gradient=torch.cat([g.detach().reshape(-1).cpu() for g in gradient_model.gradient])
    gradient=gradient+weight_decay*base
    beta=0. if mode=='plain' else beta
    previous=torch.zeros_like(base) if previous is None else previous
    local=beta*previous+gradient
    direction=gradient+beta*local if nesterov and mode!='plain' else local
    packet=base-lr*direction
    # Only the parameter vector is sent. Buffers follow the local trainer.
    return replace(gradient_model,packet),local,packet


def finish(base, local_model, previous, local_momentum, packets, weights, lr, mu, mode):
    mixed=torch.zeros_like(packets[0])
    for packet,weight in zip(packets,weights):
        mixed.add_(packet,alpha=weight)
    model=replace(local_model,mixed)
    if mode=='quasi':
        old=torch.zeros_like(mixed) if previous is None else previous
        state=mu*old+(1-mu)*(flat(base)-mixed)/lr
    elif mode=='local':
        state=local_momentum.clone()
    else:
        state=None
    return model,state
