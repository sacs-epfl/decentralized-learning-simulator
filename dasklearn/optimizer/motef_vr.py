"""MoTEF-VR (Algorithm 2): two fresh, coupled stochastic gradients per step."""
import copy
import random
from contextlib import contextmanager

import numpy as np
import torch

from dasklearn.optimizer.gt_dsgd import flat
from dasklearn.optimizer.motef import compress, seed


@contextmanager
def stochastic_context(settings, task_seed):
    """Isolate all supported RNGs, including the selected accelerator stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(settings.torch_device_name)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(task_seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(task_seed)
            random.seed(task_seed)
            np.random.seed(task_seed % 2**32)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def paired_oracle(settings, params, previous_model=None):
    """Materialize a fresh transformed minibatch ONCE, then reuse it.

    Both passes start from the same forward RNG state, and independent copies
    of the old local buffers. Only the next-model pass's buffers survive.
    No previous minibatch gradient is stored or reused.
    """
    from dasklearn import functions
    from dasklearn.model_trainer import ModelTrainer
    if functions.dataset is None:
        functions.dataset = functions.create_dataset(settings)
    with functions.lock:
        if functions.model_trainers is None:
            functions.model_trainers = [None] * settings.participants
    peer = params['peer']
    if functions.model_trainers[peer] is None:
        functions.model_trainers[peer] = ModelTrainer(functions.dataset, settings, peer)
    trainer = functions.model_trainers[peer]
    task_seed = functions.training_seed(settings, params)
    gradients, models = [], []
    device = torch.device(settings.torch_device_name)
    with stochastic_context(settings, task_seed):
        batch = next(iter(trainer.make_train_loader(task_seed)))
        # Replay the state AFTER sampling/transforms, so the first forward has
        # the ordinary sampling-then-forward stream, without restarting it.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        for source in [params['model']] + ([] if previous_model is None else [previous_model]):
            local = copy.deepcopy(source)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
            trainer.train(local, 1, True, settings.torch_device_name,
                          seed=task_seed, prepared_batch=copy.deepcopy(batch))
            grad = torch.cat([g.detach().reshape(-1).cpu() for g in local.gradient])
            gradients.append(grad + settings.learning.weight_decay * flat(local))
            del local.gradient
            # Do not retain backward workspaces in immutable DAG models.
            local.zero_grad(set_to_none=True)
            models.append(local)
    return models[0], gradients


def bootstrap(settings, params):
    model, (grad,) = paired_oracle(settings, params)
    state = {key: torch.zeros_like(grad) for key in ['h', 'g', 'h_memory', 'g_memory']}
    state.update(m=grad.clone(), v=grad.clone())
    return [model, state]


def gradient(settings, params):
    model, (new_gradient, old_gradient) = paired_oracle(settings, params, params['previous_model'])
    state = params['state']
    m = new_gradient + (1 - settings.momentum_weight) * (state['m'] - old_gradient)
    v = state['v'] + settings.consensus_stepsize * (state['g_memory'] - state['g']) + m - state['m']
    qg = compress(v - state['g'], settings.compressor, settings.keep_ratio, seed(settings, params, 29))
    return [model, {'m': m, 'v': v}, {'model': params['model_packet'], 'tracker': qg}]
