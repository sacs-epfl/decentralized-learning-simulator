"""Compare base MoTEF with two pinned reference recurrences.

Usage: python tests/check_motef_reference.py /path/to/MoTEF
Execute BEER_Momentum.update unchanged with declared initial state and the
authors' synthetic magnitude-Top-k helper, then also execute synthetic beerM.
No reference deployment, main-init or signed-order compressor equivalence is
claimed. Reference numerical code is loaded from the supplied checkout.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasklearn import functions
from dasklearn.optimizer.gt_dsgd import flat
from dasklearn.optimizer.motef import bootstrap, propose, gradient, receive

PIN = '10919bd38e093897329d1b096ba037701084d635'


def check(root):
    root = Path(root)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError('Unexpected reference revision: ' + revision)
    paths = [root/'MoTEF.py', root/'synthetic/solvers.py', root/'synthetic/utils.py']
    cls = next(n for n in ast.parse(paths[0].read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'BEER_Momentum')
    update = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update')
    solver = next(n for n in ast.parse(paths[1].read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'beerM')
    helpers = [n for n in ast.parse(paths[2].read_text()).body if getattr(n, 'name', '') in
               {'topK', 'stoch_gradient', 'consensus_distance', 'argmin_f_noinv'}]
    scope = {'np': np, 'xp': np}
    exec(compile(ast.Module(body=helpers + [update, solver], type_ignores=[]), '<pinned MoTEF reference>', 'exec'), scope)
    torch.manual_seed(73)
    template = torch.nn.Linear(3, 2).double()
    maximum, solver_maximum = 0., 0.
    old_oracle = functions.compute_gradient
    try:
        for topology in ['ring', 'complete']:
            W = np.array([[1 / (3 if topology == 'ring' else 4) if topology == 'complete' or (i-j) % 4 in [0, 1, 3]
                           else 0. for j in range(4)] for i in range(4)])
            # Grad_i(x)=A_i^T(A_i*x-B_i), matching the actual synthetic solver.
            A = np.array([np.eye(8) * (i + 1) / 2 for i in range(4)])
            B = np.arange(32).reshape(4, 8) / 17 + .13
            def oracle(s, params):
                model = copy.deepcopy(params['model'])
                i = params['peer']
                values = A[i].T @ (A[i] @ flat(model).numpy() - B[i])
                grad = torch.from_numpy(values)
                model.gradient = [grad[:6].reshape(2, 3), grad[6:]]
                return [model]
            functions.compute_gradient = oracle
            for compressor in ['identity', 'topk']:
                for momentum in [.2, 1.]:
                    settings = SimpleNamespace(seed=19, compressor=compressor, keep_ratio=.5,
                        consensus_stepsize=.2, momentum_weight=momentum,
                        learning=SimpleNamespace(learning_rate=.03, weight_decay=0))
                    models, states = [], []
                    initial = np.stack([flat(template).numpy()] * 4, axis=1)
                    initial_grad = scope['stoch_gradient'](initial, A, B, 0.).T
                    compressor_function = scope['topK'] if compressor == 'topk' else lambda x, ratio: x.copy()
                    ref = SimpleNamespace(x=initial.copy(), H=np.zeros_like(initial), G=np.zeros_like(initial),
                        V=initial_grad.copy(), M=initial_grad.copy(), M_previous=np.zeros_like(initial),
                        W_shifted=W-np.eye(4), eta=.03, gamma=.2, lmbd=momentum, batch_size=1,
                        comm_rounds=0, p=SimpleNamespace(n_agent=4, m=1), compressor_param=.5,
                        C=lambda x, ratio: np.apply_along_axis(lambda col: compressor_function(col, ratio), 0, x),
                        grad=lambda x, **kwargs: scope['stoch_gradient'](x, A, B, 0.).T)
                    for i in range(4):
                        model, state = bootstrap(settings, {'model': template, 'peer': i, 'round': 0})
                        models.append(model)
                        states.append(state)
                    for r in range(1, 9):
                        partials, packets = [], []
                        old_h, old_g = ref.H.copy(), ref.G.copy()
                        scope['update'](ref)
                        for i in range(4):
                            params = {'model': models[i], 'state': states[i], 'peer': i, 'round': r}
                            model, qh = propose(settings, params)
                            models[i], partial, packet = gradient(settings, {**params, 'model': model, 'model_packet': qh})
                            partials.append(partial)
                            packets.append(packet)
                        for i in range(4):
                            peers = [i] + [j for j in range(4) if j != i and W[i, j]]
                            states[i] = receive(settings, {'state': states[i], 'local_state': partials[i],
                                'packets': [packets[j] for j in peers], 'weights': [W[i, j] for j in peers]})[0]
                            comparisons = [(flat(models[i]).numpy(), ref.x[:, i])]
                            for key, expected in [('h', ref.H), ('g', ref.G), ('m', ref.M), ('v', ref.V),
                                                  ('h_memory', ref.H @ W), ('g_memory', ref.G @ W)]:
                                comparisons.append((states[i][key].numpy(), expected[:, i]))
                            for channel, expected in [('model', ref.H-old_h), ('tracker', ref.G-old_g)]:
                                packet = packets[i][channel]
                                decoded = np.zeros(8)
                                if packet['indices'] is None:
                                    decoded = packet['values'].numpy()
                                else:
                                    decoded[packet['indices'].numpy()] = packet['values'].numpy()
                                comparisons.append((decoded, expected[:, i]))
                            for actual, expected in comparisons:
                                maximum = max(maximum, float(np.max(np.abs(actual-expected))))
                                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
                    _, expected = scope['beerM'](initial, W, A, B, compressor_function, .5, .2, .03,
                                                 momentum, 0., num_iter=8)
                    actual = np.stack([flat(model).numpy() for model in models], axis=1)
                    solver_maximum = max(solver_maximum, float(np.max(np.abs(actual-expected))))
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
        print(json.dumps({'passed': True, 'reference_commit': revision, 'workers': 4, 'iterations': 8,
            'topologies': ['ring', 'complete'], 'compressors': ['identity', 'magnitude-topk'],
            'momentum_weights': [.2, 1.], 'max_state_or_packet_error': maximum,
            'max_synthetic_solver_parameter_error': solver_maximum,
            'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'scope': 'Actual BEER_Momentum.update recurrence, actual synthetic topK, and actual synthetic beerM solver; H0=G0=0, V0=M0=local gradient; deterministic non-IID quadratics; excludes main optimizer init, signed-order main compressor and deployment'}))
    finally:
        functions.compute_gradient = old_oracle


if __name__ == '__main__':
    check(sys.argv[1])
