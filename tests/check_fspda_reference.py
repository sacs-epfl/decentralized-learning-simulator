"""Audit the pinned authors' step, including its different dual update order.

python tests/check_fspda_reference.py /path/to/FSPDA

Executes unchanged AST bodies of step, parameter/dual helpers, TensorBuffer,
flatten, apply_gradient and join_thread. Communication is replaced with known
old-model packets; this checks no distributed deployment or random handshake.
This is a discrepancy audit, not full reference equivalence.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import defaultdict
from types import MethodType, SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasklearn.optimizer.fspda import finish, pack
from dasklearn.optimizer.gt_dsgd import flat

PIN = '0808887c6e9f713ca203df3138fb8995094be694'


def check(root):
    root = Path(root)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError('Unexpected reference revision: ' + revision)
    paths = [root/'pcode/optim/fspda.py', root/'pcode/optim/utils.py',
             root/'pcode/utils/tensor_buffer.py', root/'pcode/utils/communication.py']
    trees = [ast.parse(p.read_text()) for p in paths]
    cls = next(n for n in trees[0].body if isinstance(n, ast.ClassDef) and n.name == 'FSPDA')
    methods = [n for n in cls.body if getattr(n, 'name', '') in
               {'step', 'get_lambda', 'get_prm', 'get_zeros_prm_buffer', 'update_eta'}]
    helpers = [n for n in trees[1].body if getattr(n, 'name', '') in {'apply_gradient', 'join_thread'}]
    buffer = next(n for n in trees[2].body if isinstance(n, ast.ClassDef) and n.name == 'TensorBuffer')
    flatten = next(n for n in trees[3].body if isinstance(n, ast.FunctionDef) and n.name == 'flatten')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[flatten, buffer] + helpers + methods, type_ignores=[]), '<pinned FSPDA audit>', 'exec'), scope)
    scope['utils'] = SimpleNamespace(**{n.name: scope[n.name] for n in helpers})
    torch.manual_seed(92)
    models = [torch.nn.Linear(3, 2).double() for _ in range(4)]
    primal_error, order_error, dual_gap, ref_dual_sum = 0., 0., 0., 0.
    cases = 0
    for ratio in [.5, 1.]:
        settings = SimpleNamespace(seed=7, keep_ratio=ratio, consensus_stepsize=.2,
            dual_weight=.03, dual_stepsize=.4, learning=SimpleNamespace(learning_rate=.1))
        duals = torch.randn(4, 8, dtype=torch.float64)
        duals -= duals.mean(0)
        for r in range(1, 4):
            ref_next, actual_next = [], []
            for i in range(4):
                old = models[i]
                peers = [(i-1) % 4, (i+1) % 4]
                grad = flat(old) * (i + 1) - .2
                packets = {j: pack(settings, dict(model=models[j], peer=j, neighbor=i, round=r))[0] for j in peers}
                actual_model, actual_dual = finish(settings, dict(base=old, local=old, dual=duals[i],
                    gradient=grad, packets=list(packets.values())))
                ref_model = copy.deepcopy(old)
                groups, offset = [], 0
                for p in ref_model.parameters():
                    p.grad = grad[offset:offset+p.numel()].view_as(p).clone()
                    groups.append(dict(params=[p], lambdas=[duals[i, offset:offset+p.numel()].view_as(p).clone()],
                        lr=.1, momentum=0, dampening=0, weight_decay=0, nesterov=False))
                    offset += p.numel()
                def pipeline(sync_buffer):
                    sync_buffer['edge_result'] = {j: (packet['values'], packet['indices']
                        if packet['indices'] is not None else torch.arange(8)) for j, packet in packets.items()}
                ref = SimpleNamespace(param_groups=groups, param_names=list(enumerate(['weight', 'bias'])),
                    use_cuda=False, conf=SimpleNamespace(lr_change_epochs=None), state=defaultdict(dict),
                    gamma=.2, eta=.03, eta_ratio=.3, beta=.4, it=0, shapes=None,
                    neighbors_info=peers, helper_thread=None, compressor=SimpleNamespace(
                        prepare_round=lambda *a: None, pipeline=pipeline))
                for n in methods:
                    setattr(ref, n.name, MethodType(scope[n.name], ref))
                ref.step(scheduler=SimpleNamespace(get_lr=lambda: .1))
                ref_dual = ref.get_lambda(groups, ref.param_names)[1].buffer
                delta = flat(actual_model) - flat(old)
                expected_order_gap = torch.zeros_like(delta)
                for packet in packets.values():
                    indices = packet['indices'] if packet['indices'] is not None else torch.arange(8)
                    expected_order_gap[indices] += .4 * delta[indices]
                primal_error = max(primal_error, (flat(actual_model)-flat(ref_model)).abs().max().item())
                order_error = max(order_error, (ref_dual-actual_dual-expected_order_gap).abs().max().item())
                torch.testing.assert_close(flat(actual_model), flat(ref_model), rtol=1e-12, atol=1e-13)
                torch.testing.assert_close(ref_dual-actual_dual, expected_order_gap, rtol=1e-12, atol=1e-13)
                dual_gap = max(dual_gap, (ref_dual-actual_dual).abs().max().item())
                ref_next.append(ref_dual)
                actual_next.append((actual_model, actual_dual))
                cases += 1
            ref_dual_sum = max(ref_dual_sum, torch.stack(ref_next).sum(0).abs().max().item())
            models = [m for m, d in actual_next]
            duals = torch.stack([d for m, d in actual_next])
            torch.testing.assert_close(duals.sum(0), torch.zeros(8, dtype=torch.float64), rtol=0, atol=1e-12)
    assert dual_gap > 1e-4 and ref_dual_sum > 1e-4
    print(json.dumps(dict(passed=True, reference_commit=revision, matched_old_state_node_steps=cases,
        max_primal_error=primal_error, max_dual_order_identity_error=order_error,
        max_dual_discrepancy=dual_gap, max_reference_dual_sum=ref_dual_sum,
        scope='Unchanged reference step with mocked old-model packets and explicit zero-sum dual inputs. '
              'Primal matches; reference dual uses new self minus old neighbors, simulator uses old self per paper (7)/(36). '
              'No full-trajectory equivalence or distributed reference deployment claimed.',
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})))


if __name__ == '__main__':
    check(sys.argv[1])
