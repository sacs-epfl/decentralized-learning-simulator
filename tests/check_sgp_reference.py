"""Execute the authors' PushSum and numerator/ratio conversions without MPI.

Usage: python tests/check_sgp_reference.py /path/to/stochastic_gradient_push
Original arithmetic executes unchanged. Process-group discovery, buffers and
broadcast transport are replaced with deterministic in-memory equivalents.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasklearn.optimizer.gt_dsgd import flat
from dasklearn.optimizer.sgp import prepare, mix

PIN = 'e682064d3f0a6216868f8528f804afe091af19b8'


def check(root):
    root = Path(root)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError('Unexpected reference revision: ' + revision)
    paths = [root/'gossip'/name for name in ['mixing_manager.py', 'gossiper.py', 'distributed.py']]
    nodes = []
    for path, names in zip(paths, [{'MixingManager', 'UniformMixing'}, {'Gossiper', 'PushSum'}, set()]):
        tree = ast.parse(path.read_text())
        nodes.extend(n for n in tree.body if getattr(n, 'name', '') in names)
        if path.name == 'distributed.py':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GossipDataParallel')
            nodes.extend(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'ps_numerator', 'unbias'})
    rank, wire = 0, {}
    def broadcast(tensor, src, group, async_op=False):
        if async_op:
            torch.testing.assert_close(tensor, wire[src, group], rtol=1e-12, atol=1e-12)
            return SimpleNamespace(wait=lambda: None)
        tensor.copy_(wire[src, rank])
    scope = {'torch': torch, 'dist': SimpleNamespace(broadcast=broadcast)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned SGP reference>', 'exec'), scope)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(73)
        template = torch.nn.Linear(3, 2)
        max_parameter_error, max_mass_error = 0., 0.
        for topology in ['exponential', 'ring', 'irregular', 'complete']:
            ours = [copy.deepcopy(template) for _ in range(4)]
            refs = [SimpleNamespace(module=copy.deepcopy(template), ps_weight=torch.ones(1),
                    lazy_mixing=False, is_ps_numerator=False) for _ in range(4)]
            numerators, masses = [None] * 4, [1.] * 4
            for k in range(8):
                offset = 2 ** (k % 2) if topology == 'exponential' else 1
                outgoing = [list(range(4)) if topology == 'complete' else [(i + offset) % 4] for i in range(4)]
                outgoing = [[j for j in peers if j != i] for i, peers in enumerate(outgoing)]
                if topology == 'irregular':
                    outgoing[0].append(2)
                incoming = [[j for j in range(4) if i in outgoing[j]] for i in range(4)]
                local, own_packets, local_masses, gossipers, vectors = [], [], [], [], []
                wire = {}
                for rank in range(4):
                    target = torch.arange(8) * .1 + rank
                    g = flat(ours[rank]) - target
                    gradient_model = copy.deepcopy(ours[rank])
                    gradient_model.gradient = [g[:6].reshape(2, 3), g[6:]]
                    a, b = prepare(gradient_model, numerators[rank], .03, 0., 1 / (len(outgoing[rank]) + 1))
                    local.append(a)
                    own_packets.append(b)
                    local_masses.append(masses[rank] / (len(outgoing[rank]) + 1))
                    ref = refs[rank]
                    ref_g = flat(ref.module) - target
                    for p, grad in zip(ref.module.parameters(), [ref_g[:6].reshape(2, 3), ref_g[6:]]):
                        p.grad = grad.clone()
                    # The reference backward hook converts z to x before the
                    # optimizer step, retaining gradients evaluated at z.
                    scope['ps_numerator'](ref)
                    torch.optim.SGD(ref.module.parameters(), lr=.03).step()
                    vector = flat(ref.module)
                    vectors.append(vector)
                    peers = outgoing[rank]
                    manager = SimpleNamespace(get_peers=lambda peers=peers: (peers, []))
                    gossiper = scope['PushSum'].__new__(scope['PushSum'])
                    gossiper._mixing_manager = scope['UniformMixing'](manager, torch.device('cpu'))
                    gossiper._ps_weight = ref.ps_weight.clone()
                    # Exercise the general scalar-carrying path even on regular
                    # graphs; the reference can omit scalars when w is always 1.
                    gossiper.regular = False
                    gossiper.rank = rank
                    gossiper.logger = None
                    gossiper.device = torch.device('cpu')
                    gossiper.out_edges = [SimpleNamespace(src=rank, dest=j, process_group=j) for j in peers]
                    gossiper.in_edges = [SimpleNamespace(src=j, dest=rank, process_group=rank) for j in incoming[rank]]
                    gossiper.in_msg_buffer = torch.zeros(9)
                    gossiper.placeholder = torch.zeros(9)
                    gossiper.out_msg_buffer = []
                    gossiper.refresh_peers_ = lambda: None
                    messages = list(gossiper.mix_out_msg_(vector.clone(), ref.ps_weight.clone(), residual=False))
                    for j, message in zip(peers, messages[1:]):
                        wire[rank, j] = message.clone()
                        torch.testing.assert_close(message[:8], b, rtol=1e-12, atol=1e-12)
                        torch.testing.assert_close(message[8], torch.tensor(local_masses[rank]), rtol=1e-12, atol=1e-12)
                    gossipers.append(gossiper)
                for rank in range(4):
                    packets = [own_packets[rank]] + [own_packets[j] for j in incoming[rank]]
                    mass = local_masses[rank] + sum(local_masses[j] for j in incoming[rank])
                    ours[rank], numerators[rank] = mix(None, {'local_model': local[rank], 'packets': packets, 'mass': mass})
                    masses[rank] = mass
                    ref_x, ref_mass = gossipers[rank].mix(vectors[rank].clone(), refs[rank].ps_weight.clone(), residual=False)
                    with torch.no_grad():
                        refs[rank].module.weight.copy_(ref_x[:6].reshape(2, 3))
                        refs[rank].module.bias.copy_(ref_x[6:])
                    refs[rank].ps_weight = ref_mass.reshape(1).clone()
                    scope['unbias'](refs[rank])
                    max_parameter_error = max(max_parameter_error, (flat(ours[rank]) - flat(refs[rank].module)).abs().max().item())
                    max_mass_error = max(max_mass_error, abs(mass - ref_mass.item()))
                    torch.testing.assert_close(flat(ours[rank]), flat(refs[rank].module), rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(numerators[rank], ref_x, rtol=1e-12, atol=1e-12)
                    assert abs(mass - ref_mass.item()) < 1e-12
        print(json.dumps({'passed': True, 'reference_commit': revision, 'workers': 4, 'steps': 8,
            'topologies': ['exponential', 'ring', 'irregular', 'complete'], 'dtype': 'float64',
            'max_parameter_error': max_parameter_error, 'max_mass_error': max_mass_error,
            'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'scope': 'Actual UniformMixing, PushSum.mix, ps_numerator and unbias; non-IID quadratic gradients and plain PyTorch SGD; mocked broadcast transport and graph discovery; scalar optimization disabled; no overlap, momentum or decay'}))
    finally:
        torch.set_default_dtype(old_dtype)


if __name__ == '__main__':
    check(sys.argv[1])
