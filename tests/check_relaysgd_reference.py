"""Run the pinned authors' RelayMechanism with deterministic in-memory transport.

Usage: python tests/check_relaysgd_reference.py /path/to/relaysgd
The original send/receive/normalization arithmetic executes unchanged; only
distributed transport and topology discovery are replaced.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, Tuple

import networkx as nx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasklearn.optimizer.gt_dsgd import flat
from dasklearn.optimizer.relaysgd import forward, finish

PIN = 'e1e332bfaf6a2b29f1677b7fa31a27b6e0d2802a'


def check(root):
    root = Path(root)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError('Unexpected reference revision: ' + revision)
    path = root / 'utils/communication.py'
    names = {'RelayMechanism', 'MultiHandle', 'num_bytes', 'isend_to_neighbors', 'recv_sum_into_tensor'}
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') in names]
    rank, wire = 0, {}
    def send(tensor, dst, tag):
        wire[rank, dst, tensor.dtype] = tensor.clone()
        return SimpleNamespace(wait=lambda: None)
    def receive(tensor, topology, tag):
        return {peer: wire[peer, rank, tensor.dtype].clone() for peer in topology.neighbors(rank)}
    scope = {'torch': torch, 'random': random, 'Dict': Dict, 'Tuple': Tuple, 'Topology': object,
             'get_rank': lambda: rank, 'get_world_size': lambda: 7,
             'isend': send, 'recv_messages': receive}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    torch.manual_seed(73)
    template = torch.nn.Linear(3, 2).double()
    maximum = 0.
    for topology_name in ['balanced', 'path', 'star']:
        graph = nx.Graph([(int((i-1)//2), i) for i in range(1, 7)]) if topology_name == 'balanced' else (
            nx.path_graph(7) if topology_name == 'path' else nx.star_graph(6))
        topology = SimpleNamespace(num_workers=7, max_degree=max(dict(graph.degree()).values()),
                                   neighbors=lambda i: sorted(graph.neighbors(i)))
        for normalization in ['counts', 'initial']:
            settings = SimpleNamespace(participants=7, normalization=normalization)
            ours = [copy.deepcopy(template) for _ in range(7)]
            refs = [scope['RelayMechanism'](topology, normalization_mode=(
                'counts' if normalization == 'counts' else 'world_size'), initial_state=flat(template))
                for _ in range(7)]
            ref_models = [flat(template) for _ in range(7)]
            previous = [{} for _ in range(7)]
            for iteration in range(8):
                wire = {}
                local, outgoing = [], {}
                for rank in range(7):
                    # Shared deterministic non-IID quadratic oracle, applied to
                    # each implementation's own current parameters.
                    target = torch.arange(8, dtype=torch.float64) * .1 + rank
                    parcel = copy.deepcopy(ours[rank])
                    vector = flat(parcel) - .03 * (flat(parcel) - target)
                    with torch.no_grad():
                        parcel.weight.copy_(vector[:6].reshape(2, 3))
                        parcel.bias.copy_(vector[6:])
                    local.append(parcel)
                    refs[rank].send(ref_models[rank] - .03 * (ref_models[rank] - target))
                    for dst in topology.neighbors(rank):
                        branches = [p for peer, p in sorted(previous[rank].items()) if peer != dst]
                        total = forward(settings, {'local_model': parcel, 'packets': [p['sum'] for p in branches]})[0]
                        count = 1 + sum(p['count'] for p in branches)
                        torch.testing.assert_close(total, wire[rank, dst, torch.float64], rtol=1e-12, atol=1e-12)
                        assert count == wire[rank, dst, torch.int64].item()
                        outgoing[rank, dst] = {'sum': total, 'count': count}
                for rank in range(7):
                    received = {p: outgoing[p, rank] for p in topology.neighbors(rank)}
                    ours[rank] = finish(settings, {'local_model': local[rank],
                        'packets': [p['sum'] for p in received.values()],
                        'counts': [p['count'] for p in received.values()], 'initial_model': template})[0]
                    ref_models[rank] = refs[rank].receive_average()
                    maximum = max(maximum, (flat(ours[rank]) - ref_models[rank]).abs().max().item())
                    torch.testing.assert_close(flat(ours[rank]), ref_models[rank], rtol=1e-12, atol=1e-12)
                    previous[rank] = received
    result = {'passed': True, 'reference_commit': revision, 'reference_file': 'utils/communication.py',
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'workers': 7, 'steps': 8,
              'topologies': ['balanced', 'path', 'star'], 'normalizations': ['counts', 'initial'],
              'max_parameter_error': maximum,
              'scope': 'Actual reference RelayMechanism, forwarding and receive-sum arithmetic; mocked distributed transport; no drops or overlap'}
    print(json.dumps(result))


if __name__ == '__main__':
    check(sys.argv[1])
