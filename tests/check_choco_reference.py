"""Optional numerical audit against pinned epfml/ChocoSGD source helpers.

Usage: python tests/check_choco_reference.py /path/to/ChocoSGD
Extracts only the reference's pure numerical definitions, avoiding its obsolete
MPI/bit2byte/CUDA imports. The usual pytest suite has no external dependency.
"""
import ast
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasklearn.optimizer.choco import prepare, receive, decompress

PIN = '3cb976a54499c2cdc8ea68e001217b42b58870ae'


def definition(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    namespace = {'torch': torch, 'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def check(root):
    root = Path(root)
    revision = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if revision != PIN:
        raise ValueError(f'Reference must be pinned to {PIN}, got {revision}')
    base = root/'dl_code/pcode'
    compressor_path = base/'utils/sparsification.py'
    utils_path = base/'optim/utils.py'
    compressor = definition(compressor_path, 'SparsificationCompressor')()
    correct = definition(utils_path, 'update_params_from_neighbor')
    apply_gradient = definition(utils_path, 'apply_gradient')
    torch.manual_seed(81)
    template = torch.nn.Linear(3,2).double()
    ours = [copy.deepcopy(template) for _ in range(4)]
    reference = copy.deepcopy(ours)
    states = [None]*4
    hats = [torch.zeros(8,dtype=torch.float64) for _ in range(4)]
    memories = copy.deepcopy(hats)
    max_error = 0.
    for r in range(8):
        corrected, packets, reference_packets = [], [], []
        for i in range(4):
            grads = [torch.randn_like(p) for p in ours[i].parameters()]
            for actual, expected, g in zip(ours[i].parameters(), reference[i].parameters(), grads):
                with torch.no_grad():actual.sub_(g, alpha=.03)
                expected.grad = g.clone()
            group = {'params':list(reference[i].parameters()), 'lr':.03,
                     'weight_decay':0, 'momentum':0, 'dampening':0, 'nesterov':False}
            apply_gradient([group],defaultdict(dict),apply_grad_to_model=True)
            flat = SimpleNamespace(buffer=torch.cat([p.detach().reshape(-1) for p in reference[i].parameters()]))
            correct({i:SimpleNamespace(buffer=hats[i]), 'memory':SimpleNamespace(buffer=memories[i])},flat,.2,i)
            with torch.no_grad():
                offset=0
                for p in reference[i].parameters():
                    p.copy_(flat.buffer[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
            model, packet = prepare(ours[i],states[i],'topk',.25,.2,r)
            corrected.append(model);packets.append(packet)
            delta=flat.buffer-hats[i]
            q=torch.zeros_like(delta);offset=0
            for p in reference[i].parameters():
                values,indices=compressor.compress(delta[offset:offset+p.numel()],'top_k',.75,True)
                q[offset+indices]=values;offset+=p.numel()
            reference_packets.append(q)
            torch.testing.assert_close(decompress(packet,8),q,rtol=1e-12,atol=1e-12)
            for a,b in zip(model.parameters(),reference[i].parameters()):
                max_error=max(max_error,(a-b).abs().max().item())
                torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        for i in range(4):
            peers=[i,(i-1)%4,(i+1)%4]
            states[i]=receive(states[i],[packets[j] for j in peers],[1/3]*3,8)
            hats[i]=hats[i]+reference_packets[i]
            memories[i]=memories[i]+sum(reference_packets[j]/3 for j in peers)
            torch.testing.assert_close(states[i]['own'],hats[i],rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(states[i]['memory'],memories[i],rtol=1e-12,atol=1e-12)
        ours=corrected
    print(json.dumps({'passed':True,'reference_commit':revision,'workers':4,'steps':8,
        'scope':'SGD, Top-k compressor, old-hat correction and persistent estimate updates; not MPI transport',
        'max_parameter_error':max_error,
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [compressor_path,utils_path]}}))


if __name__ == '__main__':check(sys.argv[1])
