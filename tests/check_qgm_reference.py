"""Execute the pinned official QG optimizer's numerical step, without its MPI setup.

Usage: python tests/check_qgm_reference.py /path/to/quasi-global-momentum
Only transport/tensor packing and timing are replaced; the reference step's
SGD, Nesterov and virtual-sequence momentum arithmetic executes unchanged.
"""
import ast
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dasklearn.optimizer.qgm import flat,prepare,finish

PIN='d1d2f758c549db6093c52be71bcdbdf1a3976549'


class Buffer:
    def __init__(self,params):self.buffer=torch.cat([p.detach().reshape(-1) for p in params])
    def unpack(self,params):
        with torch.no_grad():
            start=0
            for p in params:p.copy_(self.buffer[start:start+p.numel()].reshape_as(p));start+=p.numel()


def check(root):
    root=Path(root)
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if revision!=PIN:raise ValueError('Unexpected reference revision: '+revision)
    path=root/'code/pcode/optim/decentralized_qg_sgd.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SGD')
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    slow=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_slow_buffer_step')
    def data(groups,names,is_get_grad=False):
        params=groups[0]['params'];return ([p.grad if is_get_grad else p for p in params],None)
    scope={'torch':torch,'comm':SimpleNamespace(get_data=data),'TensorBuffer':Buffer,
           'get_n_bits':lambda t:8*t.numel()*t.element_size()}
    exec(compile(ast.Module(body=[step,slow],type_ignores=[]),str(path),'exec'),scope)
    torch.manual_seed(73);template=torch.nn.Linear(3,2).double();maximum=0
    W=torch.tensor([[1/3 if i==j or (i-j)%4 in [1,3] else 0 for j in range(4)] for i in range(4)],dtype=torch.float64)
    for nesterov in [False,True]:
        ours=[copy.deepcopy(template) for _ in range(4)];refs=copy.deepcopy(ours);states=[None]*4
        optimizers=[]
        for model in refs:
            optimizers.append(SimpleNamespace(param_groups=[{'params':list(model.parameters()),'lr':.03}],
                param_names=[],momentum=.9,dampening=0,weight_decay=.01,nesterov=nesterov,
                momentum_buffer=torch.zeros(8,dtype=torch.float64),virtual_seq_buffer=torch.zeros(8,dtype=torch.float64),
                conf=SimpleNamespace(is_centralized=False,epoch_=0,comm_device='cpu',graph_topology='ring',
                                     slow_buffer_conf_={'factor':.1},local_index=0)))
        for r in range(8):
            old=copy.deepcopy(ours);locals=[];packets=[];local_states=[]
            for i in range(4):
                gradients=[torch.randn_like(p) for p in ours[i].parameters()]
                ours[i].gradient=gradients
                proposal,state,packet=prepare(ours[i],states[i],.03,.9,.01,nesterov,'quasi')
                locals.append(proposal);local_states.append(state);packets.append(packet)
                for p,g in zip(refs[i].parameters(),gradients):p.grad=g.clone()
            # Independent transport values computed from reference models and state.
            expected_packets=[]
            for i in range(4):
                gradient=torch.cat([p.grad.reshape(-1) for p in refs[i].parameters()])+.01*flat(refs[i])
                update=.9*optimizers[i].momentum_buffer+gradient
                direction=gradient+.9*update if nesterov else update
                expected_packets.append(flat(refs[i])-.03*direction)
            mixed=W@torch.stack(expected_packets)
            for i in range(4):
                def aggregate(value,op,peer=i):
                    torch.testing.assert_close(value,expected_packets[peer],rtol=1e-12,atol=1e-12)
                    return mixed[peer].clone()
                opt=optimizers[i];opt.conf.local_index=r+1
                opt.decentralized_aggregator=SimpleNamespace(_agg=aggregate,out_neighbors_info={j:1/3 for j in range(4) if j!=i and W[i,j]})
                scope['step'](opt,timer=lambda *a,**k:nullcontext())
                peers=[i,(i-1)%4,(i+1)%4]
                ours[i],states[i]=finish(old[i],locals[i],states[i],local_states[i],[packets[j] for j in peers],[1/3]*3,.03,.9,'quasi')
                maximum=max(maximum,(flat(ours[i])-flat(refs[i])).abs().max().item())
                torch.testing.assert_close(flat(ours[i]),flat(refs[i]),rtol=1e-11,atol=1e-12)
                torch.testing.assert_close(states[i],opt.momentum_buffer,rtol=1e-11,atol=1e-12)
    print(json.dumps({'passed':True,'reference_commit':revision,'reference_file':str(path.relative_to(root)),
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'workers':4,'steps':8,
        'nesterov':[False,True],'weight_decay':.01,'max_parameter_error':maximum,
        'scope':'Actual reference optimizer numerical step with predetermined weighted transport; excludes MPI setup'}))


if __name__=='__main__':check(sys.argv[1])
