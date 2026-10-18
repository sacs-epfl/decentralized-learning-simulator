"""Optional parity audit against downloaded, pinned official standalone demo.py.

Usage: python tests/check_demo_reference.py /path/to/official/demo.py
No network access or reference dependency is required by the normal test suite.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dasklearn.optimizer.demo import prepare, update


def check(path):
    spec=importlib.util.spec_from_file_location('official_demo',path)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    torch.manual_seed(173)
    initial=torch.nn.Sequential(torch.nn.Linear(12,8),torch.nn.Linear(8,3))
    replicas=[copy.deepcopy(initial) for _ in range(3)]
    optimizers=[official.DeMo(m.parameters(),lr=.003,compression_chunk=4,compression_topk=2,weight_decay=.02) for m in replicas]
    ours=copy.deepcopy(initial);residuals=[None]*3
    max_parameter_error=0.;max_residual_error=0.
    for step in range(7):
        gradients=[[torch.randn_like(p) for p in m.parameters()] for m in replicas]
        reference_packets=[];packets=[]
        for i,(model,opt,grads) in enumerate(zip(replicas,optimizers,gradients)):
            reference=[]
            for p,g in zip(model.parameters(),grads):
                p.grad=g.clone()
                delta=opt.demo_state[p]['delta']*.999+.003*g
                reference.append(opt.compress.compress(opt.transform.encode(delta),2))
            reference_packets.append(reference)
            packet,residuals[i]=prepare(grads,residuals[i],.003,.999,4,2);packets.append(packet)
        for i,opt in enumerate(optimizers):
            calls=iter(range(len(reference_packets[i])))
            def gather(idx,val):
                j=next(calls)
                return [p[j][0] for p in reference_packets],[p[j][1] for p in reference_packets]
            opt._demo_all_gather=gather
            opt.step()
            for j,p in enumerate(replicas[i].parameters()):
                error=(opt.demo_state[p]['delta']-residuals[i][j]).abs().max().item()
                max_residual_error=max(max_residual_error,error)
                torch.testing.assert_close(opt.demo_state[p]['delta'],residuals[i][j],atol=2e-8,rtol=2e-4)
        ours=update(ours,packets,.003,.02,4)
        for model in replicas:
            for a,b in zip(ours.parameters(),model.parameters()):
                max_parameter_error=max(max_parameter_error,(a-b).abs().max().item())
                torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    print({'passed':True,'workers':3,'steps':7,'max_parameter_error':max_parameter_error,'max_residual_error':max_residual_error})


if __name__=='__main__':check(sys.argv[1])
