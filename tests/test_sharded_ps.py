"""Matched-protocol equivalence, causal profiling and direct scheduler checks."""
import pytest
import torch
from dasklearn import functions
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.simulation.sharded_ps.settings import ShardedPSSettings, AutoDirectSettings
from dasklearn.simulation.sharded_ps.simulation import ShardedPSSimulation, AutoDirectSimulation
from dasklearn.simulation.sharded_ps.planner import pairings, RouteFeedback, summarize_probes
from dasklearn.tasks.local import execute_local
from helpers import settings, simulate


def run(cls, config_cls, path, optimizer='fedadam', **changes):
    options = dict(participants=8, sample_size=4, chunks_in_sample=8, relay_count=4,
        capacity_profile='heterogeneous', bandwidth_bytes_per_s=100000, compute_step_ms=20,
        rounds=3, duration=200_000_000, test_period=0, immediate_uploads=True,
        reducer_epoch_rounds=0, server_optimizer=optimizer,
        server_learning_rate=1 if optimizer=='fedavg' else .01,
        network_latency_ms=10, model_size_bytes=12001, server_step_ms=.1)
    options.update(changes)
    sim = simulate(cls, settings(config_cls, path, **options))
    results = execute_local(sim)
    models = {(t.data['round'], t.data['peer']): ChunkManager.get_flat_params(results[t.name][0])
              for t in sim.workflow_dag.tasks.values() if t.func=='reconstruct_from_chunks'}
    return sim, results, models


@pytest.mark.asyncio
@pytest.mark.parametrize('optimizer', ['fedavg', 'fedavgm', 'fedadam'])
async def test_independent_matched_ps_reproduces_v2_protocol_and_models(tmp_path, optimizer):
    v2, _, expected = run(ConfluxV2Simulation, ConfluxV2Settings, tmp_path/'v2', optimizer)
    ps, _, actual = run(ShardedPSSimulation, ShardedPSSettings, tmp_path/'ps', optimizer)
    assert ps.completed_rounds == v2.completed_rounds
    assert ps.contributor_sets == v2.contributor_sets
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key])
    assert [c.bw_scheduler.total_bytes_sent for c in ps.clients] == [c.bw_scheduler.total_bytes_sent for c in v2.clients]


@pytest.mark.asyncio
@pytest.mark.parametrize('optimizer', ['fedavg', 'fedavgm', 'fedadam'])
async def test_automatic_shards_match_independent_full_vector_fedopt(tmp_path, optimizer):
    sim, results, actual = run(AutoDirectSimulation, AutoDirectSettings, tmp_path, optimizer)
    assert len(actual)==12
    assert all(c.other_nodes_bws == {} for c in sim.clients)
    assert len(sim.plan_arrivals)==8 and min(t for _, t in sim.plan_arrivals)>0
    assert sim.scheduler_cpu_us>0 and sim.scheduler_candidates
    assert len(set(tuple(sim.clients[0].sample(r)) for r in range(1,4)))>1
    s = sim.settings
    base = ChunkManager.get_flat_params(functions.initialize(s, {})[0])
    moment=torch.zeros_like(base); variance=torch.ones_like(base)*s.server_tau**2
    for r in range(1, 4):
        local = {t.data['peer']: results[t.name][0] for t in sim.workflow_dag.tasks.values()
                 if t.func=='train' and t.data['round']==r}
        assert sorted(local) == sim.contributor_sets[r]
        avg=torch.stack([ChunkManager.get_flat_params(local[i]) for i in sorted(local)]).mean(0)
        g=base-avg
        if optimizer=='fedavg':
            base=avg
        elif optimizer=='fedavgm':
            moment=s.server_beta1*moment+g
            base=base-s.server_learning_rate*moment
        else:
            moment=s.server_beta1*moment+(1-s.server_beta1)*g
            variance=s.server_beta2*variance+(1-s.server_beta2)*g.square()
            base=base-s.server_learning_rate*moment/(variance.sqrt()+s.server_tau)
        values=[v for (round_nr, _), v in actual.items() if round_nr==r]
        assert len(values)==4 and all(torch.equal(values[0], v) for v in values)
        torch.testing.assert_close(values[0], base, rtol=1e-6, atol=1e-7)
    assert not sim.optimizer_handoffs
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients)==sum(c.bw_scheduler.total_bytes_received for c in sim.clients)
    sim.save_measurements()
    assert (tmp_path/'data').exists()


@pytest.mark.asyncio
async def test_user_chunk_and_reducer_choices_do_not_drive_auto_plan(tmp_path):
    a, _, _ = run(AutoDirectSimulation, AutoDirectSettings, tmp_path/'a', chunks_in_sample=2, relay_count=1)
    b, _, _ = run(AutoDirectSimulation, AutoDirectSettings, tmp_path/'b', chunks_in_sample=64, relay_count=8)
    assert a.ps_owners == b.ps_owners
    assert a.measured_profiles == b.measured_profiles
    # Measured planner CPU time varies between runs and is explicitly charged.
    assert [(r,i) for r,i,_ in a.completed_rounds] == [(r,i) for r,i,_ in b.completed_rounds]


def test_robust_route_history_does_not_permanently_adopt_one_hiccup():
    feedback=RouteFeedback()
    for ratio in [1,1,100,1,1]: feedback.observe(3, ratio*10, 10)
    assert feedback.factor(3)==1
    for _ in range(3): feedback.observe(3, 100, 10)
    assert feedback.factor(3)==10


@pytest.mark.parametrize('n', [2,3,8,64])
def test_probe_matchings_have_no_simultaneous_endpoint_conflicts(n):
    seen=set()
    for pairs in pairings(n, 4):
        endpoints=[p for a,b in pairs for p in (a,b)]
        assert len(endpoints)==len(set(endpoints))
        seen.update(endpoints)
    assert seen==set(range(n))


def test_two_size_latency_cancellation():
    records=[]
    for src,dst in [(0,1),(1,0)]:
        for phase,size in enumerate([1000,2000]):
            records.append(dict(src=src,dst=dst,phase=phase,size=size,elapsed=20000+(size+100)*10))
    profiles=summarize_probes(records,{0:300,1:400},100)
    assert profiles=={0:(100000,100000,300,20000),1:(100000,100000,400,20000)}


@pytest.mark.asyncio
async def test_inaccurate_profiles_and_one_training_hiccup_preserve_updates(tmp_path, monkeypatch):
    from dasklearn.simulation.sharded_ps import auto_client
    from dasklearn.simulation.events import Event, START_TRAIN
    _, _, expected = run(AutoDirectSimulation, AutoDirectSettings, tmp_path/'reference')
    summarize=auto_client.summarize_probes
    def inaccurate(*args):
        profiles=summarize(*args)
        return {i:(max(1,round(up*(.5 if i%2 else 2))),
                   max(1,round(down*(2 if i%2 else .5))),train,lat)
                for i,(up,down,train,lat) in profiles.items()}
    monkeypatch.setattr(auto_client,'summarize_probes',inaccurate)
    start=auto_client.AutoDirectClient.start_train
    def hiccup(self,event):
        if event.data['round']==2 and self.index==self.sample(2)[0] and not event.data.get('hiccup'):
            self.simulator.schedule(Event(self.context.now+2_000_000,self.index,START_TRAIN,
                                         dict(event.data,hiccup=True)))
        else:
            start(self,event)
    monkeypatch.setattr(auto_client.AutoDirectClient,'start_train',hiccup)
    sim, _, actual=run(AutoDirectSimulation,AutoDirectSettings,tmp_path/'perturbed')
    assert actual.keys()==expected.keys()
    for key in actual: torch.testing.assert_close(actual[key],expected[key],rtol=1e-5,atol=1e-6)
    assert len(sim.completed_rounds)==12


def test_new_algorithms_have_reproducible_config_settings():
    from args import get_args
    from dasklearn.settings_factory import session_options
    for name in ['sharded-ps','conflux-v3']:
        args=get_args(['--algorithm',name,'--local','--server-optimizer','fedadam'])
        config=session_options(args,'cpu')
        assert config['algorithm']==name and config['reducer_epoch_rounds']==0
        assert config['immediate_uploads'] and config['server_optimizer']=='fedadam'


@pytest.mark.asyncio
async def test_sender_local_feedback_cannot_desynchronize_chunk_order(tmp_path):
    sim,_,_=run(AutoDirectSimulation,AutoDirectSettings,tmp_path)
    for i,c in enumerate(sim.clients):
        for owner in set(c.owners):
            for _ in range(5):c.route_feedback.observe(owner,100*(i+owner+1),1)
    for r in range(1,4):
        orders=[sim.clients[i].upload_order(r) for i in sim.clients[0].sample(r)]
        assert all(order==orders[0] for order in orders)
