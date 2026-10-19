"""Numerical contracts for the historical baselines, independent of their trainer.

Full-batch logistic regression makes every local update reproducible without
sharing minibatch/optimizer code with the implementation under test.
"""
import copy
from random import Random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dasklearn import functions
from dasklearn.datasets.synthetic import TensorPartition
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.adpsgd.settings import ADPSGDSettings
from dasklearn.simulation.adpsgd.simulation import ADPSGDSimulation
from dasklearn.simulation.dpsgd.settings import DPSGDSettings
from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation
from dasklearn.simulation.fl.settings import FLSettings
from dasklearn.simulation.fl.simulation import FLSimulation
from dasklearn.simulation.events import Event, INCOMING_MODEL
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings


@pytest.fixture
def logistic(monkeypatch):
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(81)
    parts = [TensorPartition(torch.randn(n, 3, generator=g), torch.arange(n) % 2)
             for n in [3, 5, 7, 9]]
    initial = torch.nn.Linear(3, 2)
    with torch.no_grad():
        initial.weight.copy_(torch.tensor([[.2, -.1, .3], [-.2, .4, .1]]))
        initial.bias.copy_(torch.tensor([.1, -.1]))
    dataset = SimpleNamespace(load_partition=lambda peer, split: parts[peer],
                              load_split=lambda split: parts[0])
    monkeypatch.setattr(functions, 'create_dataset', lambda s: dataset)
    monkeypatch.setattr(functions, 'create_model', lambda *a, **k: copy.deepcopy(initial))
    monkeypatch.setattr(functions, 'copy_model_state', lambda model, *a, **k: copy.deepcopy(model))
    return initial, parts


def local_sgd(model, partition, lr, steps):
    result = copy.deepcopy(model)
    optimizer = torch.optim.SGD(result.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(result(partition.features), partition.labels).backward()
        optimizer.step()
    return result


def mean_models(models, weights=None):
    weights = weights or [1 / len(models)] * len(models)
    result = copy.deepcopy(models[0])
    with torch.no_grad():
        for target, values in zip(result.parameters(), zip(*(m.parameters() for m in models))):
            target.copy_(sum(w * p for w, p in zip(weights, values)))
    return result


def assert_model(actual, expected):
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)


def run(cls, config, skew=False):
    sim = cls(config)
    sim.setup_directories()
    def capacities():
        for peer in sim.clients:
            peer.simulated_speed = [4, 17, 8, 31][peer.index] if skew else 10
            peer.bw_scheduler.bw_limit = 100000
    sim.apply_traces = capacities
    sim.simulate()
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    return sim


def configuration(cls, tmp_path, algorithm, **kwargs):
    return settings(cls, tmp_path, algorithm=algorithm, participants=4,
                    learning=LearningSettings(.1, 0, 0, 64, kwargs.pop('steps', 1)),
                    rounds=4, duration=2 * MICROSECONDS, test_period=0,
                    model_size_bytes=1024, network_latency_ms=3, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('steps', [1, 3])
@pytest.mark.parametrize('sync,skew', [(False, False), (False, True), (True, True)])
async def test_dpsgd_multi_round_train_then_mix_reference(tmp_path, logistic, steps, sync, skew):
    initial, parts = logistic
    sim = run(DPSGDSimulation, configuration(DPSGDSettings, tmp_path, 'dpsgd',
              steps=steps, topology='ring', synchronous=sync), skew)
    result = execute_local(sim)
    states = [copy.deepcopy(initial) for _ in parts]
    for round_nr in range(1, 5):
        trained = [local_sgd(model, p, .1, steps) for model, p in zip(states, parts)]
        graph = sim.topologies[round_nr - 1]
        expected = [mean_models([trained[j] for j in [i] + list(graph.predecessors(i))])
                    for i in range(4)]
        mixes = [t for t in sim.workflow_dag.tasks.values()
                 if t.func == 'aggregate' and t.data['round'] == round_nr]
        assert len(mixes) == 4
        for task in mixes:
            assert_model(result[task.name][0], expected[task.data['peer']])
        # Regular graph mixing preserves the mean of the trained parameters.
        assert_model(mean_models(expected), mean_models(trained))
        states = expected
    assert sim.clients[0].last_round_completed == 4
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == 4 * 4 * 2 * 1024
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == 4 * 4 * 2 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize('weighting', ['uniform', 'examples'])
@pytest.mark.parametrize('sample_size', [0, 2, 4])
async def test_fl_multiround_reference_with_unequal_partitions(tmp_path, logistic, weighting, sample_size):
    initial, parts = logistic
    sim = run(FLSimulation, configuration(FLSettings, tmp_path, 'fl', steps=3,
              sample_size=sample_size, aggregation_weighting=weighting, test_interval=1), True)
    result = execute_local(sim)
    rng = Random(sim.settings.seed)
    expected = initial
    for round_nr in range(1, 5):
        selected = rng.sample(list(range(4)), sample_size or 4)
        trained = [local_sgd(expected, parts[i], .1, 3) for i in selected]
        counts = [len(parts[i]) for i in selected]
        weights = [n / sum(counts) for n in counts] if weighting == 'examples' else None
        expected = mean_models(trained, weights)
        task = next(t for t in sim.workflow_dag.tasks.values()
                    if t.func == 'aggregate' and t.data['round'] == round_nr)
        assert_model(result[task.name][0], expected)
    assert_model(result[sim.server.global_model[0]][0], expected)
    if sample_size != 2:
        assert all(c.compute_time > 0 for c in sim.clients[:-1])
    traffic = 4 * (sample_size or 4) * 2 * 1024
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == traffic
    assert sum(c.bw_scheduler.total_bytes_received for c in sim.clients) == traffic


@pytest.mark.asyncio
async def test_fl_ignores_duplicate_stale_and_unsampled_replies(tmp_path):
    sim = run(FLSimulation, configuration(FLSettings, tmp_path, 'fl', sample_size=2))
    server = sim.server
    server.round = 9
    server.selected_clients = {0, 1}
    server.responded_clients = set()
    server.incoming_models = []
    server.incoming_peers = []
    for peer, round_nr in [(0, 8), (3, 9), (0, 9), (0, 9)]:
        server.on_incoming_model(Event(sim.current_time, 4, INCOMING_MODEL,
            {'from': peer, 'model': ('sentinel', 0), 'metadata': {'round': round_nr}}))
    assert server.incoming_peers == [0]
    assert server.incoming_models == [('sentinel', 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize('skew', [False, True])
async def test_adpsgd_stale_gradients_and_identical_pairwise_mixes(tmp_path, logistic, skew):
    _, parts = logistic
    sim = run(ADPSGDSimulation, configuration(ADPSGDSettings, tmp_path, 'adpsgd'), skew)
    result = execute_local(sim)
    gradients = [t for t in sim.workflow_dag.tasks.values() if t.func == 'compute_gradient']
    updates = [t for t in sim.workflow_dag.tasks.values() if t.func == 'gradient_update']
    assert len(updates) >= 4 and len(gradients) >= len(updates)
    stale = 0
    for task in updates:
        source = sim.workflow_dag.tasks[task.data['gradient_model'][0]]
        before_gradient = result[source.data['model'][0]][0]
        current = result[task.data['model'][0]][0]
        gradient_at_old = copy.deepcopy(before_gradient)
        part = parts[task.data['peer']]
        torch.nn.functional.cross_entropy(gradient_at_old(part.features), part.labels).backward()
        expected = copy.deepcopy(current)
        with torch.no_grad():
            for p, old in zip(expected.parameters(), gradient_at_old.parameters()):
                p.sub_(old.grad, alpha=.1)
        assert_model(result[task.name][0], expected)
        stale += task.data['model'] != source.data['model']
    assert stale > 0, 'Test must actually mix during gradient computation'
    mixes = [t for t in sim.workflow_dag.tasks.values() if t.func == 'aggregate']
    passive_pairs = {frozenset(t.data['models']) for t in mixes if t.data['peer'] in sim.passive_peers}
    active_mixes = [t for t in mixes if t.data['peer'] in sim.active_peers]
    assert active_mixes
    for task in active_mixes:
        assert frozenset(task.data['models']) in passive_pairs
        expected = mean_models([result[ref[0]][ref[1]] for ref in task.data['models']])
        assert_model(result[task.name][0], expected)
    assert sum(c.bw_scheduler.total_bytes_sent for c in sim.clients) == sum(c.bw_scheduler.total_bytes_received for c in sim.clients)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['oracle', 'local'])
async def test_epidemic_multi_round_topology_and_mixing(tmp_path, logistic, mode):
    sim = run(DPSGDSimulation, configuration(DPSGDSettings, tmp_path, 'epidemic',
              k=2, el=mode, topology='ring'), True)
    result = execute_local(sim)
    assert len(sim.topologies) == 4
    assert all(set(g) == set(range(4)) for g in sim.topologies)
    for task in sim.workflow_dag.tasks.values():
        if task.func == 'aggregate':
            r, peer = task.data['round'], task.data['peer']
            assert len(task.data['models']) == 1 + sim.topologies[r - 1].in_degree(peer)
            assert_model(result[task.name][0], mean_models([result[ref[0]][0] for ref in task.data['models']]))
    if mode == 'local':
        assert all(g.out_degree(i) == 2 for g in sim.topologies for i in g)


@pytest.mark.asyncio
@pytest.mark.parametrize('cls,config,algorithm,extra', [
    (ADPSGDSimulation, ADPSGDSettings, 'adpsgd', {}),
    (DPSGDSimulation, DPSGDSettings, 'dpsgd', {'topology': 'ring'}),
    (FLSimulation, FLSettings, 'fl', {'sample_size': 4}),
])
async def test_classic_fixed_snapshots_publish_updated_models(tmp_path, cls, config, algorithm, extra):
    s = configuration(config, tmp_path, algorithm, **extra)
    s.test_period = MICROSECONDS // 2
    sim = run(cls, s, True)
    snapshots = [t for t in sim.workflow_dag.tasks.values() if t.name.startswith('test_snapshot')]
    assert len(snapshots) == 16
    assert all(sim.workflow_dag.tasks[t.data['model'][0]].func != 'initialize' for t in snapshots)
    assert all(t.data['round'] > 0 for t in snapshots)
    execute_local(sim, retain_results=False)
    sim.merge_accuracies_files()
    metrics = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert len(metrics) == 16
    assert metrics.groupby('time').peer.nunique().eq(4).all()
    assert np.isfinite(metrics[['accuracy', 'loss']]).all().all()


@pytest.mark.asyncio
async def test_fl_server_is_assigned_a_broker_even_for_saved_dag_replay(tmp_path):
    sim = run(FLSimulation, configuration(FLSettings, tmp_path, 'fl', sample_size=4))
    sim.clients = []  # Same state as loading a saved workflow rather than simulating it.
    sim.broker_addresses = {'a': 'tcp://127.0.0.1:1', 'b': 'tcp://127.0.0.1:2'}
    sim.distribute_brokers()
    assert set(sim.clients_to_brokers) == set(range(5))
    assert sim.clients_to_brokers[4] == 'a'


@pytest.mark.asyncio
@pytest.mark.parametrize('algorithm,extra', [('dpsgd', {'topology': 'ring'}),
                                          ('epidemic', {'el': 'local', 'k': 2}),
                                          ('adpsgd', {}), ('fl', {'sample_size': 2})])
async def test_classic_dag_and_training_reproducibility(tmp_path, algorithm, extra):
    from dasklearn.algorithms import class_paths, import_class
    config_type, cls = map(import_class, class_paths(algorithm))
    dags, frames = [], []
    for repeat in range(2):
        Task.COUNTER.clear()
        config = configuration(config_type, tmp_path / str(repeat), algorithm, **extra)
        config.test_period = MICROSECONDS
        sim = run(cls, config, True)
        dags.append(sim.workflow_dag.serialize())
        execute_local(sim, retain_results=False)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv'))
    assert dags[0] == dags[1]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('algorithm,extra', [('dpsgd', {'topology': 'ring'}),
                                          ('adpsgd', {}), ('fl', {'sample_size': 4})])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
async def test_classic_optimized_execution_matches_reference(tmp_path, algorithm, extra, device):
    from dasklearn.algorithms import class_paths, import_class
    from dasklearn.tasks.training import TrainingExecutor
    config_type, cls = map(import_class, class_paths(algorithm))
    config = configuration(config_type, tmp_path, algorithm, **extra)
    config.torch_device_name = device
    sim = run(cls, config, True)
    # Explicitly disable automatic backend selection for the reference execution.
    config.gpu_residency = config.training_batching = False
    expected = execute_local(sim)
    config.gpu_residency = config.training_batching = True
    backend = TrainingExecutor(config, autotune=False)
    actual = execute_local(sim, training_backend=backend)
    assert actual.keys() == expected.keys()
    for name in actual:
        assert_model(actual[name][0], expected[name][0])
        if sim.workflow_dag.tasks[name].func == 'compute_gradient':
            for a, b in zip(actual[name][0].gradient, expected[name][0].gradient):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
    assert backend.stats['batched_tasks'] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize('algorithm', ['adpsgd', 'dpsgd', 'fl'])
async def test_saved_dag_can_use_separate_output_directory(tmp_path, algorithm):
    from dasklearn.algorithms import class_paths, import_class
    config_type, cls = map(import_class, class_paths(algorithm))
    s = configuration(config_type, tmp_path, algorithm)
    s.from_dir = str(tmp_path / 'immutable-source')
    s.data_dir = str(tmp_path / 'replay-output')
    sim = cls(s)
    assert sim.data_dir == s.data_dir == str(tmp_path / 'replay-output')


@pytest.mark.asyncio
@pytest.mark.parametrize('algorithm,extra', [
    ('adpsgd', {'participants': 1}), ('adpsgd', {'churn': 'synthetic'}),
    ('fl', {'sample_size': -1}), ('fl', {'aggregation_weighting': 'unknown'}),
    ('fl', {'churn': 'synthetic'}), ('epidemic', {'el': 'unknown'}),
    ('dpsgd', {'k': 4}),
])
async def test_unsupported_classic_settings_fail_early(tmp_path, algorithm, extra):
    from dasklearn.algorithms import class_paths, import_class
    config_type, cls = map(import_class, class_paths(algorithm))
    s = configuration(config_type, tmp_path, algorithm)
    for key, value in extra.items():
        setattr(s, key, value)
    with pytest.raises(ValueError):
        cls(s)
