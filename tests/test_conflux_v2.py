"""Research-branch Conflux V2 checks, separated from the generic executor suite."""
import copy
from collections import defaultdict
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from dasklearn import functions
from dasklearn.models import create_model
from dasklearn.session_settings import LearningSettings
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.simulation.conflux_v2.settings import ConfluxV2Settings
from dasklearn.simulation.conflux_v2.simulation import ConfluxV2Simulation
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.training import TrainingExecutor
from dasklearn.tasks.task import Task
from dasklearn.util import MICROSECONDS
from helpers import settings, simulate
from test_execution_optimizations import wire_copy
from test_training_backend import assert_states


@pytest.mark.asyncio
@pytest.mark.parametrize('cls,config,kwargs', [(ConfluxV2Simulation, ConfluxV2Settings, dict(sample_size=3, chunks_in_sample=3))])
async def test_research_real_protocol_and_training_dag(tmp_path, cls, config, kwargs):
    sim = simulate(cls, settings(config, tmp_path, **kwargs))
    assert all((type(client.bw_scheduler) is BWScheduler for client in sim.clients))
    trained = [task for task in sim.workflow_dag.tasks.values() if task.func == 'train']
    assert trained and max((task.data['round'] for task in trained)) == 3
    assert sum((client.compute_time for client in sim.clients)) > 0
    assert sum((client.bw_scheduler.total_bytes_sent for client in sim.clients)) == sum((client.bw_scheduler.total_bytes_received for client in sim.clients))
    snapshots = [task for task in sim.workflow_dag.tasks.values() if task.name.startswith('test_snapshot')]
    assert len([task for task in snapshots if task.data['time'] == 2 * MICROSECONDS]) == 6
    results = execute_local(sim)
    assert len(results) == len(sim.workflow_dag.tasks)
    sim.merge_accuracies_files()
    import pandas as pd
    frame = pd.read_csv(sim.data_dir + '/accuracies.csv')
    assert frame['learning_rate'].eq(0.1).all()
    assert frame['peer'].between(0, 5).all()
    assert frame['time'].between(2, 30).all()
    assert frame['accuracy'].between(0, 100).all()
    assert len(frame) == len(snapshots)


@pytest.mark.asyncio
@pytest.mark.parametrize('fraction', [0.5, 1.0])
async def test_research_v2_reconstructs_exact_common_average(tmp_path, fraction):
    sim = simulate(ConfluxV2Simulation, settings(ConfluxV2Settings, tmp_path, sample_size=4, chunks_in_sample=3, success_fraction=fraction, rounds=2))
    results = execute_local(sim)
    for round_nr in [1, 2]:
        contributors = sim.contributor_sets[round_nr]
        models = [results[t.name][0] for t in sim.workflow_dag.tasks.values() if t.func == 'train' and t.data['round'] == round_nr and (t.data['peer'] in contributors)]
        expected = torch.stack([ChunkManager.get_flat_params(model) for model in models]).mean(0)
        reconstructed = [results[t.name][0] for t in sim.workflow_dag.tasks.values() if t.func == 'reconstruct_from_chunks' and t.data['round'] == round_nr]
        assert len(reconstructed) == sim.settings.sample_size
        for model in reconstructed:
            torch.testing.assert_close(ChunkManager.get_flat_params(model), expected)


@pytest.mark.asyncio
async def test_research_payloads_are_snapshots_and_training_does_not_mutate_predecessor(tmp_path):
    from dasklearn import functions
    sim = simulate(ConfluxV2Simulation, settings(ConfluxV2Settings, tmp_path, sample_size=2, chunks_in_sample=2))
    message = {'peers': [1]}
    sim.clients[0].send_message_to_client(1, 'ready', message)
    message['peers'].append(2)
    assert sim.events[-1][2].data['message'] == {'peers': [1]}
    functions.dataset = None
    functions.model_trainers = None
    model = functions.initialize(sim.settings, {})[0]
    before = copy.deepcopy(model.state_dict())
    functions.train(sim.settings, {'model': model, 'peer': 0, 'round': 1, 'time': 0})
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)


@pytest.mark.asyncio
async def test_research_v2_rejects_unsupported_churn(tmp_path):
    with pytest.raises(ValueError, match='stable membership'):
        ConfluxV2Simulation(settings(ConfluxV2Settings, tmp_path, sample_size=2, churn='synthetic'))


@pytest.mark.asyncio
async def test_research_sampled_clients_cannot_train_two_rounds_simultaneously(tmp_path):
    settings = ConfluxV2Settings(algorithm='conflux-v2', seed=1, work_dir=str(tmp_path), dataset='synthetic', learning=LearningSettings(0.1, 0, 0, 16, 2), participants=6, sample_size=4, chunks_in_sample=4, success_fraction=0.5, duration=5 * MICROSECONDS, rounds=0, unit_testing=True, test_interval=0, test_period=0)
    sim = ConfluxV2Simulation(settings)
    sim.setup_directories()

    def profile():
        for peer in sim.clients:
            peer.simulated_speed = 200 if peer.index == 0 else 1
    sim.apply_traces = profile
    sim.simulate()
    starts = defaultdict(list)
    for task in sim.workflow_dag.tasks.values():
        if task.func == 'train':
            starts[task.data['peer']].append(task.data['time'])
    for peer, times in starts.items():
        duration = sim.clients[peer].get_train_time()
        ordered = sorted(times)
        assert all((right - left >= duration for left, right in zip(ordered, ordered[1:])))
    assert len(starts) > 1


@pytest.mark.asyncio
async def test_research_local_steps_continue_across_epoch_boundary(tmp_path):
    settings = ConfluxV2Settings(algorithm='conflux-v2', seed=1, work_dir=str(tmp_path), dataset='synthetic', learning=LearningSettings(0.1, 0, 0, 128, 5), participants=2, sample_size=2, chunks_in_sample=2)
    from dasklearn.datasets import create_dataset
    from dasklearn.model_trainer import ModelTrainer
    trainer = ModelTrainer(create_dataset(settings), settings, 0)
    model = functions.initialize(settings, {})[0]
    result = trainer.train(model, 5, seed=1)
    assert result['samples'] == 5 * 128


@pytest.mark.asyncio
async def test_research_streaming_executor_preserves_every_snapshot(tmp_path):
    frames = []
    for retain in [True, False]:
        sim = simulate(ConfluxV2Simulation, settings(ConfluxV2Settings, tmp_path / str(retain), sample_size=3, chunks_in_sample=3))
        status = []
        result = execute_local(sim, retain_results=retain, progress=status.append)
        if not retain:
            assert result == {}
        assert status[-1]['completed'] == len(sim.workflow_dag.tasks)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv').sort_values(['time', 'peer']).reset_index(drop=True))
    pd.testing.assert_frame_equal(*frames)


@pytest.mark.asyncio
@pytest.mark.parametrize('cls,config,kwargs', [(ConfluxV2Simulation, ConfluxV2Settings, {'sample_size': 3, 'chunks_in_sample': 3})])
async def test_research_optimized_execution_matches_reference_bit_for_bit(tmp_path, monkeypatch, cls, config, kwargs):
    runs, frames = ([], [])
    for optimized in (False, True):
        Task.COUNTER.clear()
        sim = simulate(cls, settings(config, tmp_path / str(optimized), model_size_bytes=12000, **kwargs))
        assert sim.model_size == 12000
        sim.settings.release_cuda_cache = not optimized
        with monkeypatch.context() as patch:
            if not optimized:
                patch.setattr(functions, 'copy_model_state', wire_copy)
                patch.setattr(functions, 'create_model_structure', create_model)
            result = execute_local(sim, share_readonly_inputs=optimized)
        runs.append(result)
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv').sort_values(['time', 'peer']).reset_index(drop=True))
    assert runs[0].keys() == runs[1].keys()
    for name in runs[0]:
        for expected, actual in zip(runs[0][name], runs[1][name]):
            if isinstance(actual, torch.nn.Module):
                for key, value in actual.state_dict().items():
                    assert torch.equal(value, expected.state_dict()[key]), (name, key)
            else:
                assert torch.equal(actual, expected), name
    pd.testing.assert_frame_equal(*frames, check_exact=True)


@pytest.mark.asyncio
async def test_research_invalid_payload_size_rejected(tmp_path):
    with pytest.raises(ValueError, match='model_size_bytes must be positive'):
        simulate(ConfluxV2Simulation, settings(ConfluxV2Settings, tmp_path, sample_size=3, chunks_in_sample=3, model_size_bytes=0))


@pytest.mark.asyncio
async def test_research_gradient_only_copy_preserves_gradients_and_predecessor(tmp_path, monkeypatch):
    config = settings(ConfluxV2Settings, tmp_path, sample_size=3, chunks_in_sample=3)
    model = functions.initialize(config, {})[0]
    before = copy.deepcopy(model.state_dict())
    values = []
    for optimized in (False, True):
        functions.dataset = None
        functions.model_trainers = None
        with monkeypatch.context() as patch:
            if not optimized:
                patch.setattr(functions, 'copy_model_state', wire_copy)
            values.append(functions.compute_gradient(config, {'model': model, 'round': 1, 'time': 0, 'peer': 0})[0])
    for expected, actual in zip(values[0].gradient, values[1].gradient):
        assert torch.equal(actual, expected)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.asyncio
@pytest.mark.parametrize('cls,config,kwargs', [(ConfluxV2Simulation, ConfluxV2Settings, {'sample_size': 3, 'chunks_in_sample': 3})])
async def test_research_reconstruction_memo_preserves_all_snapshots(tmp_path, monkeypatch, cls, config, kwargs):
    frames = []
    reconstruct = functions.reconstruct_from_chunks
    for budget in (0, 1, 64 * 1024 * 1024):
        sim = simulate(cls, settings(config, tmp_path / str(budget), model_size_bytes=12000, **kwargs))
        tasks = [task for task in sim.workflow_dag.tasks.values() if task.func == 'reconstruct_from_chunks']
        unique = len({repr(task.data['chunks']) for task in tasks})
        calls = []

        def counted(*args):
            calls.append(1)
            return reconstruct(*args)
        status = []
        with monkeypatch.context() as patch:
            patch.setattr(functions, 'reconstruct_from_chunks', counted)
            execute_local(sim, retain_results=False, reconstruction_cache_bytes=budget, progress=status.append)
        assert len(calls) == (unique if budget > 1 else len(tasks))
        assert status[-1]['reconstruction_cache_peak_bytes'] <= budget
        assert status[-1]['reconstruction_cache_bytes'] == 0
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv').sort_values(['peer', 'time']).reset_index(drop=True))
    for actual in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], actual, check_exact=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('cls,setting,extra', [(ConfluxV2Simulation, ConfluxV2Settings, {'sample_size': 3, 'chunks_in_sample': 3})])
async def test_research_prefetch_preserves_dag_and_every_value(tmp_path, cls, setting, extra):
    results, frames = ([], [])
    for accelerate in (False, True):
        Task.COUNTER.clear()
        sim = simulate(cls, settings(setting, tmp_path / str(accelerate), model_size_bytes=12000, **extra))
        original = copy.deepcopy(sim.workflow_dag.serialize())
        backend = TrainingExecutor(sim.settings, autotune=False) if accelerate else None
        results.append(execute_local(sim, training_backend=backend))
        assert sim.workflow_dag.serialize() == original
        if accelerate:
            assert sim.execution_training_stats['batched_tasks'] > 0
        if accelerate:
            assert sim.execution_training_stats['prefetch_peak_models'] <= sim.settings.training_batch_size
        sim.merge_accuracies_files()
        frames.append(pd.read_csv(sim.data_dir + '/accuracies.csv').sort_values(['time', 'peer']).reset_index(drop=True))
    assert results[0].keys() == results[1].keys()
    for name, values in results[0].items():
        for a, b in zip(results[1][name], values):
            if isinstance(a, torch.nn.Module):
                assert_states(a, b)
            else:
                torch.testing.assert_close(a, b, rtol=0.0001, atol=1e-05)
    pd.testing.assert_frame_equal(*frames, rtol=1e-05, atol=1e-05)
