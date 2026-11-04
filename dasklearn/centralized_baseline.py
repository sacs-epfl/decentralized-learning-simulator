"""Step-budgeted pooled-data SGD, validation tuning and held-out readout.

This runner has no simulator, network or modeled training times. It keeps one
model on the selected device and iterates over the full single-node partition.
"""
from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from dasklearn.experiment_runner import _stop, provenance, write_json
from dasklearn.experiments import read_document, validate_options, validate_run


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _seeds(value, name):
    if (not isinstance(value, list) or not value or any(type(seed) is not int or seed < 0 for seed in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{name} must be unique nonnegative integer seeds')
    return sorted(value)


def load_config(path):
    from args import create_parser
    from dasklearn.session_settings import SessionSettings
    from dasklearn.settings_factory import session_options
    import argparse
    path, document = read_document(path, extra_fields={'training', 'tuning', 'evaluation'})
    if set(document) - {'version', 'name', 'output', 'settings', 'training', 'tuning', 'evaluation'}:
        raise ValueError('Centralized references use settings, training, tuning and evaluation sections')
    parser = create_parser()
    specific = validate_options(parser, document.get('settings', {}), path.parent)
    allowed = {'dataset', 'dataset_base_path', 'dataset_name', 'dataset_config', 'dataset_revision',
               'dataset_cache_dir', 'validation_set_fraction', 'model', 'device', 'batch_size',
               'momentum', 'weight_decay', 'local_steps', 'torch_threads'}
    if set(specific) - allowed:
        raise ValueError(f'Centralized settings must be from {sorted(allowed)}')
    options = vars(parser.parse_args([])) | dict(peers=1, partitioner='iid', local_steps=1, momentum=0,
                 evaluation_split='test', evaluation_examples=0, torch_threads=4, rounds=1, stop='rounds') | specific
    validate_run(options)
    if (options['local_steps'] != 1 or options['momentum'] != 0 or options['dataset'] not in {'cifar10', 'synthetic'}
            or options['dataset'] == 'cifar10' and (options['model'] != 'gnlenet' or not options['dataset_base_path'])):
        raise ValueError('Reference supports native CIFAR-10/GNLeNet or synthetic checks, local_steps=1 and momentum=0')
    training, tuning, evaluation = (document.get(key) for key in ('training', 'tuning', 'evaluation'))
    if not isinstance(training, dict) or set(training) != {'steps', 'evaluation_period'}:
        raise ValueError('training requires steps and evaluation_period')
    if not isinstance(tuning, dict) or set(tuning) != {'steps', 'seeds', 'learning_rates', 'evaluation_examples'}:
        raise ValueError('tuning requires steps, seeds, learning_rates and evaluation_examples')
    if not isinstance(evaluation, dict) or set(evaluation) != {'seeds'}:
        raise ValueError('evaluation requires seeds')
    for name, value in [('training.steps', training['steps']), ('training.evaluation_period', training['evaluation_period']),
                        ('tuning.steps', tuning['steps']), ('tuning.evaluation_examples', tuning['evaluation_examples'])]:
        _positive(value, name)
    tuning = deepcopy(tuning)
    tuning['seeds'] = _seeds(tuning['seeds'], 'tuning.seeds')
    final_seeds = _seeds(evaluation['seeds'], 'evaluation.seeds')
    if set(final_seeds) & set(tuning['seeds']):
        raise ValueError('Tuning and final seeds must be disjoint')
    rates = tuning['learning_rates']
    if (not isinstance(rates, list) or not rates or len(set(rates)) != len(rates)
            or any(type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0 for rate in rates)):
        raise ValueError('learning_rates must be unique positive finite numbers')
    if tuning['steps'] > training['steps']:
        raise ValueError('Tuning step budget cannot exceed final training steps')
    device = options['device']
    if device == 'auto' or not __import__('re').fullmatch(r'cpu|mps|cuda(?::[0-9]+)?', device):
        raise ValueError('Select a concrete execution device')
    settings = session_options(argparse.Namespace(**(options | dict(local=True))), device)
    settings = {key: value for key, value in settings.items() if key in {field.name for field in fields(SessionSettings)}}
    settings.update(algorithm='centralized', participants=1, churn='none', traces='none')
    for field in fields(SessionSettings):
        if settings.get(field.name) is None and isinstance(field.default, bool):
            settings[field.name] = field.default
    settings = SessionSettings(**settings).to_dict()
    output = (path.parent / document.get('output', '../output/centralized-reference')).resolve()
    return dict(version=1, name=document['name'], output=str(output), source=str(path),
                settings=settings, training=training, tuning=tuning, seeds=final_seeds,
                selection_rule='Highest mean endpoint validation accuracy across tuning seeds; lower LR breaks ties',
                checkpoint_rule='Lowest validation loss; earlier step breaks ties',
                time_basis='optimizer_steps')


def train_worker(document):
    """A single model/seed/LR with a persistent shuffled training loader."""
    import numpy as np
    import torch
    from dasklearn.datasets import create_dataset
    from dasklearn.model_evaluator import ModelEvaluator
    from dasklearn.model_trainer import ModelTrainer
    from dasklearn.models import create_model
    from dasklearn.session_settings import SessionSettings
    directory = Path(document['output'])
    settings = SessionSettings.from_dict(document['settings'])
    seed = settings.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings.torch_threads)
    device = settings.torch_device_name
    data = create_dataset(settings)
    trainer = ModelTrainer(data, settings, 0)
    loader = trainer.make_train_loader(seed=seed + 2026)
    if not len(loader):
        raise ValueError('Centralized training partition is empty')
    model = create_model(settings.dataset, architecture=settings.model, settings=settings).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning.learning_rate,
                                momentum=0, weight_decay=settings.learning.weight_decay)
    criterion = torch.nn.NLLLoss() if settings.dataset == 'cifar10' else torch.nn.CrossEntropyLoss()
    validation = ModelEvaluator(data, replace(settings, evaluation_split='validation',
        evaluation_examples=document['validation_examples']))
    started = time.monotonic()
    best_state, best = None, None
    curve, batches, samples = [], iter(loader), 0
    total = document['steps']
    metrics = dict(status='running', seed=seed, learning_rate=settings.learning.learning_rate,
                   completed_steps=0, total_steps=total, training_examples=len(trainer.partition),
                   validation_examples=len(data.load_split('validation')), test=None, curve=curve)

    def publish(step):
        metrics.update(completed_steps=step, samples_trained=samples,
                       elapsed_seconds=time.monotonic()-started, best_validation=best)
        write_json(directory / 'metrics.json', metrics)

    for step in range(total + 1):
        if step:
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            model.train()
            features, labels = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            samples += len(labels)
        if step == 0 or step == total or step % document['evaluation_period'] == 0:
            accuracy, loss = validation.evaluate_accuracy(model, device)
            model.to(device)  # Shared evaluator releases its GPU state after reading.
            finite = math.isfinite(accuracy) and math.isfinite(loss)
            point = dict(step=step, accuracy=accuracy if finite else None, loss=loss if finite else None,
                         elapsed_seconds=time.monotonic()-started)
            curve.append(point)
            if finite and (best is None or (loss, step) < (best['loss'], best['step'])):
                best = dict(step=step, accuracy=accuracy, loss=loss)
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if not finite and step == total:
                metrics.update(status='diverged', error='Nonfinite validation metrics at the full step budget')
            publish(step)
        elif step % 200 == 0:
            publish(step)
    # Tuning ranks only exact-budget validation scores. Final test is read once,
    # after all training, using the validation-selected checkpoint, never a test maximum.
    if metrics['status'] != 'diverged' and document['phase'] == 'evaluation':
        if best_state is None:
            raise ValueError('No finite validation checkpoint')
        model.load_state_dict(best_state)
        test = ModelEvaluator(data, replace(settings, evaluation_split='test', evaluation_examples=0))
        accuracy, loss = test.evaluate_accuracy(model, device)
        if not math.isfinite(accuracy) or not math.isfinite(loss):
            raise ValueError('Nonfinite held-out test evaluation')
        metrics['test'] = dict(accuracy=accuracy, loss=loss, checkpoint_step=best['step'],
                               examples=len(data.load_split('test')))
        # JSON metadata avoids serializing enum/custom settings objects, allowing
        # the checkpoint to load with PyTorch's default weights_only=True.
        torch.save(dict(state_dict=best_state, validation=best, settings=json.loads(settings.to_json())),
                   directory / 'best-model.pt')
    if metrics['status'] != 'diverged':
        metrics['status'] = 'completed'
    publish(total)
    import csv
    with (directory / 'validation-curve.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=['step', 'accuracy', 'loss', 'elapsed_seconds'])
        writer.writeheader()
        writer.writerows(curve)
    return metrics


def select_rate(records, spec):
    scores = []
    for index, rate in enumerate(spec['learning_rates']):
        runs = [row for row in records if row['rate_index'] == index]
        if ({row['seed'] for row in runs} != set(spec['seeds']) or len(runs) != len(spec['seeds'])
                or any(row['status'] not in {'completed', 'diverged'} or row['completed_steps'] != spec['steps'] for row in runs)):
            raise ValueError('Every LR trial must finish its allocated step budget')
        finite = all(row['status'] == 'completed' and row['completed_steps'] == spec['steps']
                     and row['curve'][-1]['step'] == spec['steps'] and row['curve'][-1]['accuracy'] is not None for row in runs)
        score = statistics.mean(row['curve'][-1]['accuracy'] for row in runs) if finite else None
        scores.append(dict(learning_rate=rate, mean_validation_accuracy=score))
    eligible = [score for score in scores if score['mean_validation_accuracy'] is not None]
    if not eligible:
        raise ValueError('No finite centralized LR candidate')
    winner = min(eligible, key=lambda row: (-row['mean_validation_accuracy'], row['learning_rate']))
    return dict(learning_rate=winner['learning_rate'], candidates=scores)


def reusable_tuning(config, previous):
    """Reuse the exact completed validation-only search; checkpoint choice is independent."""
    previous = Path(previous).resolve()
    resolved = json.loads((previous / 'resolved.json').read_text())
    if any(resolved[key] != config[key] for key in ('settings', 'tuning', 'seeds')):
        raise ValueError('Previous tuning has different data, optimizer, search budget or seeds')
    records = json.loads((previous / 'tuning/manifest.json').read_text())['runs']
    for record in records:
        directory = previous / 'tuning' / record['name']
        job = json.loads((directory / 'resolved.json').read_text())
        metrics = json.loads((directory / 'metrics.json').read_text())
        expected = deepcopy(config['settings'])
        expected.update(seed=record['seed'], torch_device_name=job['settings']['torch_device_name'],
                        work_dir=str(directory))
        expected['learning']['learning_rate'] = config['tuning']['learning_rates'][record['rate_index']]
        if (job['settings'] != expected or job['phase'] != 'tuning'
                or job['steps'] != config['tuning']['steps'] or job['evaluation_period'] != job['steps']
                or job['validation_examples'] != config['tuning']['evaluation_examples']
                or record['learning_rate'] != expected['learning']['learning_rate']
                or metrics['test'] is not None or metrics['status'] != record['status']
                or any(record[key] != metrics[key] for key in ('curve', 'completed_steps', 'seed', 'learning_rate'))):
            raise ValueError(f'Previous raw tuning trial differs: {record["name"]}')
        record.update(reused_from=str(directory))
    selection = select_rate(records, config['tuning'])
    if selection != {key: value for key, value in json.loads((previous / 'selection.json').read_text()).items()
                     if key in selection}:
        raise ValueError('Previous LR selection disagrees with the raw validation endpoints')
    return records, selection


def execute(config, *, devices, poll_seconds=1, reuse_tuning=None):
    """Detached-friendly process queue; reference lines wait for all final seeds."""
    output = Path(config['output'])
    reused = reusable_tuning(config, reuse_tuning) if reuse_tuning is not None else None
    output.mkdir(parents=True, exist_ok=False)
    source = Path(config['source'])
    (output / ('source' + source.suffix)).write_bytes(source.read_bytes())
    write_json(output / 'resolved.json', config)
    state = dict(version=1, kind='centralized-reference', name='Centralized', status='running', phase='tuning',
                 time_basis='optimizer_steps', settings=config['settings'], seeds=config['seeds'],
                 steps=config['training']['steps'], tuning=config['tuning'], learning_rate=None,
                 checkpoint_rule=config['checkpoint_rule'], selection_rule=config['selection_rule'],
                 reused_tuning_trials=len(reused[0]) if reused else 0,
                 reused_tuning_source=str(Path(reuse_tuning).resolve()) if reused else None,
                 runs=[], accuracy=None, loss=None, started_at=datetime.now(timezone.utc).isoformat(),
                 provenance=provenance(Path(__file__).resolve().parents[1]))

    def publish():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'reference.json', state)

    publish()
    active = {}
    try:
        for phase in ('tuning', 'evaluation'):
            state['phase'] = phase
            if phase == 'tuning' and reused:
                records, selection = reused
                state.update(runs=records, learning_rate=selection['learning_rate'])
                (output / 'tuning').mkdir()
                write_json(output / 'tuning/manifest.json', dict(phase=phase, runs=records))
                write_json(output / 'selection.json', selection | dict(rule=config['selection_rule'],
                           reused_from=state['reused_tuning_source']))
                publish()
                continue
            if phase == 'tuning':
                records = [dict(name=f'lr{index+1:02d}-s{seed}', seed=seed, rate_index=index, learning_rate=rate,
                                status='planned') for index, rate in enumerate(config['tuning']['learning_rates'])
                           for seed in config['tuning']['seeds']]
            else:
                records = [dict(name=f'centralized-s{seed}', seed=seed, learning_rate=state['learning_rate'],
                                status='planned') for seed in config['seeds']]
            state['runs'] = records
            position = 0
            while position < len(records) or active:
                for slot, device in enumerate(devices):
                    if slot in active or position >= len(records):
                        continue
                    record = records[position]; position += 1
                    directory = output / phase / record['name']; directory.mkdir(parents=True)
                    settings = deepcopy(config['settings'])
                    settings.update(seed=record['seed'], torch_device_name=device, work_dir=str(directory))
                    settings['learning']['learning_rate'] = record['learning_rate']
                    tuning = phase == 'tuning'
                    job = dict(settings=settings, output=str(directory), phase=phase,
                               steps=config['tuning']['steps'] if tuning else config['training']['steps'],
                               evaluation_period=config['tuning']['steps'] if tuning else config['training']['evaluation_period'],
                               validation_examples=config['tuning']['evaluation_examples'] if tuning else 0)
                    write_json(directory / 'resolved.json', job)
                    log = (directory / 'worker.log').open('w')
                    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/run_centralized_baseline.py'),
                               '--worker', str(directory / 'resolved.json')]
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                               start_new_session=True)
                    active[slot] = dict(process=process, record=record, directory=directory, log=log)
                    record.update(status='running', device=device, pid=process.pid, total_steps=job['steps'], completed_steps=0)
                    print(f'{phase}: {record["name"]} on {device}', flush=True)
                for slot, job in list(active.items()):
                    record, process = job['record'], job['process']
                    metrics = job['directory'] / 'metrics.json'
                    if metrics.exists():
                        record.update(json.loads(metrics.read_text()))
                        record['status'] = 'running'
                    code = process.poll()
                    if code is not None:
                        if code != 0 or not metrics.exists():
                            record.update(status='failed', error=f'Worker exited {code}; see {job["directory"] / "worker.log"}')
                        else:
                            record.update(json.loads(metrics.read_text()))
                        job['log'].close(); del active[slot]
                publish()
                if active:
                    time.sleep(poll_seconds)
            write_json(output / phase / 'manifest.json', dict(phase=phase, runs=records))
            if phase == 'tuning':
                selection = select_rate(records, config['tuning'])
                state['learning_rate'] = selection['learning_rate']
                write_json(output / 'selection.json', selection | dict(rule=config['selection_rule']))
            else:
                if any(record['status'] != 'completed' or record.get('test') is None for record in records):
                    raise ValueError('Centralized evaluation did not complete on every seed')
                for metric in ('accuracy', 'loss'):
                    values = [record['test'][metric] for record in records]
                    state[metric] = dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else None)
                state.update(status='completed', phase='finished')
        publish()
        return 0
    except KeyboardInterrupt:
        state.update(status='interrupted', error='Centralized reference interrupted')
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.update(status='failed', error=str(exc))
        print(str(exc), flush=True)
        return 1
    finally:
        for job in active.values():
            _stop(job['process']); job['log'].close()
            job['record'].update(status='interrupted')
        publish()
