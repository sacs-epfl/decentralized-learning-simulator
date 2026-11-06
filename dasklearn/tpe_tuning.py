"""Batched, validation-only TPE with full paired trials and no pruning.

Optuna is confined to proposing configurations. The benchmark runner owns worker
isolation, device slots, budgets and measurements; the grid scorer owns the metric.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import time

from dasklearn.benchmark import BenchmarkRun, TUNABLE_OPTIONS, TUNABLE_PREFIXES, save_results
from dasklearn.experiment_runner import write_json
from dasklearn.experiments import validate_options, validate_run


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ValueError('TPE search requires Optuna; install the simulator requirements') from exc
    return optuna


def validate_search(spec, benchmark, names, selected_names):
    """Validate distributions without importing Optuna, models, or datasets."""
    if 'learning_rates' in spec:
        raise ValueError('TPE uses search_spaces instead of learning_rates')
    spaces = spec.get('search_spaces')
    if not isinstance(spaces, dict) or set(spaces) != names:
        raise ValueError('tuning.search_spaces must cover exactly the named algorithm variants')
    counts = dict(n_trials=40, n_startup_trials=12, sampler_seed=0,
                  batch_size=max(1, math.ceil(len(benchmark.execution_devices or [None]) / len(spec.get('seeds', [0])))))
    for key, default in counts.items():
        value = spec.get(key, default)
        if type(value) is not int or value < (0 if key == 'sampler_seed' else 1):
            raise ValueError(f'tuning.{key} must be a {"nonnegative" if key == "sampler_seed" else "positive"} integer')
        counts[key] = value
    if counts['n_startup_trials'] >= counts['n_trials']:
        raise ValueError('TPE n_startup_trials must be smaller than n_trials to allow adaptive suggestions')
    if counts['batch_size'] > counts['n_startup_trials']:
        raise ValueError('TPE batch_size must not exceed n_startup_trials')
    groups = 1 if spec.get('select_variant') else len(selected_names)
    if counts['sampler_seed']+groups-1 >= 2**32:
        raise ValueError('tuning.sampler_seed and study offsets must fit an unsigned 32-bit seed')
    if counts['n_trials'] * len(spec['seeds']) * groups > 10000:
        raise ValueError('TPE search exceeds 10,000 seed runs')
    from args import create_parser
    parser = create_parser()
    templates = {run.variant: run for run in benchmark.runs}
    for variant, space in spaces.items():
        if not isinstance(space, dict) or not space:
            raise ValueError(f'search_spaces.{variant} must be a nonempty parameter mapping')
        for name, distribution in space.items():
            if (not isinstance(name, str) or name == 'algorithm' or '-' in name
                    or name not in TUNABLE_OPTIONS and not name.startswith(TUNABLE_PREFIXES)):
                raise ValueError(f'{name!r} is not a searchable algorithm parameter')
            if not isinstance(distribution, dict):
                raise ValueError(f'{name} requires a distribution mapping')
            kind = distribution.get('type')
            if kind == 'categorical':
                choices = distribution.get('choices')
                if (set(distribution) != {'type', 'choices'} or not isinstance(choices, list) or not choices
                        or any(type(value) not in (str, int, float, bool) or
                               isinstance(value, float) and not math.isfinite(value) for value in choices)
                        or len(set(choices)) != len(choices)):
                    raise ValueError(f'{name} requires unique finite scalar categorical choices')
                probes = choices
            elif isinstance(kind, str) and kind in {'float', 'int'}:
                if set(distribution) - {'type', 'low', 'high', 'log', 'step'}:
                    raise ValueError(f'Unknown distribution fields for {name}')
                low, high = distribution.get('low'), distribution.get('high')
                types = (int,) if kind == 'int' else (int, float)
                if (type(low) not in types or type(high) not in types or not math.isfinite(low)
                        or not math.isfinite(high) or low >= high):
                    raise ValueError(f'{name} requires finite {kind} bounds low < high')
                log = distribution.get('log', False)
                step = distribution.get('step', 1 if kind == 'int' else None)
                if type(log) is not bool or log and low <= 0:
                    raise ValueError(f'{name} log sampling requires a boolean log and positive bounds')
                if (kind == 'int' and step is None
                        or step is not None and (type(step) not in types or not math.isfinite(step) or step <= 0)
                        or log and step is not None and not (kind == 'int' and step == 1)):
                    raise ValueError(f'{name} has an invalid step/log combination')
                if step is not None and not math.isclose((high-low)/step, round((high-low)/step), abs_tol=1e-9):
                    raise ValueError(f'{name} bounds must be aligned to step')
                probes = [float(low), float(high)] if kind == 'float' else [low, high]
            else:
                raise ValueError(f'{name} distribution type must be float, int, or categorical')
            for value in probes:
                normalized = validate_options(parser, {name: value}, benchmark.path.parent)
                if variant in templates:
                    validate_run(templates[variant].options | normalized)
    return dict(method='tpe', search_spaces={name: spaces[name] for name in selected_names}, **counts)


def study_groups(spec):
    variants = list(spec['search_spaces'])
    return {'joint': variants} if spec.get('select_variant') else {name: [name] for name in variants}


def candidate_runs(benchmark, candidates):
    templates = {run.variant: run for run in benchmark.runs}
    runs = []
    spec = benchmark.tuning
    for candidate in candidates:
        for seed in spec['seeds']:
            options = templates[candidate['variant']].options | candidate['parameters'] | dict(
                seed=seed, duration=spec['budget'], evaluation_split='validation',
                evaluation_examples=spec['evaluation_examples'])
            validate_run(options)
            runs.append(BenchmarkRun(f"{candidate['id']}-s{seed}", candidate['id'], seed, options))
    return runs


def tuning_suite(benchmark, candidates):
    spec = benchmark.tuning
    times = sorted({0, *range(spec['evaluation_period'], spec['budget']+1, spec['evaluation_period']), spec['budget']})
    return replace(benchmark, name=benchmark.name+' · TPE tuning', output=benchmark.output/'tuning',
                   budgets=[spec['budget']], seeds=spec['seeds'], times=times,
                   runs=candidate_runs(benchmark, candidates), tuning=None)


def checked_records(benchmark, result, candidates):
    """Verify trial identities and every numerical setting, including on recovery."""
    if result.get('evaluation_model', 'local') != benchmark.evaluation_model:
        raise ValueError('Tuning and final evaluation must use the same model readout')
    expected = candidate_runs(benchmark, candidates)
    by_name = {record['name']: record for record in result['runs']}
    if len(by_name) != len(result['runs']) or set(by_name) != {run.name for run in expected}:
        raise ValueError('Incomplete or duplicate TPE seed runs')
    for run in expected:
        record = by_name[run.name]
        # Device assignment is an execution detail; every numerical setting is fixed.
        actual = {key: value for key, value in record['options'].items() if key != 'device'}
        declared = {key: value for key, value in run.options.items() if key != 'device'}
        if actual != declared or record['variant'] != run.variant or record['seed'] != run.seed:
            raise ValueError('TPE trial settings differ from the declared candidate')
    return by_name


def candidate_scores(benchmark, result, candidates):
    """Verify the actual options and every seed before feeding an observation to TPE."""
    from dasklearn.lr_tuning import score_candidate
    by_name = checked_records(benchmark, result, candidates)
    return {candidate['id']: score_candidate(
        [by_name[f"{candidate['id']}-s{seed}"] for seed in benchmark.tuning['seeds']],
        benchmark.tuning, benchmark.evaluation_model) for candidate in candidates}


def select_search(benchmark, result, candidates):
    spec = benchmark.tuning
    groups = study_groups(spec)
    if result.get('status') != 'completed' or len(candidates) != len(groups)*spec['n_trials']:
        raise ValueError('All declared TPE candidates must complete before selection')
    ids = {row['id'] for row in candidates}
    if len(ids) != len(candidates):
        raise ValueError('Duplicate TPE candidate IDs')
    for group, variants in groups.items():
        rows = [row for row in candidates if row['study'] == group]
        if (sorted(row['number'] for row in rows) != list(range(spec['n_trials']))
                or any(row['variant'] not in variants for row in rows)):
            raise ValueError('Incomplete TPE study')
    scores = candidate_scores(benchmark, result, candidates)
    templates = {run.variant: run for run in benchmark.runs}
    direction = 1 if spec['metric'] == 'loss' else -1
    winners, table = {}, {name: [] for name in spec['search_spaces']}
    for candidate in candidates:
        row = dict(scores[candidate['id']], candidate_id=candidate['id'], parameters=candidate['parameters'],
                   learning_rate=candidate['parameters'].get('learning_rate', templates[candidate['variant']].options['learning_rate']))
        table[candidate['variant']].append(row)
        if row['mean_score'] is not None:
            key = (direction*row['mean_score'], row['learning_rate'], candidate['variant'], candidate['id'])
            group = candidate['study']
            if group not in winners or key < winners[group][0]:
                winners[group] = (key, candidate, row)
    if set(winners) != set(groups):
        raise ValueError('No finite TPE candidate for every declared algorithm study')
    selected, parameters, selected_candidates = {}, {}, {}
    for _, candidate, row in winners.values():
        variant = candidate['variant']
        selected[variant] = row['learning_rate']
        parameters[variant] = candidate['parameters']
        selected_candidates[variant] = candidate['id']
    rule = (f"{'Minimize' if direction == 1 else 'Maximize'} mean validation {spec['metric']} across all paired seeds; "
            'lower LR, then variant name and candidate ID, break between-candidate ties; divergent candidates are ineligible')
    if spec['metric'] == 'best_mean_accuracy':
        rule = ('Maximize validation accuracy averaged across all tuning seeds at one shared scheduled training checkpoint '
                f'within {spec["budget"]} simulated seconds; earliest checkpoint breaks within-candidate ties; '
                'lower LR, then variant name and candidate ID, break between-candidate ties; divergent candidates are ineligible')
    selection = dict(version=1, method='tpe', selected=selected, selected_parameters=parameters,
                     selected_candidates=selected_candidates, scores=table, specification=spec,
                     selection_split='validation', evaluation_split='test', evaluation_seeds=benchmark.seeds, rule=rule)
    if spec.get('select_variant'):
        _, candidate, row = winners['joint']
        selection.update(selected_variant=candidate['variant'], selected_learning_rate=row['learning_rate'])
        if spec['metric'] == 'best_mean_accuracy':
            selection['selected_time'] = row['selected_time']
    else:
        selection['selected_times'] = {candidate['variant']: row['selected_time'] for _, candidate, row in winners.values()
                                       if row.get('selected_time') is not None}
    return selection


def execute_search(benchmark, *, on_update=None, resume=False):
    """Ask/tell in deterministic batches, sharing device slots across all studies."""
    from dasklearn.benchmark import execute_benchmark
    optuna = require_optuna()
    spec = benchmark.tuning
    studies = {group: optuna.create_study(direction='minimize' if spec['metric'] == 'loss' else 'maximize',
                    sampler=optuna.samplers.TPESampler(seed=spec['sampler_seed']+index,
                        n_startup_trials=spec['n_startup_trials'], constant_liar=True),
                    pruner=optuna.pruners.NopPruner())
               for index, group in enumerate(study_groups(spec))}
    candidates, records = [], {}
    output = benchmark.output/'tuning'
    saved_candidates = []
    previous = None
    if resume:
        from dasklearn.lr_tuning import score_candidate
        previous = json.loads((output/'search.json').read_text())
        if previous['specification'] != spec or previous['optuna_version'] != optuna.__version__:
            raise ValueError('Recovery requires the same TPE specification and Optuna version')
        saved_candidates = previous['candidates']
        saved_report = json.loads((output/'leaderboard.json').read_text())
        records = checked_records(benchmark, saved_report, saved_candidates)
        for record in records.values():
            if record['status'] in {'completed', 'diverged'}:
                score_candidate([record], spec | dict(seeds=[record['seed']]), benchmark.evaluation_model)
    else:
        output.mkdir()
    state = dict(version=1, method='tpe', status='running', specification=spec, optuna_version=optuna.__version__,
                 sampler='TPESampler', sampler_seed=spec['sampler_seed'], constant_liar=True, pruning=False,
                 scheduling='deterministic batches; tell in candidate order after every seed finishes',
                 candidates=candidates, started_at=datetime.now(timezone.utc).isoformat())
    elapsed_before = previous.get('elapsed_seconds', 0) if previous else 0
    if previous:
        state.update(started_at=previous['started_at'], provenance=previous.get('provenance'),
                     resumptions=previous.get('resumptions', []) + [dict(
                         resumed_at=datetime.now(timezone.utc).isoformat(),
                         retained_seed_runs=sum(record['status'] in {'completed', 'diverged'} for record in records.values()),
                         previous_status=previous['status'], previous_provenance=previous.get('provenance'))])
    started = time.monotonic()
    recovery_verified = not resume

    def publish(status='running', current=None):
        suite = tuning_suite(benchmark, candidates).manifest()
        observed = records | {record['name']: record for record in current or []}
        suite.update(status=status, provenance=state.get('provenance'), runs=[observed.get(run['name'], dict(run, status='planned', curve=[]))
                                          for run in suite['runs']])
        report = save_results(output, suite)
        state['status'] = status
        state['elapsed_seconds'] = elapsed_before + time.monotonic()-started
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(output/'search.json', state)
        if on_update:
            on_update(report)

    def update(result):
        state['provenance'] = result.get('provenance')
        publish(current=result['runs'])

    try:
        for start in range(0, spec['n_trials'], spec['batch_size']):
            batch, pending = [], []
            for group, variants in study_groups(spec).items():
                study = studies[group]
                for _ in range(min(spec['batch_size'], spec['n_trials']-start)):
                    trial = study.ask()
                    variant = trial.suggest_categorical('variant', variants) if len(variants)>1 else variants[0]
                    parameters = {}
                    for name, distribution in spec['search_spaces'][variant].items():
                        key = f'{variant}.{name}'
                        kwargs = {key: value for key, value in distribution.items() if key != 'type'}
                        parameters[name] = getattr(trial, 'suggest_'+distribution['type'])(key, **kwargs)
                    candidate = dict(id=f'tpe{len(candidates):05d}', study=group, number=trial.number,
                                     variant=variant, parameters=parameters, status='running',
                                     sampling_phase='adaptive' if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
                                     >= spec['n_startup_trials'] else 'startup')
                    if len(candidates) < len(saved_candidates):
                        saved = saved_candidates[len(candidates)]
                        if any(candidate[key] != saved[key] for key in ('id', 'study', 'number', 'variant', 'parameters', 'sampling_phase')):
                            raise ValueError('Replayed Optuna proposal differs from the saved search; recovery aborted')
                        candidate = dict(saved)
                    candidates.append(candidate)
                    batch.append(candidate)
                    pending.append((study, trial, candidate))
            recovery_verified = len(candidates) >= len(saved_candidates)
            trials = tuning_suite(benchmark, batch)
            remaining = [run for run in trials.runs if records.get(run.name, {}).get('status') not in {'completed', 'diverged'}]
            if remaining:
                for candidate in batch:
                    candidate['status'] = 'running'
                planned = [dict(asdict(run), status='planned', curve=[]) for run in remaining]
                publish(current=planned)
                directory = output/f'batch-{start//spec["batch_size"]:04d}'
                retry = 0
                while directory.exists():
                    retry += 1
                    directory = output/f'batch-{start//spec["batch_size"]:04d}-recovery-{retry:04d}'
                trials = replace(trials, output=directory, runs=remaining)
                code = execute_benchmark(trials, allow_divergent_trials=True, on_update=update)
                result = json.loads((trials.output/'leaderboard.json').read_text())
                for record in result['runs']:
                    record['trial_directory'] = str(trials.output/'runs'/record['name'])
                    records[record['name']] = record
                if code:
                    for study, trial, candidate in pending:
                        candidate['status'] = 'interrupted' if code == 130 else 'failed'
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    publish('interrupted' if code == 130 else 'failed')
                    return code
            result = tuning_suite(benchmark, batch).manifest() | dict(
                runs=[records[run.name] for run in tuning_suite(benchmark, batch).runs])
            scores = candidate_scores(benchmark, result, batch)
            for study, trial, candidate in pending:
                score = scores[candidate['id']]
                if candidate['status'] in {'completed', 'diverged'} and candidate.get('mean_score') != score['mean_score']:
                    raise ValueError('Saved TPE score differs from the retained observations')
                candidate.update(score, status='completed' if score['mean_score'] is not None else 'diverged')
                # Divergence is a completed full-budget observation of a bad configuration.
                value = score['mean_score']
                study.tell(trial, value if value is not None else (float('inf') if spec['metric']=='loss' else -float('inf')))
            # Replay historical batches silently; publish once recovery reaches new work.
            if remaining or len(candidates) >= len(saved_candidates):
                publish()
        publish('completed')
        return 0
    except KeyboardInterrupt:
        if recovery_verified:
            publish('interrupted')
        return 130
    except (OSError, ValueError, KeyError, TypeError):
        if recovery_verified:
            publish('failed')
        raise


def search_progress(spec, search, report):
    """Public progress uses recorded observations only, including full seed groups."""
    candidates = search.get('candidates', []) if search else []
    records = report.get('runs', []) if report else []
    minimize = spec['metric'] == 'loss'
    studies, history = [], []
    for group in study_groups(spec):
        rows = [row for row in candidates if row['study'] == group]
        finished = [row for row in rows if row['status'] in {'completed', 'diverged'}]
        best = None
        for row in finished:
            value = row.get('mean_score')
            if value is not None and (best is None or (value < best['mean_score'] if minimize else value > best['mean_score'])):
                best = row
            history.append(dict(study=group, candidate_id=row['id'], number=row['number']+1,
                                score=value, best_score=best['mean_score'] if best else None,
                                sampling_phase=row['sampling_phase']))
        studies.append(dict(study=group, proposed=len(rows), completed=len(finished), total=spec['n_trials'],
                            phase='completed' if len(finished)==spec['n_trials'] else
                                  'adaptive' if len(finished)>=spec['n_startup_trials'] else 'startup',
                            best_candidate=best))
    completed_seeds = sum(row['status'] in {'completed', 'diverged'} for row in records)
    return dict(studies=studies, history=history, proposed=len(candidates),
                completed=sum(row['completed'] for row in studies), total=spec['n_trials']*len(studies),
                divergent=sum(row['status']=='diverged' for row in candidates),
                running_seeds=sum(row['status']=='running' for row in records), completed_seeds=completed_seeds,
                allocated_simulated_seconds=spec['budget']*len(spec['seeds'])*spec['n_trials']*len(studies),
                completed_simulated_seconds=completed_seeds*spec['budget'],
                elapsed_seconds=search.get('elapsed_seconds') if search else None,
                optuna_version=search.get('optuna_version') if search else None,
                sampler_seed=spec['sampler_seed'], updated_at=search.get('updated_at') if search else None)
