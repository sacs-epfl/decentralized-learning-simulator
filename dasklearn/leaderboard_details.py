"""Algorithm explanations and live tuning evidence from an explicit catalog."""
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import re
import statistics
from urllib.parse import urlsplit


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def load_algorithms(catalog):
    if catalog is None:
        return None
    catalog = Path(catalog).resolve()
    source = read_json(catalog).get('algorithms')
    if source is None:
        return None
    if not isinstance(source, str) or not source.strip():
        raise ValueError('algorithms must name an algorithm registry file')
    path = (catalog.parent / source).resolve()
    registry = read_json(path)
    if not isinstance(registry, dict) or registry.get('version') != 1 or not registry.get('algorithms'):
        raise ValueError('An algorithm registry requires version 1 and nonempty algorithms')
    from args import create_parser
    from dasklearn.benchmark import load_benchmark
    identifiers = set()
    for entry in registry['algorithms']:
        identifier = entry['id']
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', identifier) or identifier in identifiers:
            raise ValueError('Algorithm IDs must be valid and unique')
        identifiers.add(identifier)
        for link in (entry.get('paper'), entry.get('author_code'), entry.get('supplement')):
            if link and (urlsplit(link['url']).scheme != 'https' or not urlsplit(link['url']).netloc):
                raise ValueError('Paper and code links require absolute HTTPS URLs')
        entry['_benchmark'] = load_benchmark(path.parent / entry['config'], create_parser())
        evaluation = entry.get('evaluation')
        if evaluation is not None and (not isinstance(evaluation, dict)
                or set(evaluation) != {'policy', 'primary_time'} or evaluation['policy'] != 'fixed_endpoint'
                or type(evaluation['primary_time']) is not int
                or evaluation['primary_time'] not in entry['_benchmark'].budgets):
            raise ValueError('Fixed evaluation endpoint must be a declared budget')
        if entry['tuning_setting'] not in registry['scenarios']:
            raise ValueError('Unknown tuning setting')
    return registry


def public_algorithms(registry):
    return [dict(id=entry['id'], label=entry['label'], variants=sorted({run.variant for run in entry['_benchmark'].runs}))
            for entry in registry['algorithms']] if registry else []


def algorithm_entry(registry, identifier):
    return next((entry for entry in registry['algorithms'] if entry['id'] == identifier), None) if registry else None


def scenario_benchmark(entry, registry, setting):
    benchmark = entry['_benchmark']
    scenario = registry['scenarios'][setting]
    return replace(benchmark, network=benchmark.network | dict(profile=scenario['profile']),
                   runs=[replace(run, options=run.options | dict(partitioner=scenario['partitioner'], alpha=.1))
                         for run in benchmark.runs])


def campaign_status(output):
    """Prefer measured campaign state, falling back to the detached queue."""
    return (read_json(output / 'campaign.json') or read_json(output / 'pipeline.json') or
            read_json(output.with_name(output.name + '-launch') / 'queue.json'))


def pending_leaderboard(registry, setting):
    """Show registered work without fabricating measurements or creating run dirs."""
    from dasklearn.benchmark import checkpoint_budget_view, leaderboard
    templates, records = [], []
    status = 'planned'
    for entry in registry['algorithms']:
        benchmark = scenario_benchmark(entry, registry, setting)
        templates.append(benchmark)
        selection = read_json(benchmark.output / 'selection.json')
        pipeline = campaign_status(benchmark.output)
        if pipeline:
            status = pipeline['status'] if pipeline['status'] != 'completed' else 'planned'
        winner = selection.get('selected_variant') if selection else None
        for seed in benchmark.seeds:
            run = next(run for run in benchmark.runs if run.seed == seed and (winner is None or run.variant == winner))
            options = run.options | (selection.get('selected_parameters', {}).get(winner, {}) if winner else {})
            options.update(learning_rate=selection['selected'][winner] if winner else None,
                           local_steps=options['local_steps'] if winner else None)
            records.append(dict(name=f"{entry['id']}-s{seed}", variant=winner or entry['id'], seed=seed,
                                options=options, status='planned', curve=[]))
    suite = templates[0].manifest() | dict(name=registry['title'], status=status, runs=records)
    return checkpoint_budget_view(leaderboard(suite), 600, times=templates[0].times, seeds=templates[0].seeds)


def annotate_leaderboard(document, registry):
    if registry:
        document = dict(document, algorithms=public_algorithms(registry), campaign_notice=registry['notice'])
        policies = [entry.get('evaluation') for entry in registry['algorithms']]
        if any(policy is not None for policy in policies):
            if any(policy != policies[0] for policy in policies):
                raise ValueError('Registered algorithms must use the same primary evaluation policy')
            document['evaluation'] = policies[0]
    return document


def combined_leaderboard(registry, setting):
    """Combine registered held-out runs at read time; never rewrite experiment files."""
    from dasklearn.benchmark import checkpoint_budget_view, leaderboard
    records, statuses, provenances, selections = [], [], {}, {}
    template = None
    budgets = set()
    for entry in registry['algorithms']:
        benchmark = scenario_benchmark(entry, registry, setting)
        expected = benchmark.manifest()
        if template is None:
            template = expected
        for key in ('seeds', 'times', 'network', 'evaluation_model', 'accuracy_targets'):
            if expected[key] != template[key]:
                raise ValueError(f"Registered algorithms have different {key}")
        for key in ('dataset', 'model', 'peers', 'partitioner', 'alpha', 'batch_size', 'evaluation_split',
                    'evaluation_examples', 'validation_set_fraction', 'network_latency_ms'):
            if expected['runs'][0]['options'][key] != template['runs'][0]['options'][key]:
                raise ValueError(f"Registered algorithms have different {key}")
        budgets.update(benchmark.budgets)
        phase = 'evaluation' if setting == entry['tuning_setting'] else 'evaluation-'+setting
        suite = read_json(benchmark.output / phase / 'manifest.json')
        if suite is None:
            pending = pending_leaderboard(registry | dict(algorithms=[entry]), setting)
            records.extend(pending['runs'])
            statuses.append(pending['status'])
        else:
            for key in ('seeds', 'times', 'network', 'evaluation_model', 'accuracy_targets'):
                if suite[key] != expected[key]:
                    raise ValueError(f"Recorded {entry['id']} has different {key}")
            if {run['seed'] for run in suite['runs']} != set(benchmark.seeds):
                raise ValueError(f"Recorded {entry['id']} has different evaluation seeds")
            for run in suite['runs']:
                for key in ('dataset', 'model', 'peers', 'partitioner', 'alpha', 'evaluation_split', 'evaluation_examples'):
                    if run['options'][key] != expected['runs'][0]['options'][key]:
                        raise ValueError(f"Recorded {entry['id']} has different {key}")
            records.extend(suite['runs'])
            statuses.append(suite['status'])
            budgets.update(suite['budgets'])
            provenances[entry['id']] = suite.get('provenance')
        selection = read_json(benchmark.output / 'selection.json')
        if selection:
            selections[entry['id']] = selection
    if len({record['name'] for record in records}) != len(records):
        raise ValueError('Registered algorithms have duplicate evaluation run names')
    status = ('failed' if 'failed' in statuses else 'interrupted' if 'interrupted' in statuses else
              'completed' if all(state=='completed' for state in statuses) else
              'running' if 'running' in statuses else 'planned')
    selection = dict(selected_parameters={name: params for item in selections.values()
                                          for name, params in item.get('selected_parameters', {}).items()},
                     selected_times={name: item['selected_time'] for name, item in selections.items()
                                     if item.get('selected_time') is not None})
    times = set(selection['selected_times'].values())
    if len(selections)==len(registry['algorithms']) and len(times)==1:
        selection['selected_time'] = next(iter(times))
    suite = template | dict(name=registry['title'], runs=records, budgets=sorted(budgets), status=status,
                            provenance=dict(algorithms=provenances), selection=selection, tuning=None)
    result = checkpoint_budget_view(leaderboard(suite), 600, times=template['times'], seeds=template['seeds'])
    return annotate_leaderboard(result, registry)


def pending_csv(document):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['budget', 'variant', 'rank', 'provisional', 'seeds', 'learning_rate'])
    for table in document['tables']:
        for row in table['rows']:
            writer.writerow([table['budget'], row['variant'], '', True, 0, row['learning_rate']])
    return stream.getvalue().encode()


def algorithm_document(entry, registry, setting):
    benchmark = scenario_benchmark(entry, registry, setting)
    spec = benchmark.tuning
    output = benchmark.output
    selection = read_json(output / 'selection.json')
    pipeline = campaign_status(output)
    report = read_json(output / 'tuning' / 'leaderboard.json')
    records = {run['name']: run for run in report['runs']} if report else {}
    groups, trials = [], []
    templates = {run.variant: run for run in benchmark.runs}
    adaptive = spec.get('method') == 'tpe'
    search = read_json(output / 'tuning' / 'search.json') if adaptive else None
    candidates = []
    if adaptive:
        for candidate in search['candidates'] if search else []:
            options = templates[candidate['variant']].options | candidate['parameters']
            candidates.append(dict(variant=candidate['variant'], label=candidate['id'],
                                   rate=options['learning_rate'], options=options, parameters=candidate['parameters']))
    else:
        for variant, rates in spec['learning_rates'].items():
            for index, rate in enumerate(rates, 1):
                candidates.append(dict(variant=variant, label=f'{variant}-lr{index:02d}', rate=rate,
                                       options=templates[variant].options, parameters=dict(learning_rate=rate)))
    for candidate in candidates:
        variant, rate, label = candidate['variant'], candidate['rate'], candidate['label']
        scores, losses, statuses, wall_times = [], [], [], []
        for seed in spec['seeds']:
            name = f'{label}-s{seed}'
            record = records.get(name, {})
            status = record.get('status', 'planned')
            statuses.append(status)
            wall_seconds = record.get('wall_seconds')
            if wall_seconds is not None:
                wall_times.append(wall_seconds)
            point = next((point for point in record.get('curve', []) if point['time'] == spec['budget']), None)
            complete = status == 'completed' and point is not None
            accuracy = point['accuracy'] if complete else None
            loss = point['loss'] if complete else None
            if complete:
                scores.append(accuracy)
                losses.append(loss)
            trials.append(dict(name=name, local_steps=candidate['options']['local_steps'], parameters=candidate['parameters'],
                               learning_rate=rate, seed=seed, status=status, accuracy=accuracy, loss=loss,
                               wall_seconds=wall_seconds, error=record.get('error')))
        complete = len(scores) == len(spec['seeds'])
        selected_time = spec['budget']
        if spec['metric']=='best_mean_accuracy':
            selected_time = None
            if complete:
                from dasklearn.lr_tuning import best_mean_checkpoint
                group_records = [records[f'{label}-s{seed}'] for seed in spec['seeds']]
                selected_time = best_mean_checkpoint(group_records, spec)
                points = [next(point for point in record['curve'] if point['time']==selected_time) for record in group_records]
                scores = [point['accuracy'] for point in points]
                losses = [point['loss'] for point in points]
        state = 'completed' if complete else ('diverged' if 'diverged' in statuses else
                'failed' if 'failed' in statuses else 'interrupted' if 'interrupted' in statuses else
                'running' if 'running' in statuses or scores else 'planned')
        groups.append(dict(variant=variant, candidate_id=label, parameters=candidate['parameters'], local_steps=candidate['options']['local_steps'],
                           learning_rate=rate, completed_seeds=sum(status in {'completed', 'diverged'} for status in statuses), expected_seeds=len(spec['seeds']),
                           status=state, accuracy=dict(mean=statistics.mean(scores), sd=statistics.stdev(scores) if len(scores)>1 else None) if complete else None,
                           loss=dict(mean=statistics.mean(losses), sd=statistics.stdev(losses) if len(losses)>1 else None) if complete else None,
                           selected_time=selected_time,
                           wall_seconds=statistics.mean(wall_times) if wall_times else None, timed_seeds=len(wall_times),
                           seed_scores=(scores if complete else [None]*len(spec['seeds'])) if spec['metric']=='best_mean_accuracy' else
                                       [trial['loss' if spec['metric']=='loss' else 'accuracy'] for trial in trials[-len(spec['seeds']):]],
                           selected=bool(selection and (selection.get('selected_candidates', {}).get(variant) == label
                                         if adaptive else selection.get('selected_variant') == variant
                                         and selection['selected'][variant] == rate))))
    options = benchmark.runs[0].options
    chosen_variant = selection.get('selected_variant') if selection else None
    chosen = templates.get(chosen_variant)
    if chosen:
        chosen = replace(chosen, options=chosen.options | selection.get('selected_parameters', {}).get(chosen_variant, {}))
    chosen_rate = selection['selected'][chosen_variant] if chosen else None
    if adaptive:
        distribution = spec['search_spaces'].get(chosen_variant, {}).get('learning_rate', {})
        bounds = distribution.get('choices', [distribution.get('low'), distribution.get('high')])
        boundary = bool(chosen and chosen_rate in bounds and (distribution.get('type') != 'categorical' or chosen_rate in (min(bounds), max(bounds))))
        local_space = spec['search_spaces'][next(iter(spec['search_spaces']))].get('local_steps')
        local_description = json.dumps(local_space) if local_space else ', '.join(str(run.options['local_steps']) for run in templates.values())
    else:
        boundary = bool(chosen and chosen_rate in (min(spec['learning_rates'][chosen_variant]), max(spec['learning_rates'][chosen_variant])))
        local_description = ', '.join(str(run.options['local_steps']) for run in templates.values())
    parameters = [
        ('Dataset / model', f"{options['dataset']} / {options['model']}"),
        ('Clients n / degree k', f"{options['peers']} / {options['k']} = log₂(n)") if options['algorithm']!='shatter' else
        ('Real nodes n / virtual graph degree r', f"{options['peers']} / {options['r']} = log₂(n)"),
        ('Displayed data partition', 'IID' if options['partitioner']=='iid' else 'Dirichlet α = 0.1'),
        ('Displayed hardware profile', benchmark.network['profile']),
        ('Tuning setting', entry['tuning_setting']),
        ('Training / validation / test images', '45,000 / 5,000 / 10,000'),
        ('Batch size per client', options['batch_size']),
        ('Local SGD steps H', chosen.options['local_steps'] if chosen else 'Joint search: '+local_description),
        ('Search method', 'TPE; full-budget trials without early stopping' if adaptive else 'Declared grid'),
        ('Learning rate', chosen_rate if chosen else 'Joint search; not selected'),
        ('Learning-rate schedule', 'Constant'),
        ('Momentum / weight decay', f"{options['momentum']} / {options['weight_decay']}"),
        ('Mean simulated SGD step', f"{benchmark.network['compute_step_ms']} ms"),
        ('Mean client bandwidth', f"{benchmark.network['bandwidth_bytes_per_second']:,} bytes/s"),
        ('Network latency', f"{options['network_latency_ms']} ms per transfer"),
        ('Hardware heterogeneity', 'Uniform capacities' if benchmark.network['profile']=='uniform' else 'Independent factors {0.5, 1, 2, 4}, normalized to mean one for step duration and bandwidth'),
        ('Membership / preprocessing', 'Stable membership; normalized images, no augmentation'),
        ('Readout', 'Uniform parameter average of every client’s latest published model; offline diagnostic'),
        ('Tuning duration per trial', f"{spec['budget']} simulated seconds"),
        ('Primary tuning metric', 'Best mean validation accuracy at a shared checkpoint within the budget' if spec['metric']=='best_mean_accuracy' else
                                 'Validation negative log-likelihood (lower is better)' if spec['metric']=='loss' else 'Validation accuracy (higher is better)'),
        ('Tuning evaluation split / size', f"Validation / {spec['evaluation_examples'] or 'all 5,000'} images"),
        ('Tuning seeds', ', '.join(map(str,spec['seeds']))),
        ('Evaluation seeds', ', '.join(map(str,benchmark.seeds))),
        ('Evaluation budgets', ', '.join(map(str,benchmark.budgets))+' simulated seconds'),
        ('Evaluation interval', f"Every {spec['evaluation_period']} simulated seconds, including initialization and exact budgets"),
        ('Selection rule', selection['rule'] if selection else
                          f"Highest validation accuracy averaged across all seeds at a shared training checkpoint ({spec['evaluation_period']}–{spec['budget']} s); earliest time breaks within-pair ties; lower LR, then variant name, break between-pair ties" if spec['metric']=='best_mean_accuracy' else
                          f"{'Lowest mean endpoint validation NLL' if spec['metric']=='loss' else 'Highest mean endpoint validation accuracy'} at {spec['budget']} s across all tuning seeds; lower LR, then variant name, break exact ties"),
        ('Frozen evaluation checkpoint', f"{selection['selected_time']} simulated seconds" if selection and selection.get('selected_time') is not None else 'Not yet selected'),
    ]
    if entry.get('evaluation', {}).get('policy') == 'fixed_endpoint':
        parameters = [('Original validation-selected checkpoint' if key == 'Frozen evaluation checkpoint' else key, value)
                      for key, value in parameters]
        parameters.append(('Primary fixed evaluation endpoint', f"{entry['evaluation']['primary_time']} simulated seconds"))
    if options['algorithm'] == 'shatter':
        parameters[2:2] = [('Virtual nodes per real node', options['k']),
                           ('Total virtual nodes', options['peers']*options['k']),
                           ('Virtual topology', 'Undirected r-regular graph, redrawn each round with seed + round - 1')]
    elif options['algorithm'] == 'demo':
        parameters[1] = ('Clients / collective', f"{options['peers']} / directed ring all-gather, {options['peers'] - 1} stages")
        parameters = [('Gradient accumulation minibatches H' if key == 'Local SGD steps H' else
                       'SGD momentum / weight decay' if key == 'Momentum / weight decay' else
                       'Mean simulated minibatch compute' if key == 'Mean simulated SGD step' else key, value)
                      for key, value in parameters]
        space = spec.get('search_spaces', {}).get(benchmark.runs[0].variant, {})
        for label, name in [('Compression residual decay', 'demo_decay'),
                            ('DCT block size', 'demo_chunk'), ('Coefficients per block', 'demo_topk')]:
            value = chosen.options[name] if chosen else 'Joint search: ' + json.dumps(space.get(name, options[name]))
            parameters.append((label, value))
    public = {key:value for key,value in entry.items() if not key.startswith('_') and key not in {'config'}}
    progress = None
    if adaptive:
        from dasklearn.tpe_tuning import search_progress
        progress = search_progress(spec, search, report)
    return public | dict(setting=setting, status=pipeline['status'] if pipeline else 'planned',
                         phase=pipeline['phase'] if pipeline else 'tuning not started', error=pipeline.get('error') if pipeline else None,
                         experiment_parameters=[dict(parameter=key,value=value) for key,value in parameters],
                         tuning=dict(groups=groups,trials=trials,seeds=spec['seeds'],budget=spec['budget'],metric=spec['metric'],
                                     setting=entry['tuning_setting'], partitioner=entry['_benchmark'].runs[0].options['partitioner'],
                                     alpha=entry['_benchmark'].runs[0].options['alpha'], hardware_profile=entry['_benchmark'].network['profile'],
                                     method=spec.get('method', 'grid'), search_spaces=spec.get('search_spaces'),
                                     progress=progress,
                                     n_startup_trials=spec.get('n_startup_trials'), batch_size=spec.get('batch_size'),
                                     total_trials=spec['n_trials']*len(spec['seeds']) if adaptive else len(trials),
                                     completed_trials=sum(row['status'] in {'completed', 'diverged'} for row in trials),
                                     selected_variant=chosen_variant,selected_learning_rate=chosen_rate,
                                     selected_local_steps=chosen.options['local_steps'] if chosen else None,
                                     selected_parameters=(selection.get('selected_parameters', {}).get(chosen_variant)
                                                          if selection else None),
                                     selected_time=selection.get('selected_time') if selection else None,
                                     boundary_winner=boundary),
                         configuration=json.loads(benchmark.path.read_text()))
