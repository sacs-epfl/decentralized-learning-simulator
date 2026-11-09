"""Tune one registered algorithm once, freeze its winning variant, evaluate four settings."""
import argparse
import fcntl
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def execute_campaign(registry, entry, *, reuse_tuning=None, resume_tuning=False):
    from dasklearn.benchmark import execute_benchmark
    from dasklearn.experiment_runner import write_json
    from dasklearn.leaderboard_details import scenario_benchmark
    from dasklearn.lr_tuning import execute_tuned_benchmark, frozen_runs
    benchmark = entry['_benchmark']
    if benchmark.output.exists() and not resume_tuning:
        raise ValueError(f'Output directory already exists: {benchmark.output}')
    code = execute_tuned_benchmark(benchmark, reuse_tuning=reuse_tuning, resume_tuning=resume_tuning)
    if code:
        return code
    selection = json.loads((benchmark.output / 'selection.json').read_text())
    chosen = selection['selected_variant']
    frozen = frozen_runs(benchmark, selection)
    campaign = dict(version=1, status='running', phase='transfer evaluation', selected_variant=chosen,
                    selected_learning_rate=selection['selected'][chosen], settings_completed=[entry['tuning_setting']])
    try:
        for setting in registry['scenarios']:
            if setting == entry['tuning_setting']:
                continue
            campaign['current_setting'] = setting
            write_json(benchmark.output / 'campaign.json', campaign)
            final = scenario_benchmark(entry, registry, setting)
            options = {run.name:run.options for run in final.runs}
            runs = [replace(run, options=options[run.name] | selection.get('selected_parameters', {}).get(run.variant, {})
                            | dict(learning_rate=run.options['learning_rate'])) for run in frozen]
            final = replace(final, name=f"{entry['label']} · {setting}", output=benchmark.output / f'evaluation-{setting}',
                            runs=runs, tuning=None, selection=selection)
            if selection.get('selected_time') is not None:
                final = replace(final,budgets=sorted(set(final.budgets+[selection['selected_time']])))
            code = execute_benchmark(final)
            if code:
                campaign.update(status='interrupted' if code == 130 else 'failed', error=f'{setting} evaluation did not complete')
                break
            campaign['settings_completed'].append(setting)
        else:
            campaign.update(status='completed',phase='evaluation completed')
    except KeyboardInterrupt:
        code = 130
        campaign.update(status='interrupted',error='Campaign interrupted')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        code = 1
        campaign.update(status='failed',error=str(exc))
    campaign['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(benchmark.output / 'campaign.json', campaign)
    return code


def graph_plan(benchmark):
    """Declare physical versus virtual graph size and validate the seeded topology."""
    import networkx as nx
    graphs = {}
    options = benchmark.runs[0].options
    n = options['peers']
    algorithm = options['algorithm']
    if algorithm == 'teleportation':
        return teleportation_graph_plan(benchmark)
    if algorithm == 'demo':
        if n < 2:
            raise ValueError('The DeMo campaign requires at least two ring participants')
        # DeMoClient sends to i+1 and receives from i-1, independently of seed.
        edges = [[node, (node + 1) % n] for node in range(n)]
        return {str(seed): dict(edges=edges, degree=1, connected=True, directed=True,
                               real_nodes=n, graph_nodes=n, virtual_nodes_per_real=0,
                               resampled_each_round=False, collective='ring all-gather',
                               stages_per_round=n - 1, gradient_accumulation=True)
                for seed in benchmark.tuning['seeds'] + benchmark.seeds}
    if algorithm in {'dpsgd', 'epidemic'}:
        degree, vertices = options['k'], n
        if options['topology'] != 'kreg':
            raise ValueError('This campaign requires the k-regular topology')
        if algorithm == 'epidemic' and options['el'] != 'oracle':
            raise ValueError('The regular-graph Epidemic campaign requires EL-Oracle')
    elif algorithm == 'shatter':
        degree, vertices = options['r'], n * options['k']
        if options['k'] < 1:
            raise ValueError('Shatter requires positive virtual nodes per real node')
    else:
        raise ValueError(f'No declared regular-graph campaign for {algorithm}')
    if degree < 1 or 2 ** degree != n or degree >= vertices or vertices * degree % 2:
        raise ValueError('The graph degree must be integer log2(real nodes) and define a valid regular graph')
    for seed in benchmark.tuning['seeds'] + benchmark.seeds:
        graph = nx.random_regular_graph(degree, vertices, seed=seed)
        if not nx.is_connected(graph):
            raise ValueError(f'The declared graph seed {seed} is disconnected; revise the protocol')
        record = dict(edges=sorted(sorted(edge) for edge in graph.edges), degree=degree, connected=True,
                      real_nodes=n, graph_nodes=vertices, virtual_nodes_per_real=options['k'] if algorithm=='shatter' else 0,
                      resampled_each_round=algorithm in {'shatter', 'epidemic'})
        if algorithm in {'shatter', 'epidemic'}:
            record.update(validation_scope='initial round', graph_seed_rule='seed + round - 1')
        if algorithm != 'shatter':
            eigenvalues = nx.linalg.spectrum.adjacency_spectrum(graph).real
            record['spectral_gap'] = float(1-sorted(abs(value) for value in (eigenvalues+1)/(degree+1))[-2])
        graphs[str(seed)] = record
    return graphs


def teleportation_graph_plan(benchmark):
    """Check every declared token graph using the simulator's actual routing rules."""
    from types import SimpleNamespace
    import networkx as nx
    from dasklearn.simulation.teleportation.simulation import TeleportationSimulation
    options = benchmark.runs[0].options
    space = benchmark.tuning.get('search_spaces', {}).get(benchmark.runs[0].variant, {})

    def choices(name):
        distribution = space.get(name)
        if distribution is None:
            return [options[name]]
        if distribution.get('type') != 'categorical':
            raise ValueError(f'Teleportation graph planning requires categorical {name}')
        return distribution['choices']

    graphs = {}
    for seed in benchmark.tuning['seeds'] + benchmark.seeds:
        token_graphs = []
        for size in choices('sample_size'):
            if type(size) is not int or not 1 <= size <= options['peers']:
                raise ValueError('Teleportation active clients must be in [1, peers]')
            for topology in choices('teleportation_topology'):
                if topology not in {'ring', 'hypercube', 'complete'}:
                    raise ValueError('Unknown Teleportation topology')
                if topology == 'hypercube' and size & (size - 1):
                    raise ValueError('Hypercube token count must be a power of two')
                sim = TeleportationSimulation.__new__(TeleportationSimulation)
                sim.settings = SimpleNamespace(participants=options['peers'], sample_size=size,
                                               topology=topology, seed=seed, shuffle_tokens=True)
                graph = nx.Graph()
                graph.add_nodes_from(range(size))
                graph.add_edges_from((token, peer) for token in range(size) for peer in sim.neighbors(token))
                degrees = {degree for _, degree in graph.degree()}
                if not nx.is_connected(graph) or len(degrees) != 1:
                    raise ValueError('Teleportation requires connected regular token graphs')
                token_graphs.append(dict(active_clients=size, topology=topology,
                    edges=sorted(sorted(edge) for edge in graph.edges), degree=next(iter(degrees)),
                    self_and_neighbor_weight=1 / (next(iter(degrees)) + 1), connected=True,
                    first_token_owners=sim.sample(1), next_token_owners=sim.sample(2)))
        graphs[str(seed)] = dict(real_nodes=options['peers'], token_graphs=token_graphs,
            token_topology_fixed=True, client_assignment_resampled_each_round=True,
            sampling='Seeded uniform sampling without replacement, then shuffled token assignment',
            handoff='Token i moves from sample(round)[i] to sample(round + 1)[i]; remote full-model transfers are charged',
            validation_scope='All declared token graphs; first two token assignments')
    return graphs


def reevaluate_campaign(registry, entry, *, replace_evaluations=False):
    """Run a new horizon with saved hyperparameters; retain the original tuning."""
    from dasklearn.benchmark import execute_benchmark
    from dasklearn.experiment_runner import write_json
    from dasklearn.leaderboard_details import scenario_benchmark
    from dasklearn.lr_tuning import frozen_runs
    benchmark = entry['_benchmark']
    output = benchmark.output
    selection = json.loads((output / 'selection.json').read_text())
    previous = json.loads((output / 'resolved.json').read_text())
    current = benchmark.manifest()
    for key in ('network', 'seeds', 'evaluation_model', 'tuning'):
        if previous[key] != current[key]:
            raise ValueError(f'Frozen evaluation must preserve {key}')
    old_runs = {run['name']: run for run in previous['runs']}
    if set(old_runs) != {run['name'] for run in current['runs']}:
        raise ValueError('Frozen evaluation must preserve run identities')
    for run in current['runs']:
        old = old_runs[run['name']]
        old_options = {key: value for key, value in old['options'].items() if key not in {'duration', 'device'}}
        new_options = {key: value for key, value in run['options'].items() if key not in {'duration', 'device'}}
        if any(old[key] != run[key] for key in ('seed', 'variant')) or old_options != new_options:
            raise ValueError('Frozen evaluation must preserve every numerical option except duration')
    if selection['specification'] != benchmark.tuning or not frozen_runs(benchmark, selection):
        raise ValueError('Saved selection must match the original tuning declaration')
    directories = [output / ('evaluation' if setting == entry['tuning_setting'] else 'evaluation-' + setting)
                   for setting in registry['scenarios']]
    if any(path.is_symlink() for path in directories):
        raise ValueError('Evaluation directories must not be symbolic links')
    if any(path.exists() for path in directories) and not replace_evaluations:
        raise ValueError('Existing evaluations require --replace-evaluations')
    pipeline = dict(version=1, status='running', phase='evaluation', tuning=benchmark.tuning,
                    selected=selection['selected'], final_seeds=benchmark.seeds,
                    evaluation_budget=max(benchmark.budgets), tuning_reused=True,
                    started_at=datetime.now(timezone.utc).isoformat())
    campaign = dict(version=1, status='running', phase='frozen-parameter evaluation',
                    selected_variant=selection['selected_variant'], selected_parameters=selection['selected_parameters'],
                    evaluation_budget=max(benchmark.budgets), settings_completed=[], tuning_reused=True)
    write_json(output / 'pipeline.json', pipeline)
    write_json(output / 'campaign.json', campaign)
    code = 0
    try:
        for path in directories:
            if path.exists():
                shutil.rmtree(path)
        for name in ('leaderboard.json', 'leaderboard.csv'):
            (output / name).unlink(missing_ok=True)
        for setting in [entry['tuning_setting'], *[s for s in registry['scenarios'] if s != entry['tuning_setting']]]:
            campaign['current_setting'] = setting
            write_json(output / 'campaign.json', campaign)
            scenario = scenario_benchmark(entry, registry, setting)
            final = replace(scenario, output=output / ('evaluation' if setting == entry['tuning_setting'] else 'evaluation-' + setting),
                            runs=frozen_runs(scenario, selection), tuning=None, selection=selection)
            code = execute_benchmark(final)
            if code:
                break
            campaign['settings_completed'].append(setting)
    except KeyboardInterrupt:
        code = 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        code = 1
        campaign['error'] = str(exc)
    status = 'completed' if code == 0 else 'interrupted' if code == 130 else 'failed'
    campaign.update(status=status, phase='evaluation completed' if not code else 'evaluation stopped',
                    finished_at=datetime.now(timezone.utc).isoformat())
    pipeline.update(status=status, finished_at=campaign['finished_at'])
    if code:
        pipeline['error'] = campaign.setdefault('error', 'Frozen-parameter evaluation did not complete')
    write_json(output / 'pipeline.json', pipeline)
    write_json(output / 'campaign.json', campaign)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1] / 'experiments/leaderboard-settings.json')
    parser.add_argument('--algorithm', default='dpsgd-kreg-log2')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--workers-per-gpu', type=int, default=3,
                        help='Maximum concurrent workers per GPU for new launches (default: 3)')
    parser.add_argument('--reuse-tuning', type=Path, help='Verify and reuse matching completed validation trials')
    parser.add_argument('--resume-tuning', action='store_true', help='Recover failed TPE tuning, retaining completed seeds and retrying incomplete seeds')
    parser.add_argument('--evaluation-only', action='store_true', help='Reuse saved tuning and evaluate the configured horizon')
    parser.add_argument('--replace-evaluations', action='store_true', help='Remove existing held-out evaluation artifacts before the evaluation-only run')
    args = parser.parse_args(argv)
    if args.workers_per_gpu < 1:
        parser.error('--workers-per-gpu must be positive')
    if args.replace_evaluations and not args.evaluation_only:
        parser.error('--replace-evaluations requires --evaluation-only')
    if args.evaluation_only and (args.reuse_tuning or args.resume_tuning):
        parser.error('--evaluation-only cannot change or recover tuning')
    from dasklearn.leaderboard_details import algorithm_entry, load_algorithms
    registry = load_algorithms(args.catalog)
    entry = algorithm_entry(registry, args.algorithm)
    if entry is None:
        parser.error('Unknown registered algorithm')
    benchmark = entry['_benchmark']
    if benchmark.execution_devices:
        counts, devices = {}, []
        for device in benchmark.execution_devices:
            if device.startswith('cuda') or device == 'mps':
                counts[device] = counts.get(device, 0) + 1
                if counts[device] > args.workers_per_gpu:
                    continue
            devices.append(device)
        benchmark = replace(benchmark, execution_devices=devices)
        entry['_benchmark'] = benchmark
    if not benchmark.tuning or not benchmark.tuning.get('select_variant'):
        parser.error('This campaign requires joint selection of one algorithm variant')
    try:
        graphs = graph_plan(benchmark)
    except ValueError as exc:
        parser.error(str(exc))
    plan = dict(algorithm=entry['id'], benchmark=benchmark.manifest(), output=str(benchmark.output),
                workers_per_gpu=args.workers_per_gpu,
                tuning_setting=entry['tuning_setting'], evaluation_settings=list(registry['scenarios']), graphs=graphs,
                reuse_tuning=str(args.reuse_tuning.resolve()) if args.reuse_tuning else None,
                resume_tuning=args.resume_tuning)
    plan.update(evaluation_only=args.evaluation_only, replace_evaluations=args.replace_evaluations)
    if args.plan:
        print(json.dumps(plan,indent=2,allow_nan=False))
        return 0
    from dasklearn.experiment_runner import write_json
    launch = benchmark.output.with_name(benchmark.output.name + '-launch')
    launch.mkdir(parents=True,exist_ok=True)
    with (launch / 'campaign.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another process already owns this campaign')
        write_json(launch / 'plan.json',plan)
        if args.evaluation_only:
            return reevaluate_campaign(registry, entry, replace_evaluations=args.replace_evaluations)
        return execute_campaign(registry,entry,reuse_tuning=args.reuse_tuning,resume_tuning=args.resume_tuning)


if __name__ == '__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    sys.exit(main())
