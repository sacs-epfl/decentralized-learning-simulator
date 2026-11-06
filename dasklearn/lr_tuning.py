"""Equal-budget validation-only LR search followed by frozen held-out evaluation."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import statistics

from dasklearn.benchmark import BenchmarkRun, _numbers, budget_metrics, execute_benchmark
from dasklearn.experiment_runner import write_json


def validate_tuning(spec, benchmark, names):
    allowed = {"learning_rates", "seeds", "budget", "evaluation_period", "evaluation_examples", "metric", "select_variant",
               "method", "search_spaces", "n_trials", "n_startup_trials", "sampler_seed", "batch_size"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError(f"tuning requires a mapping with fields {sorted(allowed)}")
    method = spec.get("method", "grid")
    if not isinstance(method, str) or method not in {"grid", "tpe"}:
        raise ValueError("tuning.method must be grid or tpe")
    if method == "grid":
        if set(spec) & {"search_spaces", "n_trials", "n_startup_trials", "sampler_seed", "batch_size"}:
            raise ValueError("Adaptive search options require tuning.method: tpe")
        rates = spec.get("learning_rates")
        if not isinstance(rates, dict) or set(rates) != names:
            raise ValueError("tuning.learning_rates must cover exactly the named algorithm variants")
        rates = {name: _numbers(values, f"learning_rates.{name}") for name, values in rates.items()}
        if len({len(values) for values in rates.values()}) != 1:
            raise ValueError("Equal tuning budget requires the same number of LR candidates for every algorithm")
    seeds = _numbers(spec.get("seeds"), "tuning.seeds", integer=True, zero=True)
    if set(seeds) & set(benchmark.seeds):
        raise ValueError("Tuning seeds and final evaluation seeds must be disjoint")
    budget = spec.get("budget")
    _numbers([budget], "tuning.budget", integer=True)
    period = spec.get("evaluation_period", budget)
    _numbers([period], "tuning.evaluation_period", integer=True)
    if budget // period + 2 > 10000:
        raise ValueError("More than 10,000 tuning checkpoints")
    examples = spec.get("evaluation_examples", benchmark.runs[0].options["evaluation_examples"])
    if type(examples) is not int or examples < 0:
        raise ValueError("tuning.evaluation_examples must be a nonnegative integer")
    metric = spec.get("metric", "accuracy")
    if metric not in {"accuracy", "average_accuracy", "loss", "best_mean_accuracy"}:
        raise ValueError("tuning.metric must be accuracy, average_accuracy, loss, or best_mean_accuracy")
    if metric == "average_accuracy" and period >= budget:
        raise ValueError("average_accuracy tuning requires intermediate evaluation checkpoints")
    if any(run.options["evaluation_split"] != "test" for run in benchmark.runs):
        raise ValueError("A tuned benchmark must use evaluation_split: test for final evaluation; tuning always uses validation")
    select_variant = spec.get("select_variant", False)
    if type(select_variant) is not bool:
        raise ValueError("tuning.select_variant must be a boolean")
    if select_variant and len({run.options["algorithm"] for run in benchmark.runs}) != 1:
        raise ValueError("Joint variant selection must compare variants of one algorithm")
    selected_names = list(dict.fromkeys(run.variant for run in benchmark.runs))
    if method == "tpe":
        from dasklearn.tpe_tuning import validate_search
        result = validate_search(spec, benchmark, names, selected_names)
        result.update(seeds=seeds, budget=budget, evaluation_period=period, evaluation_examples=examples, metric=metric)
        result.update(trials_per_algorithm=len(seeds) * result["n_trials"],
                      simulated_seconds_per_algorithm=budget * len(seeds) * result["n_trials"])
        if select_variant:
            result["select_variant"] = True
        return result
    result = dict(learning_rates={name: rates[name] for name in selected_names}, seeds=seeds,
                budget=budget, evaluation_period=period, evaluation_examples=examples, metric=metric,
                trials_per_algorithm=len(seeds) * len(next(iter(rates.values()))),
                simulated_seconds_per_algorithm=budget * len(seeds) * len(next(iter(rates.values()))))
    if select_variant:
        result["select_variant"] = True
    return result


def best_mean_checkpoint(records, spec):
    """One shared validation checkpoint; never average independent seed maxima."""
    times = sorted({0, *range(spec['evaluation_period'], spec['budget'] + 1, spec['evaluation_period']), spec['budget']})
    curves = []
    for record in records:
        curve = record['curve']
        if ([point['time'] for point in curve] != times
                or any(not math.isfinite(point['accuracy']) or not math.isfinite(point['loss']) for point in curve)):
            raise ValueError('Best-mean selection requires every declared finite checkpoint on every seed')
        curves.append({point['time']: point for point in curve})
    # Initialization is a reference, not a trained candidate checkpoint.
    return min(times[1:], key=lambda time: (-statistics.mean(curve[time]['accuracy'] for curve in curves), time))


def score_candidate(records, spec, evaluation_model):
    """Score a full paired candidate using the same objective for grids and TPE."""
    observations, finite = [], []
    if [record['seed'] for record in records] != spec['seeds']:
        raise ValueError('Every candidate requires exactly the declared paired seeds')
    for record in records:
        if record['status'] not in {'completed', 'diverged'}:
            raise ValueError('Every candidate must finish every tuning seed before scoring')
        options = record['options']
        if (options['seed'] != record['seed'] or options['duration'] != spec['budget']
                or options['evaluation_split'] != 'validation'
                or options['evaluation_examples'] != spec['evaluation_examples']):
            raise ValueError('Tuning trial settings differ from the declared validation-only budget')
        if record['status'] == 'diverged':
            if (evaluation_model != 'global' or record.get('completed_budget', record.get('divergence_time')) != spec['budget']
                    or not isinstance(record.get('divergence_time'), (int, float))
                    or not 0 <= record['divergence_time'] <= spec['budget']
                    or record.get('tuning_budget_completed') is not True):
                raise ValueError('Diverged trials must reach the exact global validation budget')
            observations.append(dict(seed=record['seed'], score=None, status='diverged'))
            continue
        metrics = budget_metrics(record['curve'], spec['budget'], [])
        if metrics is None:
            raise ValueError('Missing exact tuning-budget checkpoint')
        value = metrics['accuracy' if spec['metric'] == 'best_mean_accuracy' else spec['metric']]
        if value is None or not math.isfinite(value):
            raise ValueError('Completed tuning scores must be finite')
        observations.append(dict(seed=record['seed'], score=value))
        finite.append(record)
    selected_time = None
    if spec['metric'] == 'best_mean_accuracy' and len(finite) == len(records):
        selected_time = best_mean_checkpoint(finite, spec)
        observations = [dict(seed=record['seed'], score=next(point['accuracy'] for point in record['curve']
                             if point['time'] == selected_time)) for record in finite]
    result = dict(mean_score=statistics.mean(row['score'] for row in observations)
                  if all(row['score'] is not None for row in observations) else None, observations=observations)
    if spec['metric'] == 'best_mean_accuracy':
        result['selected_time'] = selected_time
    return result


def tuning_benchmark(benchmark):
    spec = benchmark.tuning
    if spec.get("method") == "tpe":
        raise ValueError("TPE candidates are adaptive; use execute_tuned_benchmark rather than a static tuning grid")
    times = sorted({0, *range(spec["evaluation_period"], spec["budget"] + 1, spec["evaluation_period"]), spec["budget"]})
    templates = {run.variant: run for run in benchmark.runs}
    runs = []
    for variant, rates in spec["learning_rates"].items():
        for index, rate in enumerate(rates, 1):
            label = f"{variant}-lr{index:02d}"
            for seed in spec["seeds"]:
                options = templates[variant].options | dict(learning_rate=rate, seed=seed,
                    duration=spec["budget"], evaluation_split="validation", evaluation_examples=spec["evaluation_examples"])
                runs.append(BenchmarkRun(f"{label}-s{seed}", label, seed, options))
    return replace(benchmark, name=benchmark.name + " · LR tuning", output=benchmark.output / "tuning",
                   budgets=[spec["budget"]], seeds=spec["seeds"], times=times, runs=runs, tuning=None)


def select_learning_rates(benchmark, tuning_result):
    """Reject incomplete trial grids; no test metrics or early peaks enter selection."""
    spec = benchmark.tuning
    if tuning_result.get("status") != "completed":
        raise ValueError("All tuning trials must complete before LR selection")
    if tuning_result.get("evaluation_model", "local") != benchmark.evaluation_model:
        raise ValueError("Tuning and final evaluation must use the same model readout")
    records = tuning_result["runs"]
    expected_count = len(spec["learning_rates"]) * spec["trials_per_algorithm"]
    if len(records) != expected_count:
        raise ValueError("Incomplete tuning trial grid")
    by_name = {record["name"]: record for record in records}
    if len(by_name) != len(records):
        raise ValueError("Duplicate tuning trial")
    selected, scores = {}, {}
    direction = 1 if spec['metric'] == 'loss' else -1
    for variant, rates in spec["learning_rates"].items():
        candidates = []
        for index, rate in enumerate(rates, 1):
            candidate_records = []
            for seed in spec["seeds"]:
                record = by_name.get(f"{variant}-lr{index:02d}-s{seed}")
                if record is None or record["status"] not in {"completed", "diverged"}:
                    raise ValueError("Every algorithm must complete every candidate on every tuning seed")
                options = record["options"]
                if (options["learning_rate"] != rate or options["seed"] != seed
                        or options["duration"] != spec["budget"] or options["evaluation_split"] != "validation"
                        or options["evaluation_examples"] != spec["evaluation_examples"]):
                    raise ValueError("Tuning trial settings differ from the declared validation-only budget")
                candidate_records.append(record)
            candidate = dict(learning_rate=rate, **score_candidate(candidate_records, spec, benchmark.evaluation_model))
            candidates.append(candidate)
        eligible = [row for row in candidates if row["mean_score"] is not None]
        if not eligible:
            if not spec.get('select_variant'):
                raise ValueError(f"No finite LR candidate for {variant}")
        else:
            winner = min(eligible, key=lambda row: (direction * row["mean_score"], row["learning_rate"]))
            selected[variant] = winner["learning_rate"]
        scores[variant] = candidates
    result = dict(version=1, selected=selected, scores=scores, specification=spec,
                selection_split="validation", evaluation_split="test", evaluation_seeds=benchmark.seeds,
                rule=f"{'Minimize' if direction == 1 else 'Maximize'} mean {spec['metric']} at {spec['budget']} simulated seconds across all tuning seeds; lower LR breaks exact ties; candidates with nonfinite endpoint validation metrics are ineligible")
    if spec.get("select_variant"):
        candidates = [(variant, row) for variant, rows in scores.items() for row in rows
                      if row["mean_score"] is not None]
        if not candidates:
            raise ValueError('No finite joint parameter candidate')
        variant, winner = min(candidates, key=lambda item: (direction * item[1]["mean_score"], item[1]["learning_rate"], item[0]))
        result.update(selected_variant=variant, selected_learning_rate=winner["learning_rate"])
        if spec['metric']=='best_mean_accuracy':
            result['selected_time'] = winner['selected_time']
        result["rule"] += "; jointly select one variant and LR; variant name breaks remaining exact ties"
    if spec['metric']=='best_mean_accuracy':
        result['rule'] = ('Maximize validation accuracy averaged across all tuning seeds at one shared scheduled training checkpoint '
                          f'within {spec["budget"]} simulated seconds; earliest checkpoint breaks within-pair ties; '
                          'lower LR, then variant name, break between-pair ties; divergent candidates are ineligible')
    return result


def execute_tuned_benchmark(benchmark, *, reuse_tuning=None, resume_tuning=False):
    """Expose the active phase at the root, so the same dashboard URL stays live."""
    output = benchmark.output
    adaptive = benchmark.tuning.get("method") == "tpe"
    if adaptive and reuse_tuning is not None:
        raise ValueError("TPE cannot reuse a grid search; choose a fresh adaptive search output")
    if resume_tuning and (not adaptive or reuse_tuning is not None):
        raise ValueError('Recovery requires TPE and cannot be combined with grid reuse')
    if adaptive:
        from dasklearn.tpe_tuning import require_optuna
        require_optuna()
    tuning = None if adaptive else tuning_benchmark(benchmark)
    reused = {}
    if reuse_tuning is not None:
        from dasklearn.leaderboard_extension import reusable_trials
        reused = reusable_trials(tuning, reuse_tuning)
    if resume_tuning:
        previous = json.loads((output / 'pipeline.json').read_text())
        if previous['status'] not in {'failed', 'interrupted'} or previous['phase'] != 'tuning':
            raise ValueError('Recovery requires a failed or interrupted tuning phase')
        if (output / 'selection.json').exists() or (output / 'evaluation').exists():
            raise ValueError('Tuning recovery must precede selection and held-out evaluation')
        if json.loads((output / 'resolved.json').read_text()) != benchmark.manifest():
            raise ValueError('Recovery requires the original benchmark declaration')
        from pathlib import Path
        import shutil
        snapshot = output / ('recovery-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        snapshot.mkdir()
        for directory in (output, output / 'tuning'):
            target = snapshot if directory == output else snapshot / 'tuning'
            target.mkdir(exist_ok=True)
            for path in directory.iterdir():
                if path.is_file() and path.suffix in {'.json', '.csv'}:
                    shutil.copy2(path, target / path.name)
    else:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError(f"Output directory already exists: {output}; choose a fresh --output") from exc
        (output / ("source" + benchmark.path.suffix)).write_text(benchmark.path.read_text())
    pipeline = dict(version=1, name=benchmark.name, status="running", phase="tuning",
                    tuning=benchmark.tuning, selected=None, final_seeds=benchmark.seeds,
                    started_at=datetime.now(timezone.utc).isoformat())
    if resume_tuning:
        pipeline.update(started_at=previous['started_at'], resumed_at=datetime.now(timezone.utc).isoformat(),
                        recovery_snapshot=str(snapshot))
    if reuse_tuning is not None:
        pipeline.update(reused_tuning_trials=len(reused), reuse_tuning=str(reuse_tuning))
    write_json(output / "resolved.json", benchmark.manifest())
    dataset_path = benchmark.runs[0].options["dataset_base_path"]
    if dataset_path:
        from pathlib import Path
        source = Path(dataset_path) / "source.json"
        if source.exists():
            (output / "dataset-source.json").write_bytes(source.read_bytes())
    write_json(output / "pipeline.json", pipeline)

    def mirror(result):
        result = dict(result, name=benchmark.name, pipeline=dict(pipeline))
        write_json(output / "leaderboard.json", result)
        source = output / pipeline["phase"] / "leaderboard.csv"
        temporary = output / "leaderboard.csv.tmp"
        temporary.write_bytes(source.read_bytes())
        temporary.replace(output / "leaderboard.csv")

    try:
        if adaptive:
            from dasklearn.tpe_tuning import execute_search
            code = execute_search(benchmark, on_update=mirror, resume=resume_tuning)
        elif reused:
            from dasklearn.leaderboard_extension import execute_remaining_tuning
            code = execute_remaining_tuning(tuning, reused, on_update=mirror)
        else:
            code = execute_benchmark(tuning, on_update=mirror, allow_divergent_trials=True)
        if code:
            pipeline.update(status="interrupted" if code == 130 else "failed",
                            error="Tuning did not complete; final evaluation was not started")
        else:
            if adaptive:
                from dasklearn.tpe_tuning import select_search
                selection = select_search(benchmark, json.loads((output / "tuning" / "leaderboard.json").read_text()),
                                          json.loads((output / "tuning" / "search.json").read_text())["candidates"])
            else:
                selection = select_learning_rates(benchmark, json.loads((tuning.output / "leaderboard.json").read_text()))
            write_json(output / "selection.json", selection)
            pipeline.update(phase="evaluation", selected=selection["selected"])
            write_json(output / "pipeline.json", pipeline)
            runs = frozen_runs(benchmark, selection)
            final = replace(benchmark, output=output / "evaluation", runs=runs, tuning=None, selection=selection)
            if selection.get('selected_time') is not None:
                final = replace(final, budgets=sorted(set(final.budgets + [selection['selected_time']])))
            if selection.get('selected_times'):
                final = replace(final, budgets=sorted(set(final.budgets + list(selection['selected_times'].values()))))
            code = execute_benchmark(final, on_update=mirror)
            pipeline["status"] = "completed" if code == 0 else ("interrupted" if code == 130 else "failed")
    except KeyboardInterrupt:
        code = 130
        pipeline.update(status="interrupted", error="Pipeline interrupted")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        code = 1
        pipeline.update(status="failed", error=str(exc))
    pipeline["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "pipeline.json", pipeline)
    latest = output / "leaderboard.json"
    if latest.exists():
        result = json.loads(latest.read_text())
        result.update(status=pipeline["status"], pipeline=pipeline)
        write_json(latest, result)
    if pipeline.get("error"):
        print(pipeline["error"], flush=True)
    return code


def frozen_runs(benchmark, selection):
    """Carry every selected parameter, as well as LR, to all held-out scenarios."""
    return [replace(run, options=run.options | selection.get("selected_parameters", {}).get(run.variant, {})
                    | dict(learning_rate=selection["selected"][run.variant]))
            for run in benchmark.runs
            if not selection.get("selected_variant") or run.variant == selection["selected_variant"]]
