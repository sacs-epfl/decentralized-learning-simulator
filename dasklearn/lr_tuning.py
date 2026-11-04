"""Equal-budget validation-only LR search followed by frozen held-out evaluation."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import statistics

from dasklearn.benchmark import BenchmarkRun, _numbers, budget_metrics, execute_benchmark
from dasklearn.experiment_runner import write_json


def validate_tuning(spec, benchmark, names):
    allowed = {"learning_rates", "seeds", "budget", "evaluation_period", "evaluation_examples", "metric"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError(f"tuning requires a mapping with fields {sorted(allowed)}")
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
    if metric not in {"accuracy", "average_accuracy"}:
        raise ValueError("tuning.metric must be accuracy or average_accuracy")
    if metric == "average_accuracy" and period >= budget:
        raise ValueError("average_accuracy tuning requires intermediate evaluation checkpoints")
    if any(run.options["evaluation_split"] != "test" for run in benchmark.runs):
        raise ValueError("A tuned benchmark must use evaluation_split: test for final evaluation; tuning always uses validation")
    selected_names = list(dict.fromkeys(run.variant for run in benchmark.runs))
    return dict(learning_rates={name: rates[name] for name in selected_names}, seeds=seeds,
                budget=budget, evaluation_period=period, evaluation_examples=examples, metric=metric,
                trials_per_algorithm=len(seeds) * len(next(iter(rates.values()))),
                simulated_seconds_per_algorithm=budget * len(seeds) * len(next(iter(rates.values()))))


def tuning_benchmark(benchmark):
    spec = benchmark.tuning
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
    for variant, rates in spec["learning_rates"].items():
        candidates = []
        for index, rate in enumerate(rates, 1):
            observations = []
            for seed in spec["seeds"]:
                record = by_name.get(f"{variant}-lr{index:02d}-s{seed}")
                if record is None or record["status"] not in {"completed", "diverged"}:
                    raise ValueError("Every algorithm must complete every candidate on every tuning seed")
                options = record["options"]
                if (options["learning_rate"] != rate or options["seed"] != seed
                        or options["duration"] != spec["budget"] or options["evaluation_split"] != "validation"
                        or options["evaluation_examples"] != spec["evaluation_examples"]):
                    raise ValueError("Tuning trial settings differ from the declared validation-only budget")
                if record["status"] == "diverged":
                    if (benchmark.evaluation_model != "global" or record.get("divergence_time") != spec["budget"]
                            or record.get("tuning_budget_completed") is not True):
                        raise ValueError("Diverged trials must reach the exact global validation budget")
                    observations.append(dict(seed=seed, score=None, status="diverged"))
                    continue
                metrics = budget_metrics(record["curve"], spec["budget"], [])
                if metrics is None:
                    raise ValueError("Missing exact tuning-budget checkpoint")
                observations.append(dict(seed=seed, score=metrics[spec["metric"]]))
            candidates.append(dict(learning_rate=rate, mean_score=statistics.mean(row["score"] for row in observations)
                                   if all(row["score"] is not None for row in observations) else None,
                                   observations=observations))
        eligible = [row for row in candidates if row["mean_score"] is not None]
        if not eligible:
            raise ValueError(f"No finite LR candidate for {variant}")
        winner = min(eligible, key=lambda row: (-row["mean_score"], row["learning_rate"]))
        selected[variant] = winner["learning_rate"]
        scores[variant] = candidates
    return dict(version=1, selected=selected, scores=scores, specification=spec,
                selection_split="validation", evaluation_split="test", evaluation_seeds=benchmark.seeds,
                rule=f"Maximize mean {spec['metric']} at {spec['budget']} simulated seconds across all tuning seeds; lower LR breaks exact ties; candidates with nonfinite endpoint validation metrics are ineligible")


def execute_tuned_benchmark(benchmark):
    """Expose the active phase at the root, so the same dashboard URL stays live."""
    output = benchmark.output
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"Output directory already exists: {output}; choose a fresh --output") from exc
    (output / ("source" + benchmark.path.suffix)).write_text(benchmark.path.read_text())
    pipeline = dict(version=1, name=benchmark.name, status="running", phase="tuning",
                    tuning=benchmark.tuning, selected=None, final_seeds=benchmark.seeds,
                    started_at=datetime.now(timezone.utc).isoformat())
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
        tuning = tuning_benchmark(benchmark)
        code = execute_benchmark(tuning, on_update=mirror, allow_divergent_trials=True)
        if code:
            pipeline.update(status="interrupted" if code == 130 else "failed",
                            error="Tuning did not complete; final evaluation was not started")
        else:
            selection = select_learning_rates(benchmark, json.loads((tuning.output / "leaderboard.json").read_text()))
            write_json(output / "selection.json", selection)
            pipeline.update(phase="evaluation", selected=selection["selected"])
            write_json(output / "pipeline.json", pipeline)
            runs = [replace(run, options=run.options | dict(learning_rate=selection["selected"][run.variant])) for run in benchmark.runs]
            final = replace(benchmark, output=output / "evaluation", runs=runs, tuning=None, selection=selection)
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
