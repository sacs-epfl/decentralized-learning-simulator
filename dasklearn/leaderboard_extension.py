"""Tune new algorithms once and extend existing paired scenario comparisons."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from dasklearn.benchmark import NonfiniteEvaluation, execute_benchmark, leaderboard, read_curve, save_results
from dasklearn.experiment_runner import provenance, write_json
from dasklearn.leaderboard_server import load_settings
from dasklearn.lr_tuning import select_learning_rates, tuning_benchmark


def load_baselines(benchmark, catalog, *, allow_pending=False):
    """Reject incompatible or incomplete baselines before starting any training."""
    if benchmark.tuning is None:
        raise ValueError("An extension requires an equal-budget tuning specification")
    _, entries = load_settings(catalog=catalog)
    additions = {run.variant for run in benchmark.runs}
    shared = ("dataset", "model", "peers", "dataset_base_path", "validation_set_fraction",
              "evaluation_split", "evaluation_examples", "local_steps", "batch_size",
              "momentum", "weight_decay", "network_latency_ms")
    template = benchmark.runs[0].options
    settings = []
    tuning_scenario_found = False
    for entry in entries:
        suite = json.loads((entry["output"] / "manifest.json").read_text())
        if allow_pending and (suite["status"] not in {"running", "completed"}
                or any(run["status"] not in {"planned", "running", "completed"} for run in suite["runs"])):
            raise ValueError(f"Baseline {entry['id']} has failed or interrupted runs")
        if not allow_pending and (suite["status"] != "completed" or any(run["status"] != "completed" for run in suite["runs"])):
            raise ValueError(f"Baseline {entry['id']} must be fully completed")
        if any(suite[field] != getattr(benchmark, field) for field in ("budgets", "seeds", "times")):
            raise ValueError(f"Baseline {entry['id']} has different budgets, seeds, or checkpoints")
        if suite["accuracy_targets"] != benchmark.targets or suite.get("evaluation_model", "local") != benchmark.evaluation_model:
            raise ValueError(f"Baseline {entry['id']} has different evaluation targets or model readout")
        if suite["network"] | {"profile": "uniform"} != benchmark.network | {"profile": "uniform"}:
            raise ValueError(f"Baseline {entry['id']} has different base network capacities")
        if additions & {run["variant"] for run in suite["runs"]}:
            raise ValueError(f"Baseline {entry['id']} already contains an added variant")
        if any(any(run["options"][key] != template[key] for key in shared) for run in suite["runs"]):
            raise ValueError(f"Baseline {entry['id']} has incompatible data/model/training settings")
        scenario = {key: suite["runs"][0]["options"][key] for key in ("partitioner", "alpha")}
        if any(any(run["options"][key] != value for key, value in scenario.items()) for run in suite["runs"]):
            raise ValueError(f"Baseline {entry['id']} mixes data scenarios")
        if (not allow_pending or suite["status"] == "completed") and any(table["provisional"] for table in leaderboard(suite)["tables"]):
            raise ValueError(f"Baseline {entry['id']} lacks complete paired checkpoints")
        if suite["network"] == benchmark.network and all(template[key] == value for key, value in scenario.items()):
            tuning_scenario_found = True
        settings.append(dict(entry, suite=suite, scenario=scenario))
    if not tuning_scenario_found:
        raise ValueError("The declared tuning data/hardware scenario must match one baseline")
    return settings


def final_benchmark(benchmark, setting, selected):
    """Apply frozen validation-selected LRs; only partition and capacities vary."""
    runs = [replace(run, options=run.options | setting["scenario"] |
                    dict(learning_rate=selected[run.variant])) for run in benchmark.runs]
    return replace(benchmark, name=benchmark.name + "-" + setting["id"],
                   output=benchmark.output / "evaluation" / setting["id"],
                   network=dict(setting["suite"]["network"]), runs=runs, tuning=None)


def reusable_trials(trials, previous):
    """Reuse only matching completed validation trials, verified against raw CSVs."""
    previous = Path(previous).resolve()
    suite = json.loads((previous / "manifest.json").read_text())
    expected = trials.manifest()
    for key in ("budgets", "seeds", "times", "network", "evaluation_model"):
        if suite.get(key) != expected[key]:
            raise ValueError(f"Previous tuning has different {key}")
    templates = {run.name: run for run in trials.runs}
    reused = {}
    for record in suite["runs"]:
        if record["name"] not in templates or record["status"] not in {"completed", "failed", "diverged"}:
            continue
        template = templates[record["name"]]
        options = {key: value for key, value in record["options"].items() if key != "device"}
        requested = {key: value for key, value in template.options.items() if key != "device"}
        if (record["name"] in reused or options != requested or record["variant"] != template.variant
                or record["seed"] != template.seed):
            raise ValueError(f"Previous tuning trial differs from requested settings: {record['name']}")
        directory = Path(record.get("trial_directory", previous / "runs" / record["name"])).resolve()
        resolved = json.loads((directory / "resolved.json").read_text())
        if (resolved["options"] != record["options"] or resolved["times"] != trials.times
                or resolved["network"] != trials.network or resolved["evaluation_model"] != trials.evaluation_model):
            raise ValueError(f"Previous raw trial settings differ: {record['name']}")
        try:
            curve = read_curve(directory, template.options["peers"], trials.times,
                               strict=True, evaluation_model=trials.evaluation_model)
        except NonfiniteEvaluation as exc:
            if (record["status"] not in {"failed", "diverged"} or trials.evaluation_model != "global"
                    or template.options["evaluation_split"] != "validation"
                    or exc.stamp != max(trials.times) or len(exc.curve) != len(trials.times) - 1):
                raise ValueError(f"Previous trial did not complete its validation budget: {record['name']}") from exc
            record = deepcopy(record) | dict(status="diverged", curve=exc.curve,
                divergence_time=exc.stamp, tuning_budget_completed=True,
                error="Nonfinite validation metrics at the exact tuning budget; candidate is ineligible")
            curve = exc.curve
        if record["status"] == "failed":
            continue
        if curve != record["curve"]:
            raise ValueError(f"Previous trial curve differs from raw observations: {record['name']}")
        reused[record["name"]] = deepcopy(record) | dict(reused_from=str(directory), trial_directory=str(directory),
                                                       reused_provenance=suite.get("provenance"))
    return reused


def execute_remaining_tuning(trials, reused, *, on_update):
    """Publish the full declared grid, executing only trials without cached results."""
    trials.output.mkdir(parents=True, exist_ok=False)
    suite = trials.manifest() | dict(status="running", provenance=provenance(Path(__file__).resolve().parents[1]))
    suite["runs"] = [deepcopy(reused[run.name]) if run.name in reused else
                     dict(asdict(run), status="planned", curve=[]) for run in trials.runs]
    (trials.output / ("source" + trials.path.suffix)).write_bytes(trials.path.read_bytes())

    def publish(result=None):
        if result is not None:
            updates = {run["name"]: dict(run, trial_directory=str(trials.output.parent / "tuning-new" / "runs" / run["name"]))
                       for run in result["runs"]}
            suite["runs"] = [deepcopy(updates.get(run["name"], run)) for run in suite["runs"]]
            suite["status"] = result["status"]
        on_update(save_results(trials.output, suite))

    publish()
    remaining = [run for run in trials.runs if run.name not in reused]
    if not remaining:
        suite["status"] = "completed"
        publish()
        return 0
    pending = replace(trials, output=trials.output.parent / "tuning-new", runs=remaining)
    return execute_benchmark(pending, on_update=publish, allow_divergent_trials=True)


def execute_extension(benchmark, settings, *, baseline_catalog, reuse_tuning=None, follow_baselines=False):
    """Publish all combined comparisons while running a detached-friendly pipeline."""
    output = benchmark.output
    trials = tuning_benchmark(benchmark)
    reused = reusable_trials(trials, reuse_tuning) if reuse_tuning is not None else {}
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"Output directory already exists: {output}; choose a fresh output") from exc
    (output / ("source" + benchmark.path.suffix)).write_bytes(benchmark.path.read_bytes())
    (output / "baseline-catalog.json").write_bytes(Path(baseline_catalog).read_bytes())
    write_json(output / "resolved.json", benchmark.manifest())
    root = Path(__file__).resolve().parents[1]
    extension_provenance = provenance(root)
    # Interleave algorithms so expensive asynchronous trials do not occupy all
    # slots at the start of the search. Every declared trial still runs in full.
    grouped = [[run for run in trials.runs if run.variant.rsplit("-lr", 1)[0] == variant]
               for variant in benchmark.tuning["learning_rates"]]
    trials = replace(trials, runs=[run for batch in zip(*grouped) for run in batch])
    state = dict(status="running", phase="tuning", current_setting=None,
                 added_variants=list(benchmark.tuning["learning_rates"]), following_baselines=follow_baselines,
                 tuning_completed=len(reused), tuning_total=len(trials.runs), selected=None,
                 reused_tuning_trials=len(reused),
                 diverged_tuning_trials=sum(run["status"] == "diverged" for run in reused.values()),
                 tuning=benchmark.tuning, started_at=datetime.now(timezone.utc).isoformat())
    selection = None
    records = {}
    for setting in settings:
        (output / "settings" / setting["id"]).mkdir(parents=True)
        records[setting["id"]] = [dict(asdict(run), options=run.options | {"learning_rate": None},
            status="planned", curve=[]) for run in benchmark.runs]

    def publish(setting):
        identifier = setting["id"]
        if follow_baselines:
            setting["suite"] = json.loads((setting["output"] / "manifest.json").read_text())
        suite = deepcopy(setting["suite"])
        baseline_complete = suite["status"] == "completed" and all(run["status"] == "completed" for run in suite["runs"])
        for record in suite["runs"]:
            record["source_output"] = str(setting["output"])
        suite["runs"] += deepcopy(records[identifier])
        added = records[identifier]
        suite.update(name=setting["suite"]["name"] + "-extended",
                     execution_devices=benchmark.execution_devices,
                     provenance=dict(baseline=setting["suite"].get("provenance"),
                                     additions=extension_provenance),
                     selection=dict(baseline=setting["suite"].get("selection"), additions=selection),
                     extension=dict(state),
                     status="completed" if baseline_complete and all(run["status"] == "completed" for run in added)
                     else state["status"] if state["status"] != "completed" else "failed")
        save_results(output / "settings" / identifier, suite)

    def publish_all():
        write_json(output / "pipeline.json", state)
        for setting in settings:
            publish(setting)

    def tuning_update(result):
        completed = sum(run["status"] in {"completed", "diverged"} for run in result["runs"])
        state["diverged_tuning_trials"] = sum(run["status"] == "diverged" for run in result["runs"])
        if completed != state["tuning_completed"]:
            state["tuning_completed"] = completed
            publish_all()

    def stop_unstarted(message):
        for added in records.values():
            for record in added:
                if record["status"] == "planned":
                    record.update(status="interrupted", error=message)

    publish_all()
    try:
        code = (execute_remaining_tuning(trials, reused, on_update=tuning_update) if reused
                else execute_benchmark(trials, on_update=tuning_update, allow_divergent_trials=True))
        if code == 130:
            raise KeyboardInterrupt
        if code:
            raise ValueError("Tuning did not complete; no final runs were started")
        selection = select_learning_rates(benchmark, json.loads((trials.output / "leaderboard.json").read_text()))
        write_json(output / "selection.json", selection)
        state.update(phase="evaluation", selected=selection["selected"])
        for setting in settings:
            final = final_benchmark(benchmark, setting, state["selected"])
            records[setting["id"]] = [dict(asdict(run), status="planned", curve=[]) for run in final.runs]
        publish_all()
        failed = False
        for setting in settings:
            identifier = setting["id"]
            state["current_setting"] = identifier
            publish_all()
            final = replace(final_benchmark(benchmark, setting, state["selected"]), selection=selection)

            def evaluation_update(result):
                records[identifier] = [dict(run, source_output=str(final.output)) for run in result["runs"]]
                publish(setting)

            code = execute_benchmark(final, on_update=evaluation_update)
            if code == 130:
                raise KeyboardInterrupt
            failed |= bool(code)
        if follow_baselines and not failed:
            state.update(phase="waiting_for_baselines", current_setting=None)
            while True:
                publish_all()
                if any(setting["suite"]["status"] not in {"running", "completed"} for setting in settings):
                    raise ValueError("An independently running baseline did not complete successfully")
                if all(setting["suite"]["status"] == "completed"
                       and all(run["status"] == "completed" for run in setting["suite"]["runs"]) for setting in settings):
                    break
                time.sleep(2)
        code = int(failed)
        state.update(status="failed" if failed else "completed", phase="finished", current_setting=None)
    except KeyboardInterrupt:
        code = 130
        state.update(status="interrupted", error="Extension interrupted")
        stop_unstarted(state["error"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        code = 1
        state.update(status="failed", error=str(exc))
        stop_unstarted(state["error"])
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    publish_all()
    if state.get("error"):
        print(state["error"], flush=True)
    return code
