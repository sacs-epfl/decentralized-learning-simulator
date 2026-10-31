"""Versioned experiment files using the same options and types as the CLI.

Resolution does not create models, access datasets, or execute protocols.
"""
import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import re


CONTROL_OPTIONS = {"config", "local", "dry_run", "work_dir", "from_dir"}
EXECUTIONS = ("local", "simulate", "distributed")
PATH_OPTIONS = {"dataset_base_path", "dataset_cache_dir", "feedback_replay"}
SOURCE_OPTIONS = {"dataset_name", "text_dataset", "base_model", "tokenizer", "image_processor"}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key!r}")
        result[key] = value
    return result


def read_document(path):
    path = Path(path).expanduser().resolve()
    try:
        source = path.read_text()
        if path.suffix.lower() == ".json":
            document = json.loads(source, object_pairs_hook=_unique_pairs)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            class Loader(yaml.SafeLoader):
                pass
            def mapping(loader, node):
                loader.flatten_mapping(node)
                return _unique_pairs(loader.construct_pairs(node))
            Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
            document = yaml.load(source, Loader=Loader)
        else:
            raise ValueError("Configuration files must use .yaml, .yml, or .json")
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    except ImportError as exc:
        raise ValueError("YAML files require PyYAML; install the simulator requirements") from exc
    except Exception as exc:
        # SafeLoader rejects Python object constructors and reports line/column.
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("An experiment must be an object/mapping")
    if not all(isinstance(key, str) for key in document):
        raise ValueError("Experiment field names must be strings")
    unknown = set(document) - {"version", "name", "output", "execution", "settings", "matrix", "runs"}
    if unknown:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
    if type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("An experiment requires version: 1")
    if "matrix" in document and "runs" in document:
        raise ValueError("Use either matrix or runs, not both")
    if document.get("execution", "local") not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}")
    if "output" in document and (not isinstance(document["output"], str) or not document["output"].strip()):
        raise ValueError("output must be a nonempty directory path")
    name = document.get("name", re.sub(r"[^a-zA-Z0-9_.-]", "-", path.stem))
    _name(name)
    document["name"] = name
    return path, document


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value):
        raise ValueError("Run/experiment names must start with a letter/digit and contain only letters, digits, '.', '_' or '-'")
    return value


def explicit_options(parser, argv):
    return {parser._option_string_actions[token.split("=", 1)[0]].dest for token in argv
            if token.split("=", 1)[0] in parser._option_string_actions}


def _path(value, base):
    if "://" in value:
        return value
    return str((base / Path(value).expanduser()).resolve())


def validate_options(parser, options, base):
    if not isinstance(options, dict):
        raise ValueError("settings must be an object/mapping of CLI option names to values")
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    result = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise ValueError("Setting names must be strings")
        field = key.replace("-", "_")
        if field in result:
            raise ValueError(f"Duplicate setting after normalizing hyphens: {field}")
        if field in CONTROL_OPTIONS:
            raise ValueError(f"{field!r} is not a setting; use top-level execution/output (saved DAG replay uses --from-dir)")
        if field not in actions:
            raise ValueError(f"Unknown setting: {key!r}")
        action = actions[field]
        boolean = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction))
        if value is None and action.default is None:
            pass
        elif boolean:
            if type(value) is not bool:
                raise ValueError(f"{key} must be true or false")
        elif action.type is int:
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        elif action.type is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            value = float(value)
        elif field == "dataset_data_files":
            if not isinstance(value, dict) or not all(isinstance(k, str) and (
                    isinstance(v, str) or isinstance(v, list) and all(isinstance(item, str) for item in v))
                    for k, v in value.items()):
                raise ValueError("dataset_data_files must map split names to file paths or lists of paths")
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if value is not None and action.choices is not None and value not in action.choices:
            raise ValueError(f"Invalid {key}: {value!r}; expected one of {list(action.choices)}")
        if value is not None and field in PATH_OPTIONS:
            value = _path(value, base)
        elif isinstance(value, str) and field in SOURCE_OPTIONS and (
                value.startswith((".", "/", "~")) or (base / value).exists()):
            value = _path(value, base)
        elif field == "dataset_data_files" and value is not None:
            value = {split: [_path(item, base) for item in files] if isinstance(files, list)
                     else _path(files, base) for split, files in value.items()}
        result[field] = value
    return result


def execution_options(execution):
    return {"local": execution == "local", "dry_run": execution == "simulate"}


def load_single_options(path, parser):
    path, document = read_document(path)
    if "matrix" in document or "runs" in document:
        raise ValueError("Sweeps/named runs require: python scripts/run_experiment.py CONFIG")
    options = validate_options(parser, document.get("settings", {}), path.parent)
    options.update(execution_options(document.get("execution", "local")))
    if "output" in document:
        options["work_dir"] = _path(document["output"], path.parent)
    return options


@dataclass
class Run:
    name: str
    options: dict
    execution: str


@dataclass
class Experiment:
    path: Path
    name: str
    output: Path
    runs: list[Run]


def validate_run(options):
    for field in ("peers", "batch_size", "local_steps", "torch_threads", "brokers"):
        if options[field] < 1:
            raise ValueError(f"{field} must be positive")
    if options["stop"] == "rounds" and options["rounds"] < 1:
        raise ValueError("Set positive rounds, or choose stop: duration with a positive duration")
    if options["stop"] == "duration" and options["duration"] < 1:
        raise ValueError("duration must be positive")
    if options["partitioner"] == "dirichlet" and options["alpha"] <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    if options["seed"] < 0:
        raise ValueError("seed must be nonnegative")
    if options["evaluation_examples"] < 0:
        raise ValueError("evaluation_examples must be nonnegative")
    if options["model_size_bytes"] is not None and options["model_size_bytes"] <= 0:
        raise ValueError("model_size_bytes must be positive")
    for field in ("learning_rate", "momentum", "weight_decay"):
        if options[field] < 0:
            raise ValueError(f"{field} must be nonnegative")
    if not 0 <= options["validation_set_fraction"] < 1:
        raise ValueError("validation_set_fraction must be in [0, 1)")
    if options["lora"]:
        from types import SimpleNamespace
        from dasklearn.lora import LoRASpec
        LoRASpec.from_settings(SimpleNamespace(**options))


def resolve_experiment(path, parser, overrides=(), output=None, execution=None):
    path, document = read_document(path)
    base = validate_options(parser, document.get("settings", {}), path.parent)
    parsed = parser.parse_args(list(overrides))
    explicit = explicit_options(parser, overrides)
    forbidden = explicit & CONTROL_OPTIONS
    if forbidden:
        raise ValueError(f"Use --output/--execution for run controls, not {sorted(forbidden)}")
    cli = {key: getattr(parsed, key) for key in explicit}
    # CLI paths are relative to the invocation directory, as on the normal CLI.
    cli = validate_options(parser, cli, Path.cwd())
    mode = execution or document.get("execution", "local")
    if mode not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}")
    target = Path(output).expanduser().resolve() if output is not None else Path(
        _path(document.get("output", f"output/{document['name']}"), path.parent))
    scenarios = [("run", {})]
    if "runs" in document:
        if not isinstance(document["runs"], list) or not document["runs"]:
            raise ValueError("runs must be a nonempty list")
        scenarios = []
        for run in document["runs"]:
            if not isinstance(run, dict) or set(run) - {"name", "settings"} or "name" not in run:
                raise ValueError("Each named run needs name and optional settings")
            scenarios.append((_name(run["name"]), validate_options(parser, run.get("settings", {}), path.parent)))
        if len({name for name, _ in scenarios}) != len(scenarios):
            raise ValueError("Named runs must have unique names")
    if "matrix" in document:
        matrix = document["matrix"]
        if not isinstance(matrix, dict) or not matrix:
            raise ValueError("matrix must be a nonempty mapping of settings to nonempty lists")
        axes = {}
        for key, values in matrix.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"Matrix axis {key!r} must be a nonempty list")
            normalized = [validate_options(parser, {key: value}, path.parent) for value in values]
            field = next(iter(normalized[0]))
            if field in axes:
                raise ValueError(f"Duplicate matrix axis: {field}")
            items = [value[field] for value in normalized]
            if len({json.dumps(item, sort_keys=True) for item in items}) != len(items):
                raise ValueError(f"Duplicate values in matrix axis {key!r}")
            if field not in cli:
                axes[field] = items
        if math.prod(map(len, axes.values())) > 10000:
            raise ValueError("The matrix exceeds 10,000 runs; split the configuration")
        scenarios = [("matrix", dict(zip(axes, values))) for values in itertools.product(*axes.values())]
    defaults = vars(parser.parse_args([]))
    runs = []
    for index, (label, scenario) in enumerate(scenarios, 1):
        options = defaults | base | scenario | cli
        if options["dataset"] == "femnist":
            supplied = base.keys() | scenario.keys() | cli.keys()
            if "learning_rate" not in supplied:
                options["learning_rate"] = .004
            if "momentum" not in supplied:
                options["momentum"] = 0
        validate_run(options)
        suffix = f"{label}-{options['algorithm']}-{options['partitioner']}-s{options['seed']}"
        name = f"{index:04d}-{suffix}"
        settings = {key: value for key, value in options.items() if key not in CONTROL_OPTIONS}
        runs.append(Run(name, settings, mode))
    return Experiment(path, document["name"], target, runs)
