#!/usr/bin/env python3
"""Diagnose D-PSGD MIA signal using saved incoming models and cohort controls.

Replays into a fresh directory. Original experiment artifacts remain untouched.
Global versus class-matched nonmembers and optional no-mixing fits are diagnostic
controls, not tuned replacements for the original comparison.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from dasklearn import functions
from dasklearn.algorithms import class_paths, import_class
from dasklearn.model_trainer import ModelTrainer
from dasklearn.privacy.loss import evaluate_loss_attack, per_example_losses
from dasklearn.privacy.membership import _labels, make_class_matched_cohort
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.local import execute_local


def heldout_pools(labels, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 61001]))
    calibration, evaluation = [], []
    for label in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == label))
        calibration.extend(ids[:len(ids) // 2])
        evaluation.extend(ids[len(ids) // 2:])
    return np.asarray(calibration), np.asarray(evaluation)


def accuracy(model, partition):
    model = copy.deepcopy(model).cpu().eval()
    correct = count = 0
    with torch.inference_mode():
        for batch in DataLoader(partition, batch_size=256,
                                generator=torch.Generator().manual_seed(0)):
            correct += int((model(batch["img"]).argmax(1) == batch["label"]).sum())
            count += len(batch["label"])
    return correct / count


def audit(model, dataset, config, target, examples, model_kind):
    members = dataset.load_partition(target, "train")
    heldout = dataset.load_split("test")
    member_labels, heldout_labels = _labels(members), _labels(heldout)
    cohort = make_class_matched_cohort(member_labels, heldout_labels,
        seed=config.seed, target=target, max_examples=examples)
    calibration_pool, evaluation_pool = heldout_pools(heldout_labels, config.seed)
    count = len(cohort.members)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, target, 61003]))
    global_calibration = rng.permutation(calibration_pool)[:count]
    global_evaluation = rng.permutation(evaluation_pool)[:count]
    assert not set(global_calibration) & set(global_evaluation)
    training_losses = per_example_losses(model, members)
    heldout_losses = per_example_losses(model, heldout)
    # Full target label frequencies are evaluator-only oracle information. This
    # baseline diagnoses distribution shift; it is not an implementable attacker
    # with the assumptions of the main comparison.
    label_count = max(int(member_labels.max()), int(heldout_labels.max())) + 1
    prior = (np.bincount(member_labels, minlength=label_count) + 1) / (len(member_labels) + label_count)
    population = (np.bincount(heldout_labels, minlength=label_count) + 1) / (len(heldout_labels) + label_count)
    label_scores = np.log(prior / population)
    context = {"seed": config.seed, "partitioner": config.partitioner, "target": target,
               "model_kind": model_kind, "target_partition_size": len(members),
               "target_train_accuracy": accuracy(model, members),
               "global_test_accuracy": accuracy(model, heldout),
               "target_train_loss": float(training_losses.mean()),
               "global_test_loss": float(heldout_losses.mean())}
    rows = []
    for policy, calibration, evaluation in [
            ("class_matched", cohort.calibration, cohort.nonmembers),
            ("global_nonmembers", global_calibration, global_evaluation)]:
        member_loss = training_losses[cohort.members]
        nonmember_loss = heldout_losses[evaluation]
        calibration_loss = heldout_losses[calibration]
        metrics = evaluate_loss_attack(member_loss, nonmember_loss, calibration_loss, target_fpr=.1)
        median = evaluate_loss_attack(member_loss, nonmember_loss, calibration_loss, target_fpr=.5)
        truth = np.r_[np.ones(count), np.zeros(count)]
        fpr, tpr, _ = roc_curve(truth, -np.r_[member_loss, nonmember_loss])
        label_auc = roc_auc_score(truth, np.r_[label_scores[member_labels[cohort.members]],
                                               label_scores[heldout_labels[evaluation]]])
        rows.append({**context, "cohort_policy": policy, **metrics,
            "median_threshold_balanced_accuracy": median["balanced_accuracy"],
            "median_threshold_tpr": median["tpr"], "median_threshold_fpr": median["fpr"],
            "oracle_evaluation_best_balanced_accuracy": float(np.max((tpr + 1 - fpr) / 2)),
            "oracle_label_distribution_auc": float(label_auc)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/membership-incoming-digits"))
    parser.add_argument("--output", type=Path, default=Path("output/membership-diagnostics"))
    parser.add_argument("--standalone-control", action="store_true",
                        help="Also fit each target independently with the same SGD-step budget")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    original = json.loads((source / "manifest.json").read_text())
    if original["dataset"] != "digits":
        parser.error("This diagnostic currently uses the native offline digits workload")
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    for path in sorted([*(ROOT / "dasklearn").rglob("*.py"), Path(__file__).resolve()]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    (output / "manifest.json").write_text(json.dumps({
        "source": str(source), "source_experiment": original,
        "standalone_control": args.standalone_control, "source_sha256": digest.hexdigest(),
        "cohorts": "identical target members; class-matched versus global-distribution held-out nonmembers",
        "diagnostics": "10% and 50% nonmember-calibrated thresholds; oracle best evaluation threshold and oracle label-prior AUC are not deployable attack results",
        "standalone": "same initialization, target partition, per-round task seeds, SGD steps and optimizer; no model mixing"}, indent=2) + "\n")
    all_rows = []
    for run in sorted((source / "runs").glob("dpsgd-*")):
        settings_type, simulation_type = map(import_class, class_paths("dpsgd"))
        config = settings_type.from_dict(json.loads((run / "settings.json").read_text()))
        destination = output / "replays" / run.name
        destination.mkdir(parents=True)
        config.from_dir, config.data_dir = str(run), str(destination)
        sim = simulation_type(config)
        sim.workflow_dag = WorkflowDAG.load_from_file(str(run / "workflow_dag"))
        sim.workflow_dag.build_task_indices()
        results = execute_local(sim)
        dataset = functions.dataset
        ledger = json.loads((run / "incoming-observations.json").read_text())
        targets = {}
        for observation in ledger:
            if observation["round_nr"] != original["rounds"]:
                continue
            reference = observation["full_model"]
            task = sim.workflow_dag.tasks[reference[0]]
            assert task.func == "train" and task.data["peer"] == observation["sender"]
            assert task.data["round"] == observation["round_nr"]
            assert task.available_at <= observation["first_delivery_us"]
            targets[observation["sender"]] = results[reference[0]][reference[1]]
        for target, model in sorted(targets.items()):
            all_rows.extend(audit(model, dataset, config, target, original["examples"], "received_dpsgd"))
            if args.standalone_control:
                model = functions.initialize(config, {})[0]
                trainer = ModelTrainer(dataset, config, target)
                for round_nr in range(1, original["rounds"] + 1):
                    seed = functions.training_seed(config, {"peer": target, "round": round_nr})
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        trainer.train(model, config.learning.local_steps, seed=seed)
                all_rows.extend(audit(model, dataset, config, target, original["examples"], "standalone_no_mixing"))
        pd.DataFrame(all_rows).to_csv(output / "targets.csv", index=False)
        print(f"Diagnosed {run.name}: {len(targets)} final sender models", flush=True)
    frame = pd.DataFrame(all_rows)
    keys = ["partitioner", "model_kind", "cohort_policy", "seed"]
    metrics = ["roc_auc", "balanced_accuracy", "median_threshold_balanced_accuracy",
               "oracle_evaluation_best_balanced_accuracy", "oracle_label_distribution_auc",
               "target_train_accuracy", "global_test_accuracy", "target_train_loss", "global_test_loss"]
    per_seed = frame.groupby(keys, as_index=False)[metrics].mean()
    per_seed.to_csv(output / "per-seed.csv", index=False)
    summary = per_seed.groupby(keys[:-1])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    print(summary.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
