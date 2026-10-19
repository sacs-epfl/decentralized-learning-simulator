import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from dasklearn.datasets.synthetic import TensorPartition
from dasklearn.privacy.loss import calibrate_loss_threshold, evaluate_loss_attack, per_example_losses
from dasklearn.privacy.membership import (add_local_model_mia_task, make_class_matched_cohort,
                                        make_global_nonmember_cohort)
from dasklearn.tasks.dag import WorkflowDAG
from dasklearn.tasks.local import execute_local
from dasklearn.tasks.references import TaskRef
from helpers import settings, simulate


def test_loss_attack_separates_members_and_does_not_fit_on_evaluation_data():
    first = evaluate_loss_attack([0.01, 0.02], [2, 3], [1, 1.5, 2, 3], target_fpr=.25)
    assert first["balanced_accuracy"] == first["roc_auc"] == first["tpr"] == 1
    assert first["fpr"] == 0
    assert first["threshold"] == 1.5
    changed = evaluate_loss_attack([20, 30], [0, .1], [1, 1.5, 2, 3], target_fpr=.25)
    assert changed["threshold"] == first["threshold"]
    assert changed["balanced_accuracy"] == changed["roc_auc"] == 0


def test_tied_losses_are_chance_and_threshold_respects_calibration_budget():
    result = evaluate_loss_attack([1, 1], [1] * 8, [1] * 10)
    assert result["balanced_accuracy"] == result["roc_auc"] == .5
    assert result["tpr"] == result["fpr"] == result["calibration_fpr"] == 0
    for fpr in [.001, .1, .25, .99]:
        calibration = np.array([1, 1, 1, 2, 3, 3, 4])
        threshold = calibrate_loss_threshold(calibration, fpr)
        assert (calibration < threshold).sum() <= np.floor(len(calibration) * fpr)


def test_memorizing_model_is_a_positive_control_for_end_to_end_loss_attack():
    # Disjoint one-hot coordinates let a linear classifier memorize the training
    # examples while leaving held-out examples at uniform confidence.
    features = torch.eye(48)
    labels = torch.arange(48) % 2
    model = nn.Linear(48, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=2)
    for _ in range(50):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(features[:16]), labels[:16]).backward()
        optimizer.step()
    members, calibration, nonmembers = [
        per_example_losses(model, TensorPartition(features[start:start + 16], labels[start:start + 16]))
        for start in [0, 16, 32]]
    metrics = evaluate_loss_attack(members, nonmembers, calibration)
    assert metrics["balanced_accuracy"] == metrics["roc_auc"] == 1
    assert metrics["tpr"] == 1 and metrics["fpr"] == 0


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[1]]])
def test_invalid_losses_fail_explicitly(bad):
    with pytest.raises(ValueError):
        evaluate_loss_attack(bad, [1], [1])


@pytest.mark.parametrize("fpr", [0, 1, -1, np.nan])
def test_invalid_false_positive_budget(fpr):
    with pytest.raises(ValueError):
        calibrate_loss_threshold([1, 2], fpr)


@pytest.mark.parametrize("log_probabilities", [False, True])
def test_per_example_losses_match_reference_and_preserve_model_and_rng(log_probabilities):
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.5), nn.Linear(4, 2))
    if log_probabilities:
        model.append(nn.LogSoftmax(dim=1))
    model.train()
    features, labels = torch.randn(9, 3), torch.arange(9) % 2
    partition = TensorPartition(features, labels)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    actual = per_example_losses(model, partition, loss="nll" if log_probabilities else "cross_entropy", batch_size=4)
    assert model.training and all(module.training for module in model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    model.eval()
    with torch.no_grad():
        objective = nn.NLLLoss(reduction="none") if log_probabilities else nn.CrossEntropyLoss(reduction="none")
        expected = objective(model(features), labels).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_cohorts_are_balanced_class_matched_and_globally_disjoint():
    heldout_labels = np.repeat([0, 1, 2], 21)
    member_labels = np.repeat([0, 1, 2], [30, 10, 2])
    cohorts = [make_class_matched_cohort(member_labels, heldout_labels, seed=3, target=target, max_examples=40)
               for target in range(4)]
    calibration, evaluation = set(), set()
    for cohort in cohorts:
        assert len(cohort.members) == len(cohort.calibration) == len(cohort.nonmembers) <= 40
        assert len(set(cohort.members)) == len(cohort.members)
        assert sorted(member_labels[cohort.members]) == sorted(heldout_labels[cohort.nonmembers])
        assert sorted(member_labels[cohort.members]) == sorted(heldout_labels[cohort.calibration])
        calibration.update(cohort.calibration)
        evaluation.update(cohort.nonmembers)
    assert not calibration & evaluation
    assert cohorts[0].digest == make_class_matched_cohort(
        member_labels, heldout_labels, seed=3, target=0, max_examples=40).digest
    assert cohorts[0].digest != cohorts[1].digest


def test_global_cohort_preserves_member_control_but_retains_nonmember_label_shift():
    heldout = np.repeat([0, 1, 2], 100)
    members = np.zeros(80, dtype=int)
    calibration, evaluation = set(), set()
    for target in range(3):
        matched = make_class_matched_cohort(members, heldout, seed=4, target=target, max_examples=60)
        global_cohort = make_global_nonmember_cohort(members, heldout, seed=4, target=target, max_examples=60)
        np.testing.assert_array_equal(global_cohort.members, matched.members)
        assert len(global_cohort.calibration) == len(global_cohort.nonmembers) == len(matched.members)
        assert len(np.unique(heldout[global_cohort.nonmembers])) == 3
        assert global_cohort.digest != matched.digest
        calibration.update(global_cohort.calibration)
        evaluation.update(global_cohort.nonmembers)
        calibration.update(matched.calibration)
        evaluation.update(matched.nonmembers)
    assert not calibration & evaluation


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["dpsgd", "shatter", "moshpit"])
@pytest.mark.parametrize("cohort_policy", ["class_matched", "global_nonmembers"])
async def test_audit_is_reusable_streaming_replayable_and_does_not_change_training(tmp_path, algorithm, cohort_policy):
    from dasklearn.algorithms import class_paths, import_class
    config_type, cls = map(import_class, class_paths(algorithm))
    options = {"dpsgd": {"topology": "ring"}, "shatter": {"k": 4, "r": 2}, "moshpit": {"group_size": 3}}
    config = settings(config_type, tmp_path, algorithm=algorithm, **options[algorithm])
    sim = simulate(cls, config)
    baseline = execute_local(sim)
    original_names = set(sim.workflow_dag.tasks)
    model_round, reference = sim.latest_models[0]
    audit = add_local_model_mia_task(sim, 0, reference, round_nr=model_round,
        time_us=sim.current_time, targets=[0, 1, 2], max_examples=20, cohort_policy=cohort_policy)
    assert list(audit.data["targets"]) == [1, 2]
    assert [TaskRef(task.name, output) for task, output in audit.inputs] == [reference]
    sim.workflow_dag.build_task_indices()
    result = execute_local(sim)
    for name in original_names:
        for actual, expected in zip(result[name], baseline[name]):
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected)
            else:
                for key, value in expected.state_dict().items():
                    assert torch.equal(actual.state_dict()[key], value)
    expected_rows = result[audit.name][0]
    assert len(expected_rows) == 2
    path = tmp_path / "audit-dag"
    sim.workflow_dag.save_to_file(str(path))
    sim.workflow_dag = WorkflowDAG.load_from_file(str(path))
    sim.workflow_dag.build_task_indices()
    sim.workflow_dag.check_validity()
    assert execute_local(sim, retain_results=False) == {}
    artifact = json.loads((Path(sim.data_dir) / f"mia-{audit.name}.json").read_text())
    assert artifact["rows"] == expected_rows
    assert all(row["members"] == row["nonmembers"] == row["calibration_nonmembers"] for row in expected_rows)
    assert all(row["cohort_policy"] == cohort_policy for row in expected_rows)


def test_summary_uses_independent_seed_means():
    from scripts.compare_membership import summarize
    rows = []
    for seed, values in [(0, [0, 1]), (1, [1])]:
        for value in values:
            rows.append({"variant": "dpsgd", "partitioner": "iid", "seed": seed, "round": 1,
                **{key: value for key in ["balanced_accuracy", "roc_auc", "tpr", "fpr", "model_accuracy", "model_loss"]}})
    per_seed, summary = summarize(pd.DataFrame(rows))
    assert len(per_seed) == summary.seeds.iloc[0] == 2
    assert summary.balanced_accuracy_mean.iloc[0] == .75


@pytest.mark.asyncio
async def test_unavailable_model_is_rejected_without_adding_audit(tmp_path):
    from dasklearn.simulation.dpsgd.settings import DPSGDSettings
    from dasklearn.simulation.dpsgd.simulation import DPSGDSimulation
    sim = simulate(DPSGDSimulation, settings(DPSGDSettings, tmp_path, algorithm="dpsgd", topology="ring"))
    model_round, model = sim.latest_models[0]
    count = len(sim.workflow_dag.tasks)
    with pytest.raises(ValueError, match="not available"):
        add_local_model_mia_task(sim, 0, model, round_nr=model_round, time_us=0)
    assert len(sim.workflow_dag.tasks) == count
