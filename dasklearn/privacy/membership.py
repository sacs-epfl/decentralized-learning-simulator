"""Protocol-independent audit cohorts and a serializable offline DAG operation."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from torch.utils.data import ConcatDataset, Subset

from dasklearn.privacy.loss import evaluate_loss_attack, per_example_losses
from dasklearn.tasks.task import Task
from dasklearn.tasks.operations import operation_contract
from dasklearn.tasks.references import as_ref
from dasklearn.util import MICROSECONDS


@dataclass(frozen=True)
class AuditCohort:
    members: np.ndarray
    calibration: np.ndarray
    nonmembers: np.ndarray
    class_counts: dict

    @property
    def digest(self):
        digest = hashlib.sha256()
        for indices in (self.members, self.calibration, self.nonmembers):
            digest.update(np.asarray([len(indices)], dtype="<i8").tobytes())
            digest.update(np.asarray(indices, dtype="<i8").tobytes())
        return digest.hexdigest()


def make_class_matched_cohort(member_labels, heldout_labels, *, seed, target=0, max_examples=128):
    """Equal-size member/nonmember/calibration sets with identical label counts.

    Split each held-out class in half *before* target-specific sampling. The
    calibration/evaluation pools stay disjoint across all targets and variants.
    Members are sampled without replacement, with unsupported classes trimmed.
    max_examples caps each set, not the combined number of queries.
    """
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    members = np.asarray(member_labels)
    heldout = np.asarray(heldout_labels)
    if members.ndim != 1 or heldout.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    split_rng = np.random.default_rng(np.random.SeedSequence([seed, 61001]))
    sample_rng = np.random.default_rng(np.random.SeedSequence([seed, target, 61002]))
    candidates = sample_rng.permutation(len(members))[:max_examples]
    selected, calibration, nonmembers, counts = [], [], [], {}
    # Include all held-out classes so pool assignment never depends on target.
    for label in np.unique(heldout):
        pool = split_rng.permutation(np.flatnonzero(heldout == label))
        midpoint = len(pool) // 2
        left, right = pool[:midpoint], pool[midpoint:]
        local = candidates[members[candidates] == label]
        count = min(len(local), len(left), len(right))
        if count:
            selected.extend(local[:count])
            calibration.extend(sample_rng.permutation(left)[:count])
            nonmembers.extend(sample_rng.permutation(right)[:count])
            counts[str(int(label))] = count
    if not selected:
        raise ValueError("No class has both members and disjoint held-out samples")
    return AuditCohort(*(np.asarray(ids, dtype=np.int64) for ids in
                         (selected, calibration, nonmembers)), class_counts=counts)


def _labels(partition):
    # Native tensor datasets expose labels; avoid reading every image to get IDs.
    if hasattr(partition, "labels"):
        labels = np.asarray(partition.labels)
        return labels[partition.indices] if hasattr(partition, "indices") else labels
    raise ValueError("The built-in MIA adapter requires native tensor partitions")


def make_global_nonmember_cohort(member_labels, heldout_labels, *, seed, target=0, max_examples=128):
    """Keep the matched-control members; sample nonmembers from global classes.

    The member IDs and sample counts match make_class_matched_cohort exactly,
    isolating the effect of changing the nonmember distribution. Calibration and
    evaluation still use globally disjoint per-class held-out pools.
    """
    matched = make_class_matched_cohort(member_labels, heldout_labels,
        seed=seed, target=target, max_examples=max_examples)
    heldout = np.asarray(heldout_labels)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 61001]))
    calibration, evaluation = [], []
    for label in np.unique(heldout):
        ids = rng.permutation(np.flatnonzero(heldout == label))
        calibration.extend(ids[:len(ids) // 2])
        evaluation.extend(ids[len(ids) // 2:])
    sample_rng = np.random.default_rng(np.random.SeedSequence([seed, target, 61003]))
    count = len(matched.members)
    return AuditCohort(matched.members,
        sample_rng.permutation(np.asarray(calibration))[:count],
        sample_rng.permutation(np.asarray(evaluation))[:count], matched.class_counts)


def _class_counts(labels):
    classes, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(classes, counts)}


def validate_mia_dataset(settings):
    if settings.dataset in {"synthetic", "digits"}:
        return
    if settings.dataset == "cifar10" and settings.dataset_base_path and settings.model in {"lenet", "gnlenet"}:
        return
    if settings.dataset == "tiny_imagenet" and settings.dataset_base_path:
        return
    raise ValueError("MIA supports synthetic, digits, native CIFAR-10 and native Tiny ImageNet. "
                     "For other datasets use per_example_losses with deterministic evaluation transforms.")


def add_local_model_mia_task(simulation, attacker, model, *, round_nr, time_us,
                             targets=None, max_examples=128, target_fpr=0.1,
                             cohort_policy="class_matched"):
    """Attach an offline sink audit to an attacker-owned complete model reference.

    Works with any protocol's local model, with no message or training changes.
    The caller must supply a model actually available to the attacker at time_us.
    Targets' training partitions are evaluator ground truth, never attack inputs.
    Rebuild task indices and re-save the DAG after attaching audit tasks.
    """
    return add_model_mia_task(simulation, attacker, model, round_nr=round_nr,
        time_us=time_us, targets=targets, max_examples=max_examples, target_fpr=target_fpr,
        cohort_policy=cohort_policy)


def add_model_mia_task(simulation, attacker, model, *, round_nr, time_us,
                       targets=None, max_examples=128, target_fpr=0.1,
                       attacker_view="local_model", observation=None, cohort_policy="class_matched"):
    """Attach a loss audit to an explicit attacker-visible complete model view.

    Incoming partial payloads must first be completed using attacker-known state.
    The observer in incoming.py enforces that construction for delivered chunks.
    """
    validate_mia_dataset(simulation.settings)
    if attacker_view not in {"local_model", "incoming_full", "incoming_partial"}:
        raise ValueError("Unknown membership attacker view")
    if cohort_policy not in {"class_matched", "global_nonmembers"}:
        raise ValueError("Unknown MIA cohort policy")
    peers = simulation.settings.participants
    targets = list(range(peers)) if targets is None else list(targets)
    targets = sorted(set(targets) - {attacker})
    if not 0 <= attacker < peers or not targets or any(not 0 <= peer < peers for peer in targets):
        raise ValueError("MIA requires a valid attacker and at least one other target peer")
    if max_examples < 1 or not 0 < target_fpr < 1:
        raise ValueError("MIA requires positive max_examples and 0 < target_fpr < 1")
    name = Task.generate_name("membership_inference")
    model = as_ref(model)
    source = simulation.workflow_dag.tasks[model.task]
    available = source.available_at if source.available_at is not None else source.data.get("time", 0)
    if available > time_us:
        raise ValueError("The audited model was not available at the observation time")
    data = {
        "model": model, "peer": attacker, "round": round_nr, "time": time_us,
        "targets": targets, "max_examples": max_examples, "target_fpr": target_fpr,
        "audit_id": name,
        "attacker_view": attacker_view, "observation": observation or {},
        "cohort_policy": cohort_policy,
    }
    return simulation.workflow_dag.add(Task(name, "membership_inference", data,
        operation=operation_contract("membership_inference"), available_at=time_us))


def evaluate_membership(settings, params):
    """DAG operation: one attacker-visible model, several target-client audits.

    Only the model's losses and auxiliary nonmember losses determine predictions.
    Target ownership is used to construct and score the balanced audit cohorts.
    Files are task-specific, so broker workers do not append to a shared CSV.
    """
    from dasklearn import functions
    from dasklearn.datasets import create_dataset

    started = time.perf_counter()
    validate_mia_dataset(settings)
    if functions.dataset is None:
        functions.dataset = create_dataset(settings)
    dataset = functions.dataset
    split = "valid" if settings.dataset == "tiny_imagenet" else "test"
    heldout = dataset.load_split(split)
    heldout_labels = _labels(heldout)
    loss = "nll" if settings.dataset == "cifar10" else "cross_entropy"
    feature = "image" if settings.dataset == "tiny_imagenet" else "img"
    rows = []
    policy = params.get("cohort_policy", "class_matched")
    cohort_functions = {"class_matched": make_class_matched_cohort,
                        "global_nonmembers": make_global_nonmember_cohort}
    for target in params["targets"]:
        partition = dataset.load_partition(target, "train")
        cohort = cohort_functions[policy](_labels(partition), heldout_labels,
            seed=settings.seed, target=target, max_examples=params["max_examples"])
        samples = ConcatDataset([Subset(partition, cohort.members.tolist()),
                                 Subset(heldout, cohort.calibration.tolist()),
                                 Subset(heldout, cohort.nonmembers.tolist())])
        losses = per_example_losses(params["model"], samples, feature_key=feature,
                                   loss=loss, device=settings.torch_device_name)
        count = len(cohort.members)
        metrics = evaluate_loss_attack(losses[:count], losses[2 * count:], losses[count:2 * count],
                                       target_fpr=params["target_fpr"])
        rows.append({**params.get("observation", {}),
                     "attacker": params["peer"], "target": target, "round": params["round"],
                     "time": params["time"] / MICROSECONDS,
                     "attacker_view": params.get("attacker_view", "local_model"),
                     "target_partition_size": len(partition), "class_counts": cohort.class_counts,
                     "calibration_class_counts": _class_counts(heldout_labels[cohort.calibration]),
                     "nonmember_class_counts": _class_counts(heldout_labels[cohort.nonmembers]),
                     "cohort_policy": policy,
                     "cohort_sha256": cohort.digest, **metrics})
    audit_id = params["audit_id"]
    if Path(audit_id).name != audit_id:
        raise ValueError("audit_id must be a filename component")
    artifact = {"schema_version": 1, "attack": "loss_threshold",
                "threshold_method": "independent_nonmember_quantile",
                "execution_s": time.perf_counter() - started, "rows": rows}
    path = Path(settings.data_dir) / f"mia-{audit_id}.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    return [rows]
