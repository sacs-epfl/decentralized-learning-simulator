"""Loss-threshold membership inference with independent nonmember calibration."""
import copy

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def _loss_array(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Losses must be a nonempty, finite one-dimensional array")
    return values


def calibrate_loss_threshold(nonmember_losses, target_fpr=0.1):
    """Fit `loss < threshold` with at most floor(n * target_fpr) calibration FPs.

    Strict comparison handles ties conservatively. This is an empirical
    calibration constraint, not a guarantee about the evaluation population.
    No evaluation labels or member losses enter threshold selection.
    """
    losses = np.sort(_loss_array(nonmember_losses))
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be strictly between zero and one")
    return float(losses[int(np.floor(len(losses) * target_fpr))])


def evaluate_loss_attack(member_losses, nonmember_losses, calibration_losses, *, target_fpr=0.1):
    """Score an independently calibrated attack; lower loss predicts membership.

    Balanced accuracy has a 0.5 chance baseline even with unequal sample counts.
    AUC uses negative loss and gives tied member/nonmember scores half credit.
    Inputs must represent disjoint calibration/evaluation examples; dataset
    ownership and disjointness are the caller's responsibility.
    """
    members = _loss_array(member_losses)
    nonmembers = _loss_array(nonmember_losses)
    calibration = _loss_array(calibration_losses)
    threshold = calibrate_loss_threshold(calibration, target_fpr)
    tpr = float(np.mean(members < threshold))
    fpr = float(np.mean(nonmembers < threshold))
    return {
        "balanced_accuracy": (tpr + 1 - fpr) / 2,
        "roc_auc": float(roc_auc_score(
            np.r_[np.ones(len(members)), np.zeros(len(nonmembers))],
            -np.r_[members, nonmembers])),
        "tpr": tpr, "fpr": fpr, "advantage": tpr - fpr,
        "threshold": threshold, "target_fpr": target_fpr,
        "calibration_fpr": float(np.mean(calibration < threshold)),
        "members": len(members), "nonmembers": len(nonmembers),
        "calibration_nonmembers": len(calibration),
        "fpr_resolution": 1 / len(nonmembers),
        "mean_member_loss": float(members.mean()),
        "mean_nonmember_loss": float(nonmembers.mean()),
    }


def per_example_losses(model, dataset, *, feature_key="img", label_key="label",
                       loss="cross_entropy", device="cpu", batch_size=256):
    """Evaluate a classification model on a mapping-style Torch dataset.

    The caller's model, buffers, mode, gradients and global RNG are preserved.
    Supports logits (cross_entropy) or log-probabilities (nll).
    """
    if not len(dataset) or batch_size < 1:
        raise ValueError("A nonempty dataset and positive batch_size are required")
    losses = {"cross_entropy": F.cross_entropy, "nll": F.nll_loss}
    if loss not in losses:
        raise ValueError("loss must be cross_entropy or nll")
    audit_model = copy.deepcopy(model).to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    values = []
    with torch.inference_mode():
        for batch in loader:
            output = audit_model(batch[feature_key].to(device))
            values.append(losses[loss](output, batch[label_key].to(device),
                                       reduction="none").cpu().numpy())
    return _loss_array(np.concatenate(values))
