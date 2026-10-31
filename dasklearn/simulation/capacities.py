"""Reproducible controlled capacities, independent of protocol RNG and data."""
import math
from random import Random

from dasklearn import AUGMENTATION_FACTOR_SIM


def apply_capacities(clients, settings):
    profile = settings.capacity_profile
    if profile == "default":
        return
    if settings.traces != "none":
        raise ValueError("Controlled capacities cannot be combined with traces")
    if profile not in {"uniform", "network", "compute", "heterogeneous"}:
        raise ValueError(f"Unknown capacity profile: {profile}")
    if settings.bandwidth_bytes_per_s < 1 or not math.isfinite(settings.compute_step_ms) or settings.compute_step_ms <= 0:
        raise ValueError("Bandwidth and compute step time must be positive")
    # Independent streams keep node rankings fixed when toggling one factor.
    order = list(range(len(clients)))
    Random(settings.seed + 100).shuffle(order)
    fast = set(order[:max(1, len(clients) // 4)])
    rng = Random(settings.seed + 200)
    times = [rng.choice([.5, 1, 2, 4]) for _ in clients]
    mean_time = sum(times) / len(times)
    mean_bw = (8 * len(fast) + len(clients) - len(fast)) / len(clients)
    for client in clients:
        bw_factor = (8 if client.index in fast else 1) / mean_bw if profile in {"network", "heterogeneous"} else 1
        time_factor = times[client.index] / mean_time if profile in {"compute", "heterogeneous"} else 1
        client.bw_scheduler.bw_limit = round(settings.bandwidth_bytes_per_s * bw_factor)
        client.simulated_speed = settings.compute_step_ms * time_factor / AUGMENTATION_FACTOR_SIM
