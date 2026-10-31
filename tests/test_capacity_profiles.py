from types import SimpleNamespace

import pytest

from dasklearn import AUGMENTATION_FACTOR_SIM
from dasklearn.simulation.capacities import apply_capacities


def capacities(profile, seed=3):
    clients = [SimpleNamespace(index=i, bw_scheduler=SimpleNamespace(bw_limit=0)) for i in range(16)]
    settings = SimpleNamespace(capacity_profile=profile, seed=seed, traces="none",
                               bandwidth_bytes_per_s=1000000, compute_step_ms=20)
    apply_capacities(clients, settings)
    return [c.bw_scheduler.bw_limit for c in clients], [c.simulated_speed * AUGMENTATION_FACTOR_SIM for c in clients]


def test_controlled_profiles_preserve_means_and_independent_factors():
    uniform, network, compute, both = [capacities(p) for p in ("uniform", "network", "compute", "heterogeneous")]
    assert network[0] == both[0] and compute[1] == both[1]
    assert compute[0] == uniform[0] and network[1] == uniform[1]
    for bw, step in (uniform, network, compute, both):
        assert sum(bw) / 16 == pytest.approx(1000000, abs=1)
        assert sum(step) / 16 == pytest.approx(20)
    assert len(set(both[0])) > 1 and len(set(both[1])) > 1
    assert capacities("heterogeneous") == both
    assert capacities("heterogeneous", seed=4) != both
