"""Unit tests for genevac.pedestrian.registry and architectural isolation."""

import os
from pathlib import Path
import subprocess
import sys
import pytest

from genevac.environment.geometry import Point
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.pedestrian.registry import PedestrianRegistry


def test_registry_registration_and_duplicate_rejection():
    """Verify pedestrian registration and duplicate ID rejection."""
    registry = PedestrianRegistry()
    assert len(registry) == 0
    assert registry.total_count == 0

    p1 = PedestrianState(id="ped_1", position=Point(1.0, 2.0))
    registry.register(p1)

    assert len(registry) == 1
    assert registry.total_count == 1
    assert "ped_1" in registry

    # Duplicate registration
    p1_dup = PedestrianState(id="ped_1", position=Point(3.0, 4.0))
    with pytest.raises(ValueError, match="is already registered"):
        registry.register(p1_dup)

    # Invalid type registration
    with pytest.raises(TypeError, match="Expected PedestrianState"):
        registry.register("not_a_pedestrian")  # type: ignore[arg-type]


def test_registry_lookup_and_unknown_id():
    """Verify ID lookup via get, __getitem__, and contains."""
    registry = PedestrianRegistry()
    p1 = PedestrianState(id="ped_1", position=Point(5.0, 5.0))
    registry.register(p1)

    # Found lookups
    assert registry.get("ped_1") is p1
    assert registry["ped_1"] is p1
    assert "ped_1" in registry

    # Not found lookups
    assert registry.get("ped_unknown") is None
    assert "ped_unknown" not in registry
    with pytest.raises(KeyError, match="not found in registry"):
        _ = registry["ped_unknown"]


def test_registry_deterministic_iteration_order():
    """Verify registry preserves exact deterministic insertion order."""
    registry = PedestrianRegistry()
    ids = [f"ped_{i:03d}" for i in range(20)]
    for pid in ids:
        registry.register(PedestrianState(id=pid, position=Point(float(pid[-3:]), 0.0)))

    # Iteration order
    iter_ids = [p.id for p in registry]
    assert iter_ids == ids

    # all() order
    all_ids = [p.id for p in registry.all()]
    assert all_ids == ids


def test_registry_status_filtering():
    """Verify filtering pedestrians by status."""
    registry = PedestrianRegistry()

    p_act1 = PedestrianState(id="act1", position=Point(1, 1), status=PedestrianStatus.ACTIVE)
    p_act2 = PedestrianState(id="act2", position=Point(2, 2), status=PedestrianStatus.ACTIVE)
    p_evac = PedestrianState(id="evac1", position=Point(3, 3), status=PedestrianStatus.EVACUATED)
    p_blk = PedestrianState(id="blk1", position=Point(4, 4), status=PedestrianStatus.BLOCKED)
    p_inact = PedestrianState(id="inact1", position=Point(5, 5), status=PedestrianStatus.INACTIVE)

    for p in [p_act1, p_act2, p_evac, p_blk, p_inact]:
        registry.register(p)

    assert registry.get_active() == [p_act1, p_act2]
    assert registry.get_evacuated() == [p_evac]
    assert registry.get_blocked() == [p_blk]
    assert registry.get_inactive() == [p_inact]

    # String filtering
    assert registry.get_by_status("active") == [p_act1, p_act2]
    assert registry.get_by_status("evacuated") == [p_evac]

    # Invalid status filter
    with pytest.raises(ValueError, match="Invalid status filter"):
        registry.get_by_status("wandering")


def test_registry_dynamic_counts_and_evacuated_retention():
    """Verify counts dynamically reflect in-place status mutations without desynchronization."""
    registry = PedestrianRegistry()

    p1 = PedestrianState(id="ped_1", position=Point(10.0, 10.0), status=PedestrianStatus.ACTIVE)
    p2 = PedestrianState(id="ped_2", position=Point(20.0, 20.0), status=PedestrianStatus.ACTIVE)
    p3 = PedestrianState(id="ped_3", position=Point(30.0, 30.0), status=PedestrianStatus.INACTIVE)

    registry.register(p1)
    registry.register(p2)
    registry.register(p3)

    assert registry.total_count == 3
    assert registry.active_count == 2
    assert registry.evacuated_count == 0
    assert registry.blocked_count == 0
    assert registry.inactive_count == 1

    # Mutate p1 to EVACUATED via freeze_at_evacuation
    p1.freeze_at_evacuation()

    # Dynamic counts update immediately
    assert registry.total_count == 3
    assert registry.active_count == 1
    assert registry.evacuated_count == 1
    assert registry.blocked_count == 0
    assert registry.inactive_count == 1

    # Mutate p2 to BLOCKED
    p2.status = PedestrianStatus.BLOCKED

    assert registry.total_count == 3
    assert registry.active_count == 0
    assert registry.evacuated_count == 1
    assert registry.blocked_count == 1
    assert registry.inactive_count == 1

    # Evacuated pedestrian p1 remains registered and queryable
    assert "ped_1" in registry
    assert registry.get("ped_1") is p1
    assert p1.velocity == Velocity2D(0.0, 0.0)


def test_registry_clear():
    """Verify clear empties the registry."""
    registry = PedestrianRegistry()
    registry.register(PedestrianState(id="p1", position=Point(1, 1)))
    assert len(registry) == 1

    registry.clear()
    assert len(registry) == 0
    assert registry.total_count == 0
    assert registry.active_count == 0


def test_pedestrian_architectural_isolation():
    """Verify that importing genevac.pedestrian in a clean process does not import pygame, torch, mid, or numpy."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import genevac.pedestrian, sys; "
            "assert 'pygame' not in sys.modules, 'pygame was imported by pedestrian!'; "
            "assert 'torch' not in sys.modules, 'torch was imported!'; "
            "assert 'mid' not in sys.modules, 'mid was imported!'; "
            "assert 'numpy' not in sys.modules, 'numpy was imported by pedestrian!'; "
            "assert 'genevac.visualization' not in sys.modules, 'visualization was imported by pedestrian!'"
        ),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Pedestrian architectural isolation violation: {res.stderr}"
