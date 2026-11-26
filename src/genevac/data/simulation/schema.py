# -*- coding: utf-8 -*-
"""Versioned schema for GenEvac simulation recordings.

The schema is deliberately lightweight and uses only the standard library
(dataclasses) so that it works both in the test environment (where third‑party
packages such as pydantic may not be installed) and in the production code.

All fields are documented with type, unit and nullability information.
Version ``v1`` is the current schema version.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "v1"

# ---------------------------------------------------------------------------
# Run‑level metadata (one JSON file per run)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunMetadata:
    """Metadata describing a single simulation run.

    Fields
    ------
    run_id: str
        UUID4 string uniquely identifying the run.
    scenario_id: str
        Identifier of the ScenarioDefinition used.
    layout_id: str
        Identifier of the building layout (derived from the scenario).
    seed: int
        Random seed supplied to the SimulationEngine.
    population: int
        Number of pedestrians injected at the start of the run.
    environment_width: float
        Width of the environment in metres.
    environment_height: float
        Height of the environment in metres.
    timestep: float
        Simulation time step in seconds.
    max_steps: int
        Maximum number of steps configured for the engine.
    total_steps: int
        Number of steps actually executed (may be < max_steps if finished early).
    evacuated: int
        Count of pedestrians that reached an exit.
    start_time: str
        ISO‑8601 timestamp when the run began.
    end_time: str
        ISO‑8601 timestamp when the run finished.
    status: str
        ``"completed"``, ``"interrupted"`` or ``"failed"``.
    failure_reason: Optional[str]
        Human readable reason if the run failed.
    """

    run_id: str
    scenario_id: str
    layout_id: str
    seed: int
    population: int
    environment_width: float
    environment_height: float
    timestep: float
    max_steps: int
    total_steps: int
    evacuated: int
    start_time: str
    end_time: str
    status: str
    failure_reason: Optional[str] = None
    termination_reason: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to a compact JSON string (no indentation)."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> "RunMetadata":
        data = json.loads(text)
        return RunMetadata(**data)

# ---------------------------------------------------------------------------
# Trajectory record (one JSON line per observation)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryRecord:
    """A single observation of a pedestrian during a simulation run.

    The schema mirrors :class:`genevac.prediction.data_contract.TrajectoryObservation`
    with the addition of ``run_id`` and ``scenario_id`` for provenance.
    """

    run_id: str
    scenario_id: str
    layout_id: str
    pedestrian_id: str
    step_index: int
    timestamp: float
    x: float
    y: float
    vx: Optional[float] = None  # metres/second, may be null if not exposed
    vy: Optional[float] = None
    speed: Optional[float] = None  # derived speed, optional
    status: Optional[str] = None  # authoritative status string, may be null
    category: str = "simulated"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> "TrajectoryRecord":
        data = json.loads(text)
        return TrajectoryRecord(**data)

# ---------------------------------------------------------------------------
# Convenience utilities used by the recorder / validator
# ---------------------------------------------------------------------------
def generate_run_id() -> str:
    """Return a new UUID4 string for a run identifier."""
    return str(uuid.uuid4())

def iso_timestamp() -> str:
    """Current UTC time as an ISO‑8601 string (no microseconds)."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
