"""Structured simulation events for GenEvac-Sentinel.

Provides:
- SimulationEventType: Enum defining standard event types (NO_ROUTE, EXIT_CLOSED, BLOCKED)
- SimulationEvent: Immutable event record with deterministic timestamp, step, and metadata
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum
import math
import types
from typing import Any, Mapping, Optional


class SimulationEventType(str, Enum):
    """Enumeration of standard simulation event types."""

    NO_ROUTE = "no_route"
    EXIT_CLOSED = "exit_closed"
    BLOCKED = "blocked"


def _freeze_value(val: Any) -> Any:
    """Recursively freeze mappings, sequences, and sets into immutable objects."""
    if isinstance(val, MappingABC):
        return types.MappingProxyType({k: _freeze_value(v) for k, v in val.items()})
    elif isinstance(val, (list, tuple)):
        return tuple(_freeze_value(item) for item in val)
    elif isinstance(val, (set, frozenset, AbstractSet)):
        return frozenset(_freeze_value(item) for item in val)
    return val


def _unfreeze_value(val: Any) -> Any:
    """Recursively unfreeze immutable mappings and sequences into standard dicts and lists."""
    if isinstance(val, MappingABC):
        return {k: _unfreeze_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_unfreeze_value(item) for item in val]
    elif isinstance(val, (set, frozenset, AbstractSet)):
        return list(_unfreeze_value(item) for item in val)
    return val


@dataclass(frozen=True)
class SimulationEvent:
    """Immutable, deterministic simulation event record.

    Attributes:
        event_id: Unique, deterministic identifier within the simulation run.
        step: Simulation step count at which the event occurred.
        timestamp: Authoritative simulation time in seconds.
        event_type: Type of event (NO_ROUTE, EXIT_CLOSED, BLOCKED, etc.).
        pedestrian_id: Optional ID of the affected pedestrian.
        details: Immutable mapping containing event-specific metadata.
    """

    event_id: str
    step: int
    timestamp: float
    event_type: str
    pedestrian_id: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not isinstance(self.event_id, str):
            raise ValueError(f"event_id must be a non-empty string, got {self.event_id!r}")

        if not isinstance(self.step, int) or isinstance(self.step, bool):
            raise TypeError(f"step must be an int, got {type(self.step).__name__}")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")

        try:
            ts_f = float(self.timestamp)
        except (TypeError, ValueError) as err:
            raise TypeError(f"timestamp must be numeric, got {self.timestamp!r}") from err
        if not (math.isfinite(ts_f) and ts_f >= 0.0):
            raise ValueError(f"timestamp must be finite and non-negative, got {self.timestamp}")

        if not self.event_type or not isinstance(self.event_type, str):
            raise ValueError(f"event_type must be a non-empty string, got {self.event_type!r}")

        if self.pedestrian_id is not None and not isinstance(self.pedestrian_id, str):
            raise TypeError(f"pedestrian_id must be str or None, got {type(self.pedestrian_id).__name__}")

        # Coerce details to an immutable recursive mapping
        if not isinstance(self.details, MappingABC):
            raise TypeError(f"details must be a mapping/dict, got {type(self.details).__name__}")

        frozen_details = _freeze_value(self.details)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "timestamp", ts_f)
        object.__setattr__(self, "details", frozen_details)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "pedestrian_id": self.pedestrian_id,
            "details": _unfreeze_value(self.details),
        }
