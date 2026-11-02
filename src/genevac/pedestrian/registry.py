"""Pedestrian registry module for GenEvac-Sentinel.

Provides a population registry that:
- Stores pedestrians using unique stable IDs
- Rejects duplicate IDs
- Provides deterministic insertion-order iteration
- Supports status-based filtering
- Dynamically calculates accurate population counts (active, evacuated, blocked, inactive, total)
- Retains evacuated pedestrians indefinitely
"""

from __future__ import annotations

from typing import Iterator, Optional, Union

from genevac.pedestrian.models import PedestrianState, PedestrianStatus


class PedestrianRegistry:
    """Manages a population of simulated pedestrians.

    Guarantees:
    - Stable unique identifiers with duplicate rejection
    - Deterministic iteration order matching registration sequence
    - Dynamic count calculation reflecting actual current status without desynchronization
    - Permanent retention of all registered pedestrians (including evacuated agents)
    """

    def __init__(self) -> None:
        self._pedestrians: dict[str, PedestrianState] = {}

    def register(self, pedestrian: PedestrianState) -> None:
        """Register a pedestrian into the population.

        Raises:
            TypeError: If pedestrian is not an instance of PedestrianState.
            ValueError: If a pedestrian with the same ID is already registered.
        """
        if not isinstance(pedestrian, PedestrianState):
            raise TypeError(f"Expected PedestrianState, got {type(pedestrian).__name__}")

        if pedestrian.id in self._pedestrians:
            raise ValueError(f"Pedestrian with ID {pedestrian.id!r} is already registered")

        self._pedestrians[pedestrian.id] = pedestrian

    def get(self, pedestrian_id: str) -> Optional[PedestrianState]:
        """Look up a pedestrian by ID, returning None if not found."""
        return self._pedestrians.get(pedestrian_id)

    def __getitem__(self, pedestrian_id: str) -> PedestrianState:
        """Look up a pedestrian by ID, raising KeyError if not found."""
        if pedestrian_id not in self._pedestrians:
            raise KeyError(f"Pedestrian with ID {pedestrian_id!r} not found in registry")
        return self._pedestrians[pedestrian_id]

    def __contains__(self, pedestrian_id: str) -> bool:
        """Check if a pedestrian ID exists in the registry."""
        return pedestrian_id in self._pedestrians

    def __len__(self) -> int:
        """Return the total number of registered pedestrians."""
        return len(self._pedestrians)

    def __iter__(self) -> Iterator[PedestrianState]:
        """Iterate over all registered pedestrians in deterministic insertion order."""
        return iter(self._pedestrians.values())

    def all(self) -> list[PedestrianState]:
        """Return a list of all registered pedestrians in deterministic order."""
        return list(self._pedestrians.values())

    # --- Status Filtering ---

    def get_by_status(self, status: Union[PedestrianStatus, str]) -> list[PedestrianState]:
        """Return all pedestrians matching the specified status in deterministic order."""
        if isinstance(status, str) and not isinstance(status, PedestrianStatus):
            try:
                status = PedestrianStatus(status)
            except ValueError as err:
                raise ValueError(f"Invalid status filter: {status!r}") from err

        return [p for p in self._pedestrians.values() if p.status == status]

    def get_active(self) -> list[PedestrianState]:
        """Return all pedestrians with ACTIVE status."""
        return self.get_by_status(PedestrianStatus.ACTIVE)

    def get_evacuated(self) -> list[PedestrianState]:
        """Return all pedestrians with EVACUATED status."""
        return self.get_by_status(PedestrianStatus.EVACUATED)

    def get_blocked(self) -> list[PedestrianState]:
        """Return all pedestrians with BLOCKED status."""
        return self.get_by_status(PedestrianStatus.BLOCKED)

    def get_inactive(self) -> list[PedestrianState]:
        """Return all pedestrians with INACTIVE status."""
        return self.get_by_status(PedestrianStatus.INACTIVE)

    # --- Dynamic Population Counts ---

    @property
    def total_count(self) -> int:
        """Total number of registered pedestrians."""
        return len(self._pedestrians)

    @property
    def active_count(self) -> int:
        """Count of pedestrians currently in ACTIVE status."""
        return sum(1 for p in self._pedestrians.values() if p.status == PedestrianStatus.ACTIVE)

    @property
    def evacuated_count(self) -> int:
        """Count of pedestrians currently in EVACUATED status."""
        return sum(1 for p in self._pedestrians.values() if p.status == PedestrianStatus.EVACUATED)

    @property
    def blocked_count(self) -> int:
        """Count of pedestrians currently in BLOCKED status."""
        return sum(1 for p in self._pedestrians.values() if p.status == PedestrianStatus.BLOCKED)

    @property
    def inactive_count(self) -> int:
        """Count of pedestrians currently in INACTIVE status."""
        return sum(1 for p in self._pedestrians.values() if p.status == PedestrianStatus.INACTIVE)

    def clear(self) -> None:
        """Clear all registered pedestrians from the registry."""
        self._pedestrians.clear()
