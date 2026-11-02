"""Authoritative simulation engine for GenEvac-Sentinel.

Coordinates:
- Authoritative simulation clock progression
- Deterministic waypoint generation and kinematic motion
- Wall and obstacle collision detection
- Exit boundary crossings and evacuation transitions
- Periodic observation snapshot generation at exact cadences
"""

from __future__ import annotations

from typing import Optional

from genevac.environment.geometry import Point
from genevac.environment.models import Building
from genevac.pedestrian.models import PedestrianState, PedestrianStatus
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.clock import SimulationClock
from genevac.simulation.movement import (
    advance_pedestrian_step,
    generate_deterministic_route,
)
from genevac.simulation.observation import (
    SimulationObservationSnapshot,
    create_observation_snapshot,
)


class SimulationEngine:
    """Simulation engine driving deterministic evacuation dynamics.

    The engine is the sole authoritative owner of ground-truth state:
    - Maintains the simulation clock
    - Updates pedestrian positions, velocities, and statuses
    - Enforces physical boundaries and geometric clearances
    - Emits read-only observation snapshots at fixed observation intervals
    """

    def __init__(
        self,
        building: Building,
        registry: Optional[PedestrianRegistry] = None,
        dt: float = 0.1,
        observation_interval: float = 0.4,
    ) -> None:
        if not isinstance(building, Building):
            raise TypeError(f"building must be a Building instance, got {type(building).__name__}")

        self._building: Building = building
        self._registry: PedestrianRegistry = registry if registry is not None else PedestrianRegistry()
        self._clock: SimulationClock = SimulationClock(dt=dt, observation_interval=observation_interval)

        self._is_paused: bool = False
        self._latest_snapshot: Optional[SimulationObservationSnapshot] = None
        self._initialized: bool = False

        self._initialize_simulation()

    @property
    def building(self) -> Building:
        """The underlying building environment model."""
        return self._building

    @property
    def registry(self) -> PedestrianRegistry:
        """The active pedestrian population registry."""
        return self._registry

    @property
    def clock(self) -> SimulationClock:
        """The authoritative simulation clock."""
        return self._clock

    @property
    def is_paused(self) -> bool:
        """Return True if simulation stepping is paused."""
        return self._is_paused

    @property
    def latest_snapshot(self) -> Optional[SimulationObservationSnapshot]:
        """The most recently generated observation snapshot."""
        return self._latest_snapshot

    def pause(self) -> None:
        """Pause simulation execution."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume simulation execution."""
        self._is_paused = False

    def _initialize_simulation(self) -> None:
        """Validate initial pedestrian states, compute initial routes, and record t=0 snapshot."""
        # 1. Validate pedestrians against building geometry
        for ped in self._registry:
            # Check building containment
            if not (
                0.0 <= ped.position.x <= self._building.width
                and 0.0 <= ped.position.y <= self._building.height
            ):
                raise ValueError(
                    f"Pedestrian '{ped.id}' initial position ({ped.position.x}, {ped.position.y}) "
                    f"is outside building bounds [0.0, {self._building.width}] x [0.0, {self._building.height}]"
                )

            # Check target exit exists if defined
            if ped.target_exit_id is not None:
                if self._building.get_exit(ped.target_exit_id) is None:
                    raise ValueError(
                        f"Pedestrian '{ped.id}' references non-existent target exit '{ped.target_exit_id}'"
                    )

            # Check current area exists if defined
            if ped.current_area_id is not None:
                area = self._building.get_walkable_area(ped.current_area_id)
                if area is None:
                    raise ValueError(
                        f"Pedestrian '{ped.id}' references non-existent walkable area '{ped.current_area_id}'"
                    )
                if not area.bounds.contains_point(ped.position):
                    raise ValueError(
                        f"Pedestrian '{ped.id}' initial position ({ped.position.x}, {ped.position.y}) "
                        f"is not within declared area '{ped.current_area_id}' bounds"
                    )

            # 2. Automatically generate deterministic route if waypoints are empty and exit is assigned
            if not ped.waypoints and ped.target_exit_id is not None and ped.status == PedestrianStatus.ACTIVE:
                # Infer starting area if not explicitly supplied
                start_area_id = ped.current_area_id
                if start_area_id is None:
                    for wa in self._building.walkable_areas:
                        if wa.bounds.contains_point(ped.position):
                            start_area_id = wa.id
                            ped.current_area_id = wa.id
                            break

                if start_area_id is not None:
                    ped.waypoints = generate_deterministic_route(
                        building=self._building,
                        start_pos=ped.position,
                        start_area_id=start_area_id,
                        target_exit_id=ped.target_exit_id,
                    )

            # 3. Record initial t=0 history sample if history is empty
            if not ped.trajectory_history:
                ped.record_history_sample(step=0, timestamp=0.0)

        # 4. Generate initial snapshot at step 0 (t = 0.0)
        self._latest_snapshot = create_observation_snapshot(
            registry=self._registry,
            step_count=self._clock.step_count,
            timestamp=self._clock.timestamp,
            dt=self._clock.dt,
            observation_interval=self._clock.observation_interval,
        )
        self._initialized = True

    def step(self) -> Optional[SimulationObservationSnapshot]:
        """Advance the simulation deterministically by one step (dt).

        Returns:
            A new SimulationObservationSnapshot if the new step is an observation tick,
            or None if it is an intermediate step.
        """
        if self._is_paused:
            return None

        # 1. Advance movement for all active pedestrians
        for ped in self._registry:
            if ped.status == PedestrianStatus.ACTIVE:
                advance_pedestrian_step(ped, self._building, self._clock.dt)

        # 2. Advance simulation clock
        new_step, new_timestamp = self._clock.tick()

        # 3. Handle observation tick
        if self._clock.is_observation_step():
            # Record timestamped trajectory history for all pedestrians
            for ped in self._registry:
                ped.record_history_sample(step=new_step, timestamp=new_timestamp)

            # Emit new immutable observation snapshot
            snapshot = create_observation_snapshot(
                registry=self._registry,
                step_count=new_step,
                timestamp=new_timestamp,
                dt=self._clock.dt,
                observation_interval=self._clock.observation_interval,
            )
            self._latest_snapshot = snapshot
            return snapshot

        return None

    def run_steps(self, num_steps: int) -> list[SimulationObservationSnapshot]:
        """Advance the simulation by num_steps, collecting all snapshots produced.

        Args:
            num_steps: Number of simulation steps to execute.

        Returns:
            List of SimulationObservationSnapshot instances generated during execution.
        """
        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}")

        snapshots: list[SimulationObservationSnapshot] = []
        for _ in range(num_steps):
            snap = self.step()
            if snap is not None:
                snapshots.append(snap)
        return snapshots

    def step_until(self, target_step: int) -> list[SimulationObservationSnapshot]:
        """Advance the simulation until step_count reaches target_step.

        Args:
            target_step: Desired step count.

        Returns:
            List of SimulationObservationSnapshot instances generated during execution.
        """
        if target_step < self._clock.step_count:
            raise ValueError(
                f"target_step ({target_step}) cannot be earlier than current step ({self._clock.step_count})"
            )

        steps_needed = target_step - self._clock.step_count
        return self.run_steps(steps_needed)
