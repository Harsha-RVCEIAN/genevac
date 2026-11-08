"""Authoritative simulation engine for GenEvac-Sentinel.

Coordinates:
- Authoritative simulation clock progression
- Deterministic waypoint generation and kinematic motion
- Wall and obstacle collision detection
- Exit boundary crossings and evacuation transitions
- Periodic observation snapshot generation at exact cadences
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Union

from genevac.environment.geometry import Point
from genevac.environment.models import Building
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.clock import SimulationClock
from genevac.simulation.events import SimulationEvent, SimulationEventType
from genevac.simulation.exceptions import NoRouteError
from genevac.simulation.movement import (
    advance_pedestrian_step,
    advance_population_step,
    check_pedestrian_overlap,
    find_walkable_area_for_point,
    generate_deterministic_route,
    is_position_in_walkable_space,
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
    - Records immutable simulation events deterministically
    """

    def __init__(
        self,
        building: Building,
        registry: Optional[Union[PedestrianRegistry, Iterable[PedestrianState]]] = None,
        dt: float = 0.1,
        observation_interval: float = 0.4,
        block_unassigned_exits: bool = True,
    ) -> None:
        if not isinstance(building, Building):
            raise TypeError(f"building must be a Building instance, got {type(building).__name__}")

        self._building: Building = building
        if registry is None:
            self._registry = PedestrianRegistry()
        elif isinstance(registry, PedestrianRegistry):
            self._registry = registry
        else:
            self._registry = PedestrianRegistry()
            for ped in registry:
                self._registry.register(ped)
        self._clock: SimulationClock = SimulationClock(dt=dt, observation_interval=observation_interval)

        self._is_paused: bool = False
        self._latest_snapshot: Optional[SimulationObservationSnapshot] = None
        self._initialized: bool = False

        self._events: list[SimulationEvent] = []
        self._new_events_this_step: list[SimulationEvent] = []
        self._event_counter: int = 0
        self._open_exit_ids: set[str] = {ex.id for ex in self._building.exits if ex.is_open}
        self._block_unassigned_exits: bool = block_unassigned_exits

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

    @property
    def events(self) -> tuple[SimulationEvent, ...]:
        """All simulation events recorded in this run so far."""
        return tuple(self._events)

    @property
    def latest_events(self) -> tuple[SimulationEvent, ...]:
        """Simulation events recorded during the most recent step or initialization."""
        return tuple(self._new_events_this_step)

    def get_events(
        self,
        event_type: Optional[Union[SimulationEventType, str]] = None,
        pedestrian_id: Optional[str] = None,
    ) -> tuple[SimulationEvent, ...]:
        """Filter recorded events by type and/or pedestrian ID."""
        type_str = (
            event_type.value
            if isinstance(event_type, SimulationEventType)
            else str(event_type) if event_type is not None else None
        )
        results = self._events
        if type_str is not None:
            results = [e for e in results if e.event_type == type_str]
        if pedestrian_id is not None:
            results = [e for e in results if e.pedestrian_id == pedestrian_id]
        return tuple(results)

    def _record_event(
        self,
        event_type: Union[SimulationEventType, str],
        pedestrian_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> SimulationEvent:
        """Record an immutable simulation event deterministically."""
        self._event_counter += 1
        type_str = event_type.value if isinstance(event_type, SimulationEventType) else str(event_type)
        evt_id = f"evt_{self._clock.step_count:06d}_{self._event_counter:04d}"
        evt = SimulationEvent(
            event_id=evt_id,
            step=self._clock.step_count,
            timestamp=self._clock.timestamp,
            event_type=type_str,
            pedestrian_id=pedestrian_id,
            details=details or {},
        )
        self._events.append(evt)
        self._new_events_this_step.append(evt)
        return evt

    def close_exit(self, exit_id: str) -> bool:
        """Close an exit by ID, recording an EXIT_CLOSED event.

        Also marks active pedestrians targeting this exit as BLOCKED with NO_ROUTE event.
        """
        ex = self._building.get_exit(exit_id)
        if ex is None:
            return False
        if ex.is_open:
            ex.close()
            if exit_id in self._open_exit_ids:
                self._open_exit_ids.remove(exit_id)
            self._record_event(
                event_type=SimulationEventType.EXIT_CLOSED,
                details={"exit_id": exit_id},
            )
            for ped in self._registry:
                if ped.status == PedestrianStatus.ACTIVE and ped.target_exit_id == exit_id:
                    ped.status = PedestrianStatus.BLOCKED
                    ped.velocity = Velocity2D(0.0, 0.0)
                    ped.waypoints.clear()
                    self._record_event(
                        event_type=SimulationEventType.NO_ROUTE,
                        pedestrian_id=ped.id,
                        details={"reason": "target_exit_closed", "exit_id": exit_id},
                    )
        return True

    def open_exit(self, exit_id: str) -> bool:
        """Open an exit by ID."""
        ex = self._building.get_exit(exit_id)
        if ex is None:
            return False
        if not ex.is_open:
            ex.open()
            self._open_exit_ids.add(exit_id)
        return True

    def get_snapshot(self) -> SimulationObservationSnapshot:
        """Return an immutable observation snapshot of current state on demand."""
        return create_observation_snapshot(
            registry=self._registry,
            step_count=self._clock.step_count,
            timestamp=self._clock.timestamp,
            dt=self._clock.dt,
            observation_interval=self._clock.observation_interval,
        )

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

            # Check current area exists if defined, or infer and validate walkable area containment
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
            elif self._building.walkable_areas:
                inferred_area = find_walkable_area_for_point(
                    building=self._building,
                    point=ped.position,
                    velocity=ped.velocity,
                )
                if inferred_area is None:
                    raise ValueError(
                        f"Pedestrian '{ped.id}' initial position ({ped.position.x}, {ped.position.y}) "
                        f"is outside all valid walkable areas in building '{self._building.id}'"
                    )
                ped.current_area_id = inferred_area.id

            # Validate pedestrian radius and physical clearance within walkable space
            target_ex = self._building.get_exit(ped.target_exit_id) if ped.target_exit_id else None
            if not is_position_in_walkable_space(ped.position, ped.radius, self._building, target_exit=target_ex):
                raise ValueError(
                    f"Pedestrian '{ped.id}' initial position ({ped.position.x}, {ped.position.y}) "
                    f"with radius {ped.radius} violates walkable space clearance (intersects obstacle, wall, or exterior boundary)"
                )

        # 2. Validate initial pedestrian separation (reject overlapping initial placements)
        peds_list = list(self._registry)
        for i in range(len(peds_list)):
            p_i = peds_list[i]
            for j in range(i + 1, len(peds_list)):
                p_j = peds_list[j]
                if check_pedestrian_overlap(p_i.position, p_i.radius, p_j.position, p_j.radius):
                    raise ValueError(
                        f"Initial pedestrian overlap detected between '{p_i.id}' at ({p_i.position.x}, {p_i.position.y}) "
                        f"(radius={p_i.radius}) and '{p_j.id}' at ({p_j.position.x}, {p_j.position.y}) (radius={p_j.radius})"
                    )

        # 3. Automatically generate deterministic route or handle no-route safely
        for ped in self._registry:
            if ped.status == PedestrianStatus.ACTIVE:
                if ped.target_exit_id is not None:
                    target_exit = self._building.get_exit(ped.target_exit_id)
                    if target_exit is not None and not target_exit.is_open:
                        ped.status = PedestrianStatus.BLOCKED
                        ped.velocity = Velocity2D(0.0, 0.0)
                        ped.waypoints.clear()
                        self._record_event(
                            event_type=SimulationEventType.NO_ROUTE,
                            pedestrian_id=ped.id,
                            details={"reason": "target_exit_closed", "target_exit_id": ped.target_exit_id},
                        )
                    elif not ped.waypoints:
                        start_area_id = ped.current_area_id
                        if start_area_id is not None:
                            try:
                                ped.waypoints = generate_deterministic_route(
                                    building=self._building,
                                    start_pos=ped.position,
                                    start_area_id=start_area_id,
                                    target_exit_id=ped.target_exit_id,
                                    radius=ped.radius,
                                )
                            except NoRouteError as err:
                                # Safe no-route handling: transition to BLOCKED and record event without crashing
                                ped.status = PedestrianStatus.BLOCKED
                                ped.velocity = Velocity2D(0.0, 0.0)
                                ped.waypoints.clear()
                                self._record_event(
                                    event_type=SimulationEventType.NO_ROUTE,
                                    pedestrian_id=ped.id,
                                    details={
                                        "reason": "unreachable_exit",
                                        "target_exit_id": ped.target_exit_id,
                                        "error": str(err),
                                    },
                                )
                else:
                    # Pedestrian has no assigned target exit (target_exit_id is None)
                    if self._block_unassigned_exits:
                        ped.status = PedestrianStatus.BLOCKED
                        ped.velocity = Velocity2D(0.0, 0.0)
                        ped.waypoints.clear()
                        self._record_event(
                            event_type=SimulationEventType.NO_ROUTE,
                            pedestrian_id=ped.id,
                            details={"reason": "unassigned_exit"},
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

        self._new_events_this_step.clear()

        # Track external exit state changes
        for ex in self._building.exits:
            if ex.id in self._open_exit_ids and not ex.is_open:
                self._open_exit_ids.remove(ex.id)
                self._record_event(
                    event_type=SimulationEventType.EXIT_CLOSED,
                    details={"exit_id": ex.id},
                )
                for ped in self._registry:
                    if ped.status == PedestrianStatus.ACTIVE and ped.target_exit_id == ex.id:
                        ped.status = PedestrianStatus.BLOCKED
                        ped.velocity = Velocity2D(0.0, 0.0)
                        ped.waypoints.clear()
                        self._record_event(
                            event_type=SimulationEventType.NO_ROUTE,
                            pedestrian_id=ped.id,
                            details={"reason": "target_exit_closed", "exit_id": ex.id},
                        )
            elif ex.id not in self._open_exit_ids and ex.is_open:
                self._open_exit_ids.add(ex.id)

        # 1. Advance movement for all active pedestrians with deterministic separation
        newly_blocked = advance_population_step(
            self._registry,
            self._building,
            self._clock.dt,
        )
        for ped in newly_blocked:
            self._record_event(
                event_type=SimulationEventType.BLOCKED,
                pedestrian_id=ped.id,
                details={"position": (ped.position.x, ped.position.y)},
            )

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
