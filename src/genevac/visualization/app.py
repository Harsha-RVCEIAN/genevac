"""Interactive Pygame application for GenEvac-Sentinel simulation visualization.

Integrates authoritative SimulationEngine with EnvironmentRenderer:
- Renders building geometry, obstacles, doorways, and exits
- Displays pedestrians at true physical simulation positions with status indicators and ID badges
- Shows live simulation HUD (time, step, counts, pause status, control hints)
- Supports interactive controls: [SPACE] pause/resume, [R] deterministic reset, [Q/ESC] quit
- Manages fixed-timestep simulation updates decoupled from frame rate via an accumulator
  with a 5-step catch-up ceiling and wall-clock frame delta clamping.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys
from typing import Optional
import pygame

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import Point
from genevac.environment.models import Building
from genevac.environment.serialization import load_building
from genevac.pedestrian.models import PedestrianState, PedestrianStatus
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.engine import SimulationEngine
from genevac.visualization.coordinates import Viewport
from genevac.visualization.renderer import EnvironmentRenderer


def create_default_pedestrian_cohort() -> list[PedestrianState]:
    """Create a deterministic initial cohort of 8 pedestrians for the default building."""
    return [
        PedestrianState(
            id="ped_1",
            position=Point(8.0, 6.0),
            target_exit_id="exit_left",
            current_area_id="room_1",
            preferred_speed=1.3,
            radius=0.3,
        ),
        PedestrianState(
            id="ped_2",
            position=Point(14.0, 8.0),
            target_exit_id="exit_left",
            current_area_id="room_1",
            preferred_speed=1.2,
            radius=0.3,
        ),
        PedestrianState(
            id="ped_3",
            position=Point(28.0, 6.0),
            target_exit_id="exit_right",
            current_area_id="room_2",
            preferred_speed=1.4,
            radius=0.3,
        ),
        PedestrianState(
            id="ped_4",
            position=Point(35.0, 8.0),
            target_exit_id="exit_right",
            current_area_id="room_2",
            preferred_speed=1.2,
            radius=0.3,
        ),
        PedestrianState(
            id="ped_5",
            position=Point(20.0, 15.0),
            target_exit_id="exit_left",
            current_area_id="main_corridor",
            preferred_speed=1.5,
            radius=0.3,
        ),
        PedestrianState(
            id="ped_6",
            position=Point(12.0, 24.0),
            target_exit_id="exit_bottom",
            current_area_id="room_3",
            preferred_speed=1.3,
            radius=0.3,
        ),
        PedestrianState(
            id="ped_7",
            position=Point(28.0, 24.0),
            target_exit_id="exit_right",
            current_area_id="room_4",
            preferred_speed=1.3,
            radius=0.3,
        ),
        PedestrianState(
            id="ped_8",
            position=Point(35.0, 22.0),
            target_exit_id="exit_right",
            current_area_id="room_4",
            preferred_speed=1.4,
            radius=0.3,
        ),
    ]


class SimulationApp:
    """Interactive Pygame application managing SimulationEngine and EnvironmentRenderer.

    Authoritative simulation state is completely owned by SimulationEngine.
    The renderer is strictly read-only.
    Time accumulation ensures decoupled fixed-timestep simulation stepping with
    catch-up clamping (max 5 steps per frame) and residual time retention.
    Deterministic reset creates fresh, unmutated pedestrian instances via population_factory.
    """

    def __init__(
        self,
        building: Optional[Building] = None,
        engine: Optional[SimulationEngine] = None,
        population_factory: Optional[Callable[[], list[PedestrianState]]] = None,
        dt: float = 0.1,
        observation_interval: float = 0.4,
        width: int = 1200,
        height: int = 800,
        fps: int = 30,
    ) -> None:
        self._dt = dt
        self._observation_interval = observation_interval
        self._width = width
        self._height = height
        self._fps = fps
        self._sim_accumulator: float = 0.0
        self._max_catchup_steps: int = 5
        self._max_dt_wall: float = 0.25

        # Configure population factory
        self._population_factory = (
            population_factory
            if population_factory is not None
            else create_default_pedestrian_cohort
        )

        # Resolve building and authoritative engine
        if engine is not None:
            self._engine = engine
            self._building = engine.building
            self._dt = engine.clock.dt
            self._observation_interval = engine.clock.observation_interval
        else:
            self._building = (
                building if building is not None else create_initial_building()
            )
            initial_cohort = self._population_factory()
            registry = PedestrianRegistry()
            for ped in initial_cohort:
                registry.register(ped)
            self._engine = SimulationEngine(
                building=self._building,
                registry=registry,
                dt=self._dt,
                observation_interval=self._observation_interval,
            )

        # Setup coordinate viewport and renderer
        self._viewport = Viewport(
            world_width=self._building.width,
            world_height=self._building.height,
            screen_width=self._width,
            screen_height=self._height,
        )
        self._renderer = EnvironmentRenderer()
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

    @property
    def engine(self) -> SimulationEngine:
        """The authoritative simulation engine."""
        return self._engine

    @property
    def building(self) -> Building:
        """The underlying building model."""
        return self._building

    @property
    def registry(self) -> PedestrianRegistry:
        """The active pedestrian registry."""
        return self._engine.registry

    @property
    def viewport(self) -> Viewport:
        """The coordinate conversion viewport."""
        return self._viewport

    @property
    def renderer(self) -> EnvironmentRenderer:
        """The read-only environment renderer."""
        return self._renderer

    @property
    def is_paused(self) -> bool:
        """Return True if simulation stepping is paused."""
        return self._engine.is_paused

    @property
    def sim_accumulator(self) -> float:
        """Current unprocessed simulation accumulator time in seconds."""
        return self._sim_accumulator

    @property
    def dt(self) -> float:
        """Simulation physics timestep in seconds."""
        return self._dt

    def pause(self) -> None:
        """Pause simulation stepping."""
        self._engine.pause()

    def resume(self) -> None:
        """Resume simulation stepping."""
        self._engine.resume()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        if self._engine.is_paused:
            self.resume()
        else:
            self.pause()

    def reset(self) -> None:
        """Deterministically reset simulation state using fresh population factory instances."""
        fresh_cohort = self._population_factory()
        fresh_registry = PedestrianRegistry()
        for ped in fresh_cohort:
            fresh_registry.register(ped)
        was_paused = self._engine.is_paused

        self._engine = SimulationEngine(
            building=self._building,
            registry=fresh_registry,
            dt=self._dt,
            observation_interval=self._observation_interval,
        )
        if was_paused:
            self._engine.pause()

        self._sim_accumulator = 0.0

    def update(self, dt_wall: float) -> int:
        """Advance simulation using decoupled fixed-timestep accumulator.

        Clamps dt_wall to a maximum of 0.25 seconds.
        If paused, does not step simulation and does not accumulate time.
        Executes up to 5 steps per frame. Retains unprocessed remainder.

        Args:
            dt_wall: Elapsed wall-clock time in seconds.

        Returns:
            The number of simulation steps executed in this update (0 to 5).
        """
        if self._engine.is_paused:
            return 0

        clamped_dt = max(0.0, min(dt_wall, self._max_dt_wall))
        self._sim_accumulator += clamped_dt

        steps_taken = 0
        while self._sim_accumulator >= self._dt and steps_taken < self._max_catchup_steps:
            self._engine.step()
            self._sim_accumulator -= self._dt
            steps_taken += 1

        return steps_taken

    def resize(self, width: int, height: int) -> None:
        """Handle window resizing while preserving building aspect ratio."""
        self._width = max(100, width)
        self._height = max(100, height)
        if pygame.display.get_surface() is not None:
            self._screen = pygame.display.set_mode(
                (self._width, self._height), pygame.RESIZABLE
            )
        self._viewport.update_screen_size(self._width, self._height)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle an individual Pygame event. Returns False if application should quit."""
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.VIDEORESIZE:
            self.resize(event.w, event.h)
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            elif event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_r:
                self.reset()
        return True

    def render(self, surface: Optional[pygame.Surface] = None) -> None:
        """Render the current environment and simulation state onto the given surface."""
        target = surface if surface is not None else self._screen
        if target is None:
            raise RuntimeError("Cannot render without a target surface or initialized display")
        self._renderer.draw(target, self._building, self._viewport, engine=self._engine)

    def run(self, max_frames: Optional[int] = None) -> None:
        """Execute the Pygame application main loop."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()

        pygame.display.set_caption(
            f"GenEvac-Sentinel — {self._building.id} ({self._building.width:.0f}m x {self._building.height:.0f}m)"
        )

        self._screen = pygame.display.set_mode(
            (self._width, self._height), pygame.RESIZABLE
        )
        self._clock = pygame.time.Clock()

        running = True
        frames = 0
        while running:
            dt_millis = self._clock.tick(self._fps)
            dt_wall = dt_millis / 1000.0

            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
                    break

            if not running:
                break

            self.update(dt_wall)
            self.render(self._screen)
            pygame.display.flip()

            frames += 1
            if max_frames is not None and frames >= max_frames:
                break


def run_app(
    building: Optional[Building] = None,
    json_path: Optional[str | Path] = None,
    population_factory: Optional[Callable[[], list[PedestrianState]]] = None,
    width: int = 1200,
    height: int = 800,
    dt: float = 0.1,
    max_frames: Optional[int] = None,
) -> None:
    """Run the Pygame visualization window for a Building environment with simulation integration."""
    # Resolve building model
    if building is None:
        if json_path is not None:
            building = load_building(json_path)
        else:
            building = create_initial_building()

    app = SimulationApp(
        building=building,
        population_factory=population_factory,
        width=width,
        height=height,
        dt=dt,
    )

    try:
        app.run(max_frames=max_frames)
    finally:
        if pygame.get_init():
            pygame.quit()


def main() -> None:
    """CLI entry point for the visualization application."""
    parser = argparse.ArgumentParser(
        description="GenEvac-Sentinel: 2D Pygame Simulation Visualization"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to a building JSON file to visualize. If omitted, the default 40x30m initial building is displayed.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Initial window width in pixels (default: 1200).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Initial window height in pixels (default: 800).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation physics timestep in seconds (default: 0.1).",
    )

    args = parser.parse_args()

    try:
        run_app(json_path=args.json, width=args.width, height=args.height, dt=args.dt)
    except Exception as err:
        print(f"Error launching visualization: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
