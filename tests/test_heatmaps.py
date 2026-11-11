"""Comprehensive unit and regression tests for Phase 04 Spatial Heatmap Engine.

Verifies:
1. Legacy HeatmapCalculator backwards compatibility.
2. GridSpecification geometric validation and boundary conventions.
3. Coordinate mapping (half-open cells, exact max-boundary inclusion, out-of-bounds rejection).
4. Hand-calculated numerical fixtures: occupancy counts, physical density (peds/m^2), dwell time.
5. Per-agent sample-hold dwell attribution (initial observation 0.0, exact delta_t attribution).
6. Run isolation and duplicate protection across occupancy and dwell time.
7. Zero mutation of simulation engine or snapshot objects.
"""

import math
import unittest
import numpy as np

from genevac.analytics import (
    GridSpecification,
    HeatmapCalculator,
    SpatialGridResult,
    SpatialHeatmapEngine,
)
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    SimulationTrajectoryAdapter,
    TrajectoryObservation,
    TrajectorySnapshot,
)
from genevac.simulation import SimulationEngine


class TestSpatialHeatmapEnginePhase04(unittest.TestCase):
    def test_legacy_calculator_compatibility(self):
        """Preserves exact behavior for legacy HeatmapCalculator tests."""
        calc = HeatmapCalculator(grid_width=10.0, grid_height=10.0, resolution=2.0)
        calc.record_positions([(1.0, 1.0), (1.5, 1.5), (5.0, 5.0)])
        grid = calc.get_density_matrix()
        self.assertEqual(grid[0, 0], 2.0)
        self.assertEqual(grid[2, 2], 1.0)

    def test_grid_specification_validation(self):
        """Rejects invalid, zero, negative, or inverted grid bounds."""
        with self.assertRaises(ValueError):
            GridSpecification(min_x=10.0, min_y=0.0, max_x=5.0, max_y=10.0, cell_size_x=1.0, cell_size_y=1.0)
        with self.assertRaises(ValueError):
            GridSpecification(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, cell_size_x=-1.0, cell_size_y=1.0)
        with self.assertRaises(ValueError):
            GridSpecification(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, cell_size_x=float("nan"), cell_size_y=1.0)

        spec = GridSpecification(0.0, 0.0, 10.0, 10.0, 2.0, 2.0)
        self.assertEqual(spec.rows, 5)
        self.assertEqual(spec.cols, 5)
        self.assertEqual(spec.cell_area, 4.0)

    def test_coordinate_mapping_and_boundaries(self):
        """Verifies top-left floor indexing, max edge inclusion, and out-of-bounds rejection."""
        spec = GridSpecification(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, cell_size_x=2.0, cell_size_y=2.0)
        engine = SpatialHeatmapEngine(spec)

        # Standard interior points
        self.assertEqual(engine.map_point_to_cell(1.0, 1.0), (0, 0))
        self.assertEqual(engine.map_point_to_cell(2.0, 2.0), (1, 1))

        # Overall maximum edge mapping
        self.assertEqual(engine.map_point_to_cell(10.0, 10.0), (4, 4))
        self.assertEqual(engine.map_point_to_cell(10.0, 5.0), (2, 4))

        # Out-of-bounds rejection
        self.assertIsNone(engine.map_point_to_cell(-0.1, 5.0))
        self.assertIsNone(engine.map_point_to_cell(10.1, 5.0))
        self.assertIsNone(engine.map_point_to_cell(5.0, 15.0))
        self.assertIsNone(engine.map_point_to_cell(float("nan"), 5.0))

    def test_hand_calculated_occupancy_and_density_fixture(self):
        """Validates exact numerical values against hand-calculated fixture."""
        spec = GridSpecification(0.0, 0.0, 10.0, 10.0, 2.0, 2.0)
        engine = SpatialHeatmapEngine(spec)

        positions = [
            (1.0, 1.0),   # Agent A -> (0, 0)
            (1.5, 1.5),   # Agent B -> (0, 0)
            (5.0, 5.0),   # Agent C -> (2, 2)
            (10.0, 10.0), # Agent D -> (4, 4)
            (15.0, 5.0),  # Agent E -> out-of-bounds
        ]

        result = engine.aggregate_positions(positions, timestamp=0.0, step_index=0)

        self.assertEqual(result.total_agents_processed, 5)
        self.assertEqual(result.accepted_agents, 4)
        self.assertEqual(result.rejected_agents, 1)

        # Occupancy counts
        self.assertEqual(result.occupancy_grid[0, 0], 2)
        self.assertEqual(result.occupancy_grid[2, 2], 1)
        self.assertEqual(result.occupancy_grid[4, 4], 1)
        self.assertEqual(result.occupancy_grid[1, 1], 0)

        # Physical density (peds / m^2, area = 4.0 m^2)
        self.assertAlmostEqual(result.density_grid[0, 0], 2 / 4.0)  # 0.5
        self.assertAlmostEqual(result.density_grid[2, 2], 1 / 4.0)  # 0.25
        self.assertAlmostEqual(result.density_grid[4, 4], 1 / 4.0)  # 0.25
        self.assertAlmostEqual(result.density_grid[0, 1], 0.0)

        # Min-max normalization
        norm = result.normalize_min_max("density")
        self.assertAlmostEqual(norm[0, 0], 1.0)
        self.assertAlmostEqual(norm[2, 2], 0.5)
        self.assertAlmostEqual(norm[0, 1], 0.0)

    def test_dwell_time_per_agent_interval_attribution(self):
        """Verifies exact interval dwell time attribution under sample-hold policy."""
        spec = GridSpecification(0.0, 0.0, 10.0, 10.0, 2.0, 2.0)
        engine = SpatialHeatmapEngine(spec)

        # Step 0: t=0.0. Agent p1 at (1.0, 1.0) in cell (0, 0).
        # Initial observation must contribute 0.0 dwell time.
        snap0 = TrajectorySnapshot(
            current_step=0,
            current_timestamp=0.0,
            timestep=0.1,
            trajectories={"p1": (TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0),)},
        )
        res0 = engine.aggregate_snapshot(snap0, run_id="run_1")
        self.assertEqual(res0.occupancy_grid[0, 0], 1)
        self.assertEqual(res0.dwell_time_grid[0, 0], 0.0)

        # Step 4: t=0.4. Agent p1 moves to (3.0, 1.0) in cell (0, 1).
        # Elapsed interval delta_t = 0.4s attributed to previous cell (0, 0).
        snap4 = TrajectorySnapshot(
            current_step=4,
            current_timestamp=0.4,
            timestep=0.1,
            trajectories={"p1": (
                TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0),
                TrajectoryObservation("p1", 4, 0.4, 3.0, 1.0),
            )},
        )
        res4 = engine.aggregate_snapshot(snap4, run_id="run_1")
        self.assertEqual(res4.occupancy_grid[0, 1], 1)
        self.assertEqual(res4.occupancy_grid[0, 0], 0)
        self.assertAlmostEqual(res4.dwell_time_grid[0, 0], 0.4)
        self.assertEqual(res4.dwell_time_grid[0, 1], 0.0)

    def test_run_isolation_and_deduplication(self):
        """Verifies duplicate suppression and run isolation enforcement."""
        spec = GridSpecification(0.0, 0.0, 10.0, 10.0, 2.0, 2.0)
        engine = SpatialHeatmapEngine(spec)

        snap0 = TrajectorySnapshot(
            current_step=0,
            current_timestamp=0.0,
            timestep=0.1,
            trajectories={"p1": (TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0),)},
        )

        # First ingestion
        res1 = engine.aggregate_snapshot(snap0, run_id="run_A")
        self.assertFalse(res1.is_duplicate)
        self.assertEqual(res1.accepted_agents, 1)

        # Duplicate ingestion of same snapshot
        res2 = engine.aggregate_snapshot(snap0, run_id="run_A")
        self.assertTrue(res2.is_duplicate)
        self.assertEqual(res2.accepted_agents, 0)

        # Accidental mixing of different run_id raises ValueError
        with self.assertRaises(ValueError):
            engine.aggregate_snapshot(snap0, run_id="run_B")

        # Explicit reset allows clean switch to run_B
        engine.reset(new_run_id="run_B")
        res_b = engine.aggregate_snapshot(snap0, run_id="run_B")
        self.assertFalse(res_b.is_duplicate)
        self.assertEqual(res_b.run_id, "run_B")

    def test_engine_non_mutation(self):
        """Proof that heatmap aggregation does not advance or mutate simulation engine."""
        env = EnvironmentMap(width=20.0, height=20.0)
        sim = SimulationEngine(environment=env, max_steps=100, timestep=0.1)
        ped = Pedestrian("p_safe", (5.0, 5.0), velocity=(1.0, 0.0), status=PedestrianStatus.ACTIVE)
        sim.add_pedestrian(ped)

        before_step = sim.current_step
        before_pos = ped.position
        before_vel = ped.velocity
        before_status = ped.status
        before_peds_len = len(sim.pedestrians)

        spec = GridSpecification(0.0, 0.0, 20.0, 20.0, 2.0, 2.0)
        engine = SpatialHeatmapEngine(spec)
        engine.aggregate_positions([p.position for p in sim.pedestrians])

        self.assertEqual(sim.current_step, before_step)
        self.assertEqual(ped.position, before_pos)
        self.assertEqual(ped.velocity, before_vel)
        self.assertEqual(ped.status, before_status)
        self.assertEqual(len(sim.pedestrians), before_peds_len)


if __name__ == "__main__":
    unittest.main()
