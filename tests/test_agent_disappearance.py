"""Tests verifying agent disappearance, lifecycle transitions, and freshness vs history length."""

import unittest
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    TrajectoryHistoryBuffer,
    TrajectoryValidationError,
)
from genevac.simulation import SimulationEngine


class TestAgentDisappearanceAndLifecycle(unittest.TestCase):
    """Test suite for agent disappearance, evacuation, and lifecycle retention."""

    def test_history_retained_when_agent_stops_being_recorded(self):
        """When an agent disappears or evacuates, existing observations remain in buffer."""
        buffer = TrajectoryHistoryBuffer()
        # Record agent over steps 0, 1, 2
        for s in range(3):
            buffer.record("ped_evac", (float(s), float(s)), step_index=s, timestamp=s * 0.1)

        # Agent evacuates or disappears, no new records for steps 3, 4, 5
        self.assertEqual(len(buffer.get_observations("ped_evac")), 3)
        self.assertEqual(buffer.get_observations("ped_evac")[-1].step_index, 2)

    def test_has_sufficient_history_remains_true_for_disappeared_agent(self):
        """has_sufficient_history checks count of observations, NOT freshness."""
        buffer = TrajectoryHistoryBuffer(min_history_required=2)
        buffer.record("ped_disappeared", (0.0, 0.0), step_index=0, timestamp=0.0)
        buffer.record("ped_disappeared", (1.0, 1.0), step_index=1, timestamp=0.1)

        # Buffer has 2 observations, so history length is sufficient
        self.assertTrue(buffer.has_sufficient_history("ped_disappeared"))

        # Even at a much later simulation step, history count alone is still >= 2
        self.assertTrue(buffer.has_sufficient_history("ped_disappeared"))

    def test_freshness_check_distinguishes_history_length_from_recency(self):
        """Buffer differentiates between having sufficient history and being fresh."""
        buffer = TrajectoryHistoryBuffer(min_history_required=2, max_observation_age=0.2)
        buffer.record("ped_old", (0.0, 0.0), step_index=0, timestamp=0.0)
        buffer.record("ped_old", (1.0, 1.0), step_index=1, timestamp=0.1)

        # Sufficient history length is True
        self.assertTrue(buffer.has_sufficient_history("ped_old"))

        # Fresh at snapshot t=0.2 (age = 0.1 <= 0.2)
        self.assertTrue(buffer.is_latest_observation_fresh("ped_old", reference_timestamp=0.2))

        # Stale at snapshot t=1.0 (age = 0.9 > 0.2)
        self.assertTrue(buffer.has_sufficient_history("ped_old"))  # Still has 2 observations!
        self.assertFalse(buffer.is_latest_observation_fresh("ped_old", reference_timestamp=1.0))  # But not fresh!

        # Attempting snapshot at t=1.0 raises TrajectoryValidationError
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buffer.get_snapshot(current_step=10, current_timestamp=1.0, timestep=0.1)
        self.assertEqual(ctx.exception.field, "observation_age")
        self.assertEqual(ctx.exception.agent_id, "ped_old")

    def test_simulation_evacuation_lifecycle_integration(self):
        """Pedestrian status change in SimulationEngine (EVACUATED / INCAPACITATED) does not auto-clear buffer."""
        env = EnvironmentMap(width=20.0, height=20.0)
        engine = SimulationEngine(environment=env, timestep=0.1)
        ped_active = Pedestrian(id="p_active", position=(0.0, 0.0), velocity=(1.0, 0.0))
        ped_evac = Pedestrian(id="p_evac", position=(10.0, 10.0), velocity=(0.0, 0.0))
        ped_incap = Pedestrian(id="p_incap", position=(5.0, 5.0), velocity=(0.0, 0.0))

        engine.add_pedestrian(ped_active)
        engine.add_pedestrian(ped_evac)
        engine.add_pedestrian(ped_incap)

        buffer = TrajectoryHistoryBuffer(max_observation_age=0.25)

        # Step 0: Record all
        for p in engine.pedestrians:
            buffer.record(p.id, p.position, engine.current_step, engine.current_step * engine.timestep)

        # Step 1: Record all
        engine.step()
        for p in engine.pedestrians:
            buffer.record(p.id, p.position, engine.current_step, engine.current_step * engine.timestep)

        # Engine lifecycle transition: p_evac evacuates, p_incap becomes incapacitated
        ped_evac.status = PedestrianStatus.EVACUATED
        ped_incap.status = PedestrianStatus.INCAPACITATED

        # Simulation loop only records ACTIVE pedestrians
        for _ in range(4):
            engine.step()
            cur_t = engine.current_step * engine.timestep
            for p in engine.pedestrians:
                if p.status == PedestrianStatus.ACTIVE:
                    buffer.record(p.id, p.position, engine.current_step, cur_t)

        cur_t = engine.current_step * engine.timestep  # step 5, t=0.5
        self.assertEqual(engine.current_step, 5)

        # Buffer did NOT automatically clear past history of p_evac or p_incap
        self.assertEqual(len(buffer.get_observations("p_evac")), 2)
        self.assertEqual(len(buffer.get_observations("p_incap")), 2)
        self.assertEqual(len(buffer.get_observations("p_active")), 6)

        # Querying active-only agents produces a valid snapshot and prediction
        snap_active = buffer.get_snapshot(
            current_step=engine.current_step,
            current_timestamp=cur_t,
            timestep=0.1,
            agent_ids=["p_active"],
        )
        predictor = ConstantVelocityPredictor()
        pred_active = predictor.predict(snapshot=snap_active, agent_id="p_active", horizon=3)
        self.assertEqual(len(pred_active.predictions), 3)

        # Querying all agents without filtering includes stale agents whose last obs is at t=0.1 (age=0.4 > 0.25)
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buffer.get_snapshot(
                current_step=engine.current_step,
                current_timestamp=cur_t,
                timestep=0.1,
            )
        self.assertEqual(ctx.exception.field, "observation_age")
        self.assertIn(ctx.exception.agent_id, {"p_evac", "p_incap"})

        # Explicit clearing remains available when the lifecycle caller explicitly decides to retire history
        buffer.clear("p_evac")
        self.assertEqual(len(buffer.get_observations("p_evac")), 0)
        self.assertEqual(len(buffer.get_observations("p_incap")), 2)


if __name__ == "__main__":
    unittest.main()

