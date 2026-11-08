"""Person B Ingestion and Prediction Evaluation Demonstration.

Demonstrates how downstream prediction modules should:
1. Initialize the building and initial population cohort.
2. Pass the population into SimulationEngine during construction.
3. Advance the simulation and handle intermediate None returns.
4. Store predictions in separate downstream data structures.
5. Match predictions to future ground-truth snapshots by pedestrian_id.
6. Compute displacement error metrics (FDE).
"""

import math
from typing import NamedTuple

from genevac.environment.builder import create_initial_building
from genevac.pedestrian.models import PedestrianState, Point
from genevac.simulation.engine import SimulationEngine
from genevac.simulation.observation import (
    PedestrianObservation,
    SimulationObservationSnapshot,
)


class PredictedPoint(NamedTuple):
    step: int
    timestamp: float
    x: float
    y: float


def run_person_b_pipeline():
    # 1. Instantiate the authoritative building environment
    building = create_initial_building()

    # 2. Define the initial pedestrian cohort
    initial_cohort = [
        PedestrianState(
            id="agent_1",
            position=Point(8.0, 6.0),
            current_area_id="room_1",
            target_exit_id="exit_left",
            preferred_speed=1.3,
            radius=0.25,
        ),
        PedestrianState(
            id="agent_2",
            position=Point(28.0, 6.0),
            current_area_id="room_2",
            target_exit_id="exit_right",
            preferred_speed=1.2,
            radius=0.25,
        ),
    ]

    # 3. Pass population into SimulationEngine during construction (enables validation & routing)
    engine = SimulationEngine(
        building=building,
        registry=initial_cohort,
        dt=0.1,
        observation_interval=0.4,
    )

    # 4. Storage for Person B's separate predictions: {ped_id: list[PredictedPoint]}
    stored_predictions: dict[str, list[PredictedPoint]] = {}

    # 5. Warm up simulation for 10 steps (1.0 second)
    engine.run_steps(10)
    current_obs = engine.get_snapshot()

    # 6. Generate a 10-step horizon forecast (separate from simulation state)
    horizon_steps = 10
    forecast_dt = 0.1
    for ped in current_obs.pedestrians:
        if ped.status == "active":
            preds = []
            for h in range(1, horizon_steps + 1):
                future_step = current_obs.step_count + h
                future_time = current_obs.timestamp + h * forecast_dt
                # Linear constant-velocity baseline forecast
                px = ped.x + ped.vx * (h * forecast_dt)
                py = ped.y + ped.vy * (h * forecast_dt)
                preds.append(PredictedPoint(future_step, future_time, px, py))
            stored_predictions[ped.pedestrian_id] = preds

    # 7. Advance authoritative simulation to reach ground truth horizon
    engine.run_steps(horizon_steps)
    future_ground_truth = engine.get_snapshot()

    # 8. Evaluate predictions against actual future observations using matching IDs
    print(f"--- Evaluation at Step {future_ground_truth.step_count} (t={future_ground_truth.timestamp:.2f}s) ---")
    for actual_ped in future_ground_truth.pedestrians:
        if actual_ped.pedestrian_id in stored_predictions:
            final_predicted = stored_predictions[actual_ped.pedestrian_id][-1]
            fde = math.hypot(final_predicted.x - actual_ped.x, final_predicted.y - actual_ped.y)
            print(f"Pedestrian {actual_ped.pedestrian_id}:")
            print(f"  Predicted final pos: ({final_predicted.x:.3f}, {final_predicted.y:.3f})")
            print(f"  Actual ground truth: ({actual_ped.x:.3f}, {actual_ped.y:.3f})")
            print(f"  Status: {actual_ped.status}")
            print(f"  Final Displacement Error (FDE): {fde:.3f} metres")


if __name__ == "__main__":
    run_person_b_pipeline()

