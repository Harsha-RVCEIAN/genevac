"""Run a single simulation scenario, optionally recording a dataset."""
import argparse
from genevac.environment import EnvironmentMap, Exit
from genevac.simulation import SimulationEngine
from genevac.pedestrians import Pedestrian
from genevac.prediction.dataset_pipeline import SimulationDatasetRecorder


def main():
    parser = argparse.ArgumentParser(description="Run Genevac single scenario simulation")
    parser.add_argument("--scenario", type=str, default="sample", help="Scenario identifier or file path")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    parser.add_argument("--record-dataset", action="store_true", help="Export this run as a synthetic trajectory dataset")
    parser.add_argument("--dataset-output", default="data/processed/evacuation")
    parser.add_argument("--run-id", default="sample_run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Starting simulation run for: {args.scenario}")
    env = EnvironmentMap(width=30.0, height=15.0)
    env.add_exit(Exit(id="main_exit", position=(30.0, 7.5), width=2.0))
    engine = SimulationEngine(environment=env, max_steps=args.steps, seed=args.seed)
    engine.add_pedestrian(Pedestrian(id="ped_1", position=(5.0, 5.0), velocity=(1.0, 0.5)))
    engine.add_pedestrian(Pedestrian(id="ped_2", position=(5.0, 10.0), velocity=(1.2, 0.0)))

    recorder = SimulationDatasetRecorder(observation_interval=0.1) if args.record_dataset else None
    while not engine.is_finished():
        engine.step()
        if recorder is not None:
            recorder.capture(engine)
    results = {
        "total_steps": engine.current_step,
        "pedestrians_total": len(engine.pedestrians),
        "evacuated": sum(1 for p in engine.pedestrians if getattr(p.status, "name", str(p.status)) == "EVACUATED"),
    }
    print(f"Simulation finished: {results}")
    if recorder is not None:
        manifest = recorder.export(
            args.dataset_output, scenario_id=args.scenario, seed=args.seed, run_id=args.run_id
        )
        print(f"Dataset exported: {args.dataset_output} ({manifest['window_counts']})")


if __name__ == "__main__":
    main()
