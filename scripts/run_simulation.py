"""Run a single simulation scenario."""

import argparse
from genevac.environment import EnvironmentMap, Exit
from genevac.simulation import SimulationEngine
from genevac.pedestrians import Pedestrian


def main():
    parser = argparse.ArgumentParser(description="Run Genevac single scenario simulation")
    parser.add_argument("--scenario", type=str, default="sample", help="Scenario identifier or file path")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    args = parser.parse_args()

    print(f"Starting simulation run for: {args.scenario}")
    env = EnvironmentMap(width=30.0, height=15.0)
    env.add_exit(Exit(id="main_exit", position=(30.0, 7.5), width=2.0))

    engine = SimulationEngine(environment=env, max_steps=args.steps)
    engine.add_pedestrian(Pedestrian(id="ped_1", position=(5.0, 5.0), velocity=(1.0, 0.5)))
    engine.add_pedestrian(Pedestrian(id="ped_2", position=(5.0, 10.0), velocity=(1.2, 0.0)))

    results = engine.run()
    print(f"Simulation finished: {results}")


if __name__ == "__main__":
    main()
