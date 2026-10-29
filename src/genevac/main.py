"""Application entry point."""

import argparse
import sys
from genevac.config import config
from genevac.simulation import SimulationEngine
from genevac.environment import EnvironmentMap


def main():
    parser = argparse.ArgumentParser(description="Genevac Evacuation Simulation CLI")
    parser.add_argument("--scenario", type=str, default="data/scenarios/sample_hallway.json",
                        help="Path to scenario JSON configuration")
    parser.add_argument("--steps", type=int, default=config.simulation.max_steps,
                        help="Max simulation steps to run")
    parser.add_argument("--seed", type=int, default=config.simulation.random_seed,
                        help="Random seed for simulation")
    args = parser.parse_args()

    print(f"=== Genevac Evacuation Simulation [Env: {config.environment}] ===")
    print(f"Loading scenario: {args.scenario}")
    
    # Initialize basic environment and engine
    env_map = EnvironmentMap(width=50.0, height=20.0)
    engine = SimulationEngine(environment=env_map, max_steps=args.steps, seed=args.seed)
    
    print(f"Running simulation for {args.steps} steps...")
    summary = engine.run()
    print("Simulation completed.")
    print(f"Results: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
