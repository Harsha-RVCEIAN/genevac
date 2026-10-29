"""Run batch experiments across multiple scenarios."""

from genevac.experiments import ExperimentRunner


def main():
    scenarios = [
        {"name": "baseline_hallway", "width": 40.0, "height": 10.0, "max_steps": 100, "exits": [{"id": "e1", "position": [40.0, 5.0]}]},
        {"name": "bottleneck_scenario", "width": 40.0, "height": 10.0, "max_steps": 100, "exits": [{"id": "e1", "position": [40.0, 5.0], "width": 1.0}]},
    ]
    print(f"Executing batch experiments for {len(scenarios)} scenarios...")
    runner = ExperimentRunner(scenarios)
    results = runner.run_all()
    for res in results:
        print(f"Scenario '{res['name']}': {res['summary']}")


if __name__ == "__main__":
    main()
