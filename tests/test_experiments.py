import unittest
from genevac.experiments import ExperimentRunner


class TestExperiments(unittest.TestCase):
    def test_experiment_runner(self):
        scenarios = [
            {"name": "test_scenario", "width": 20.0, "height": 10.0, "max_steps": 10, "exits": []}
        ]
        runner = ExperimentRunner(scenarios)
        results = runner.run_all()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "test_scenario")
        self.assertEqual(results[0]["summary"]["total_steps"], 10)


if __name__ == "__main__":
    unittest.main()
