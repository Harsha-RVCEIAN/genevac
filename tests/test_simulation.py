import unittest
from genevac.simulation import SimulationEngine
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian


class TestSimulation(unittest.TestCase):
    def test_simulation_step(self):
        env = EnvironmentMap(width=20.0, height=20.0)
        engine = SimulationEngine(environment=env, max_steps=5, timestep=1.0)
        ped = Pedestrian(id="p1", position=(0.0, 0.0), velocity=(1.0, 0.0))
        engine.add_pedestrian(ped)

        engine.step()
        self.assertEqual(ped.position, (1.0, 0.0))
        self.assertEqual(engine.current_step, 1)


if __name__ == "__main__":
    unittest.main()
