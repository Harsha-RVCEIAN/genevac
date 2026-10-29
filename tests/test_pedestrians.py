import unittest
from genevac.pedestrians import Pedestrian, PedestrianStatus


class TestPedestrians(unittest.TestCase):
    def test_pedestrian_movement(self):
        ped = Pedestrian(id="p1", position=(2.0, 3.0), velocity=(1.0, 2.0))
        ped.update_position(dt=0.5)
        self.assertEqual(ped.position, (2.5, 4.0))

    def test_inactive_pedestrian_does_not_move(self):
        ped = Pedestrian(id="p1", position=(2.0, 3.0), velocity=(1.0, 2.0), status=PedestrianStatus.EVACUATED)
        ped.update_position(dt=0.5)
        self.assertEqual(ped.position, (2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
