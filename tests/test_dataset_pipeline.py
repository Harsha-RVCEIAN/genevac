import tempfile
import unittest
from pathlib import Path

from genevac.prediction.data_contract import TrajectoryObservation
from genevac.prediction.dataset_pipeline import (
    export_dataset,
    identify_dataset,
    inventory_datasets,
    load_eth_ucy,
    load_sdd_table,
    make_windows,
    split_by_agent,
    validate_trajectories,
)


class RealWorldDatasetPipelineTests(unittest.TestCase):
    def test_eth_ucy_ten_frame_gap_is_point_four_seconds_at_25_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "biwi_eth.txt"
            path.write_text("0 1 0.0 0.0\n10 1 1.0 1.0\n20 1 2.0 2.0\n")
            tracks = load_eth_ucy(path)
            self.assertAlmostEqual(tracks["1"][1].timestamp, 0.4)
            self.assertAlmostEqual(tracks["1"][2].timestamp, 0.8)
            self.assertEqual(tracks["1"][0].category.value, "observed")

    def test_inventory_identifies_eth_ucy_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eth" / "train"
            path.mkdir(parents=True)
            (path / "biwi_eth_train.txt").write_text("0 1 0 0\n")
            rows = inventory_datasets(directory)
            self.assertEqual(rows[0]["dataset"], "ETH")
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(identify_dataset("datasets/zara1/test.txt"), "ZARA1")

    def test_sdd_requires_explicit_world_metre_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sdd_tracks.csv"
            path.write_text("track_id,frame,x,y\nA,0,1.0,2.0\nA,10,2.0,3.0\n")
            with self.assertRaises(ValueError):
                load_sdd_table(path, seconds_per_frame=0.0333)
            tracks = load_sdd_table(
                path, seconds_per_frame=0.0333, coordinates_in_metres=True
            )
            self.assertAlmostEqual(tracks["A"][1].timestamp, 0.333)

    def test_agent_split_is_disjoint_and_windows_are_created(self):
        tracks = {
            str(agent): [
                TrajectoryObservation(
                    agent_id=str(agent), step_index=step, timestamp=step * 0.4,
                    x=float(step), y=0.0, category="observed"
                )
                for step in range(30)
            ]
            for agent in range(10)
        }
        splits = split_by_agent(tracks, seed=9)
        groups = [set(splits[name]) for name in ("train", "validation", "test")]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        windows = make_windows(tracks, splits, history_length=8, prediction_horizon=12)
        self.assertTrue(windows)
        self.assertTrue(validate_trajectories(tracks)["valid"])

    def test_export_marks_data_as_real_world_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            tracks = {
                "agent": [
                    TrajectoryObservation(
                        agent_id="agent", step_index=i, timestamp=i * 0.4,
                        x=float(i), y=0.0, category="observed"
                    )
                    for i in range(25)
                ]
            }
            manifest = export_dataset(
                tracks, directory, dataset="ETH", scene="eth_test",
                source="biwi_eth.txt", fixed_split="test"
            )
            self.assertEqual(manifest["data_origin"], "real_world_observed")
            self.assertEqual(manifest["window_counts"]["train"], 0)
            self.assertGreater(manifest["window_counts"]["test"], 0)


if __name__ == "__main__":
    unittest.main()
