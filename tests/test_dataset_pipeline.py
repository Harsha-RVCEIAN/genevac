import tempfile, unittest
from pathlib import Path
from genevac.prediction.dataset_pipeline import inventory_datasets, load_eth_ucy, split_by_agent, make_windows, validate_trajectories
from genevac.prediction.data_contract import TrajectoryObservation

class DatasetPipelineTests(unittest.TestCase):
    def test_eth_timestamp_is_04_seconds_for_ten_frames(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"eth.txt"; p.write_text("0 1 0 0\n10 1 1 1\n20 1 2 2\n")
            tracks=load_eth_ucy(p)
            self.assertAlmostEqual(tracks["1"][1].timestamp,.4)
            self.assertAlmostEqual(tracks["1"][2].timestamp,.8)
    def test_inventory_classifies_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"eth"/"train"; p.mkdir(parents=True); (p/"biwi_eth_train.txt").write_text("0 1 0 0\n")
            row=inventory_datasets(d)[0]
            self.assertEqual(row["dataset"],"ETH"); self.assertEqual(row["split"],"train"); self.assertEqual(row["line_count"],1)
    def test_agent_splits_are_disjoint_and_windows_created(self):
        tracks={str(i):[TrajectoryObservation(agent_id=str(i),step_index=j,timestamp=j*.4,x=float(j),y=0.) for j in range(30)] for i in range(10)}
        s=split_by_agent(tracks,seed=9); groups=[set(s[k]) for k in ("train","validation","test")]
        self.assertFalse(groups[0]&groups[1]); self.assertFalse(groups[0]&groups[2]); self.assertFalse(groups[1]&groups[2])
        windows=make_windows(tracks,s,history_length=8,prediction_horizon=12)
        self.assertTrue(windows); self.assertTrue(validate_trajectories(tracks)["valid"])
if __name__=="__main__": unittest.main()
