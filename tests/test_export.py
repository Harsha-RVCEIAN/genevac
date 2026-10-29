import unittest
import json
import tempfile
from pathlib import Path
from genevac.export import export_to_json, export_to_csv


class TestExport(unittest.TestCase):
    def test_export_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_file = tmp_path / "out.json"
            csv_file = tmp_path / "out.csv"
            data = [{"step": 1, "agents": 5}]

            export_to_json(data, json_file)
            self.assertTrue(json_file.exists())
            with open(json_file, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)

            export_to_csv(data, csv_file)
            self.assertTrue(csv_file.exists())


if __name__ == "__main__":
    unittest.main()
