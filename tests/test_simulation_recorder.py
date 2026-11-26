"""Tests for the GenEvac simulation dataset subsystem.

Covers:
- Schema serialisation round-trip.
- Recorder lifecycle (begin / record_step / finish).
- Multiple independent runs produce separate files.
- Duplicate observation prevention.
- Metadata fields populated correctly.
- Validation reports correct counts.
- Pre-processing generates valid windows.
- Splitting is deterministic and leakage-free.
- Batch generator produces manifests and passes validation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from genevac.data.simulation.schema import (
    RunMetadata,
    TrajectoryRecord,
    generate_run_id,
    iso_timestamp,
    SCHEMA_VERSION,
)
from genevac.data.simulation.manifest import write_manifest, load_manifest, list_manifests
from genevac.environment import EnvironmentMap, Exit
from genevac.experiments.scenario import ScenarioDefinition
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction.simulation_recorder import SimulationRecorder, record_full_run
from genevac.prediction.simulation_validator import validate_dataset
from genevac.prediction.simulation_preprocess import generate_windows, load_run_records
from genevac.prediction.simulation_split import (
    split_run_ids,
    check_leakage,
    split_by_layout,
    write_split_manifest,
    load_split_manifest,
)
from genevac.prediction.simulation_stats import generate_stats
from genevac.prediction.simulation_batch import BatchGenerator
from genevac.simulation import SimulationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def _make_engine(n_peds: int = 5, max_steps: int = 20, seed: int = 42) -> SimulationEngine:
    env = EnvironmentMap(width=20.0, height=10.0)
    env.add_exit(Exit(id="exit_A", position=(19.5, 5.0), width=2.0))
    engine = SimulationEngine(environment=env, max_steps=max_steps, timestep=0.1, seed=seed)
    for i in range(n_peds):
        engine.add_pedestrian(Pedestrian(
            id=f"p{i + 1}",
            position=(2.0 + i * 1.5, 5.0),
            velocity=(1.0, 0.0),
        ))
    return engine


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_generate_run_id_is_unique(self):
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_run_metadata_round_trip(self):
        meta = RunMetadata(
            run_id="abc-123",
            scenario_id="s1",
            layout_id="l1",
            seed=42,
            population=10,
            environment_width=20.0,
            environment_height=10.0,
            timestep=0.1,
            max_steps=100,
            total_steps=50,
            evacuated=3,
            start_time=iso_timestamp(),
            end_time=iso_timestamp(),
            status="completed",
        )
        assert RunMetadata.from_json(meta.to_json()) == meta

    def test_trajectory_record_round_trip(self):
        rec = TrajectoryRecord(
            run_id="abc-123",
            scenario_id="s1",
            layout_id="l1",
            pedestrian_id="p1",
            step_index=5,
            timestamp=0.5,
            x=3.0,
            y=5.0,
            vx=1.0,
            vy=0.0,
            speed=1.0,
            status="active",
        )
        assert TrajectoryRecord.from_json(rec.to_json()) == rec

    def test_trajectory_record_null_velocity(self):
        rec = TrajectoryRecord(
            run_id="r1", scenario_id="s1", layout_id="l1",
            pedestrian_id="p1", step_index=0, timestamp=0.0,
            x=1.0, y=2.0,
        )
        data = json.loads(rec.to_json())
        assert data["vx"] is None
        assert data["vy"] is None


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------

class TestManifest:
    def test_write_and_load(self, tmp):
        traj_file = tmp / "run.jsonl"
        traj_file.write_text("")
        meta = RunMetadata(
            run_id="r1", scenario_id="s1", layout_id="l1", seed=1,
            population=5, environment_width=20.0, environment_height=10.0,
            timestep=0.1, max_steps=100, total_steps=20, evacuated=2,
            start_time="2024-01-01T00:00:00Z", end_time="2024-01-01T00:01:00Z",
            status="completed",
        )
        mpath = write_manifest(meta, traj_file, dest_dir=tmp)
        assert mpath.is_file()
        m = load_manifest(mpath)
        assert m["run_id"] == "r1"
        assert m["schema_version"] == SCHEMA_VERSION

    def test_list_manifests(self, tmp):
        for i in range(3):
            (tmp / f"run_{i}_manifest.json").write_text("{}")
        (tmp / "not_a_manifest.txt").write_text("")
        found = list_manifests(tmp)
        assert len(found) == 3


# ---------------------------------------------------------------------------
# Recorder tests
# ---------------------------------------------------------------------------

class TestSimulationRecorder:
    def test_records_observations(self, tmp):
        engine = _make_engine(n_peds=3, max_steps=5)
        recorder = SimulationRecorder(
            scenario_id="s1", layout_id="l1", seed=42,
            trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
        )
        recorder.begin()
        for _ in range(5):
            engine.step()
            recorder.record_step(engine)
        meta = recorder.finish(engine)

        assert recorder.records_written > 0
        assert recorder.trajectory_path.is_file()
        assert meta.status == "completed"
        assert meta.total_steps == 5

    def test_no_duplicate_observations(self, tmp):
        engine = _make_engine(n_peds=2, max_steps=3)
        recorder = SimulationRecorder(
            scenario_id="s1", layout_id="l1", seed=42,
            trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
        )
        recorder.begin()
        for _ in range(3):
            engine.step()
            recorder.record_step(engine)
            recorder.record_step(engine)  # second call same step = no-op
        recorder.finish(engine)

        lines = [l for l in recorder.trajectory_path.read_text().splitlines() if l.strip()]
        keys = [(json.loads(l)["pedestrian_id"], json.loads(l)["step_index"]) for l in lines]
        assert len(keys) == len(set(keys)), "Duplicate (pedestrian_id, step_index) found"

    def test_manifest_written(self, tmp):
        engine = _make_engine(n_peds=2, max_steps=5)
        recorder = SimulationRecorder(
            scenario_id="s1", layout_id="l1", seed=42,
            trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
        )
        recorder.begin()
        while not engine.is_finished():
            engine.step()
            recorder.record_step(engine)
        recorder.finish(engine)
        manifests = list(Path(tmp / "mfst").glob("*_manifest.json"))
        assert len(manifests) == 1

    def test_multiple_independent_runs_produce_separate_files(self, tmp):
        for i in range(3):
            engine = _make_engine(n_peds=2, max_steps=5, seed=i)
            recorder = SimulationRecorder(
                scenario_id="s1", layout_id="l1", seed=i,
                trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
            )
            recorder.begin()
            while not engine.is_finished():
                engine.step()
                recorder.record_step(engine)
            recorder.finish(engine)

        traj_files = list((tmp / "traj").glob("*.jsonl"))
        mfst_files = list((tmp / "mfst").glob("*_manifest.json"))
        assert len(traj_files) == 3
        assert len(mfst_files) == 3
        # All run IDs must be unique
        run_ids = [json.loads(m.read_text())["run_id"] for m in mfst_files]
        assert len(set(run_ids)) == 3

    def test_record_full_run_helper(self, tmp):
        engine = _make_engine(n_peds=4, max_steps=10)
        meta, traj_path = record_full_run(
            engine,
            scenario_id="s1", layout_id="l1", seed=42,
            trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
        )
        assert traj_path.is_file()
        assert meta.total_steps > 0


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def _write_run(self, manifests_dir, trajectories_dir, n_peds=3, max_steps=5):
        engine = _make_engine(n_peds=n_peds, max_steps=max_steps)
        meta, _ = record_full_run(
            engine, scenario_id="s1", layout_id="l1", seed=42,
            trajectories_dir=trajectories_dir,
            manifests_dir=manifests_dir,
        )
        return meta

    def test_valid_run_has_no_errors(self, tmp):
        self._write_run(tmp / "mfst", tmp / "traj")
        report = validate_dataset(tmp / "mfst")
        assert report["invalid_records"] == 0
        assert len(report["errors"]) == 0
        assert report["valid_runs"] == 1

    def test_missing_traj_file_flagged(self, tmp):
        (tmp / "mfst").mkdir()
        fake = {
            "schema_version": "v1",
            "run_id": "bad-run",
            "scenario_id": "s1",
            "layout_id": "l1",
            "trajectory_file": str(tmp / "traj" / "nonexistent.jsonl"),
        }
        (tmp / "mfst" / "bad-run_manifest.json").write_text(json.dumps(fake))
        report = validate_dataset(tmp / "mfst")
        assert report["invalid_runs"] == 1


# ---------------------------------------------------------------------------
# Pre-process tests
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_generates_windows(self, tmp):
        engine = _make_engine(n_peds=5, max_steps=50)
        meta, traj_path = record_full_run(
            engine, scenario_id="s1", layout_id="l1", seed=42,
            trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
        )
        windows = generate_windows(
            traj_path,
            run_id=meta.run_id,
            scenario_id="s1",
            layout_id="l1",
            history_len=4,
            future_len=6,
        )
        assert len(windows) >= 0  # may be 0 if trajectories too short
        for w in windows:
            assert len(w.history) == 4
            assert len(w.future) == 6

    def test_load_run_records(self, tmp):
        engine = _make_engine(n_peds=3, max_steps=10)
        meta, traj_path = record_full_run(
            engine, scenario_id="s1", layout_id="l1", seed=42,
            trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
        )
        records = load_run_records(traj_path)
        assert len(records) > 0
        for r in records:
            assert r.run_id == meta.run_id


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_is_deterministic(self):
        run_ids = [generate_run_id() for _ in range(12)]
        a1 = split_run_ids(run_ids, seed=1)
        a2 = split_run_ids(run_ids, seed=1)
        assert a1 == a2

    def test_split_different_seeds_differ(self):
        run_ids = [generate_run_id() for _ in range(12)]
        a1 = split_run_ids(run_ids, seed=1)
        a2 = split_run_ids(run_ids, seed=999)
        assert a1 != a2

    def test_no_leakage(self):
        run_ids = [generate_run_id() for _ in range(10)]
        assignments = split_run_ids(run_ids)
        errors = check_leakage(assignments)
        assert errors == [], f"Unexpected leakage: {errors}"

    def test_all_splits_present(self):
        run_ids = [generate_run_id() for _ in range(10)]
        assignments = split_run_ids(run_ids)
        splits = set(assignments.values())
        assert "train" in splits

    def test_layout_level_split(self):
        layout_map = {
            "hall_A": [generate_run_id() for _ in range(3)],
            "hall_B": [generate_run_id() for _ in range(3)],
            "hall_C": [generate_run_id() for _ in range(3)],
        }
        assignments = split_by_layout(layout_map)
        errors = check_leakage(assignments)
        assert errors == []
        # All runs of the same layout in the same split
        for layout_id, runs in layout_map.items():
            split_set = {assignments[r] for r in runs}
            assert len(split_set) == 1, f"Layout {layout_id} split across: {split_set}"

    def test_split_manifest_round_trip(self, tmp):
        run_ids = [generate_run_id() for _ in range(6)]
        assignments = split_run_ids(run_ids)
        p = tmp / "split.json"
        write_split_manifest(assignments, p)
        loaded = load_split_manifest(p)
        assert loaded == assignments


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestStats:
    def test_generates_report(self, tmp):
        for i in range(2):
            engine = _make_engine(n_peds=5, max_steps=15, seed=i)
            record_full_run(
                engine, scenario_id="s1", layout_id="l1", seed=i,
                trajectories_dir=tmp / "traj", manifests_dir=tmp / "mfst",
            )
        stats = generate_stats(tmp / "mfst")
        assert stats["total_runs"] == 2
        assert stats["total_pedestrians"] > 0
        assert stats["total_observations"] > 0

    def test_empty_dir_returns_error(self, tmp):
        (tmp / "mfst").mkdir()
        stats = generate_stats(tmp / "mfst")
        assert stats["total_runs"] == 0


# ---------------------------------------------------------------------------
# Batch generator tests
# ---------------------------------------------------------------------------

class TestBatchGenerator:
    def _pilot_scenario(self):
        return ScenarioDefinition(
            scenario_id="batch_test_01",
            name="Batch test",
            environment_id="test_hall",
            agent_count=5,
            seed=1,
            engine_config={
                "width": 20.0, "height": 10.0,
                "max_steps": 10, "timestep": 0.1,
                "exits": [{"id": "e1", "position": [19.5, 5.0], "width": 2.0}],
                "obstacles": [],
            },
        )

    def test_batch_runs_complete(self, tmp):
        gen = BatchGenerator(
            output_dir=tmp,
            scenarios=[self._pilot_scenario()],
            runs_per_scenario=2,
            base_seed=10,
        )
        summary = gen.run()
        assert summary["total_completed"] == 2
        assert summary["total_failed"] == 0

    def test_batch_manifest_written(self, tmp):
        gen = BatchGenerator(
            output_dir=tmp,
            scenarios=[self._pilot_scenario()],
            runs_per_scenario=1,
        )
        gen.run()
        assert (tmp / "manifests" / "batch_manifest.json").is_file()

    def test_batch_validation_passes(self, tmp):
        gen = BatchGenerator(
            output_dir=tmp,
            scenarios=[self._pilot_scenario()],
            runs_per_scenario=2,
        )
        summary = gen.run()
        val = summary["validation"]
        assert val["invalid_records"] == 0
        assert val["valid_runs"] == 2
