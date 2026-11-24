"""Real-world pedestrian trajectory ingestion and window preparation.

Supported sources:
- ETH/UCY: ETH, HOTEL, UNIV, ZARA1, ZARA2 text trajectories.
- SDD: a normalized CSV/TSV table whose x/y columns have already been
  transformed from image pixels into world coordinates in metres.

This module intentionally does not generate or ingest synthetic trajectories.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from genevac.prediction.data_contract import TrajectoryCategory, TrajectoryObservation

SUPPORTED_DATASETS = ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2", "SDD")
TABULAR_SUFFIXES = {".txt", ".csv", ".tsv"}


@dataclass(frozen=True)
class DatasetWindow:
    dataset: str
    scene: str
    agent_id: str
    split: str
    reference_timestamp: float
    history: tuple
    future: tuple


def identify_dataset(path: str | Path) -> str:
    """Infer ETH/UCY subset or SDD from explicit path/name markers."""
    p = Path(path)
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    if any(token in parts or token in name for token in ("stanford", "sdd")):
        return "SDD"
    if "hotel" in parts or "hotel" in name or "biwi_hotel" in name:
        return "HOTEL"
    if "univ" in parts or "univ" in name or "students" in name or "uni_examples" in name:
        return "UNIV"
    if "zara1" in parts or "zara1" in name or "zara01" in name:
        return "ZARA1"
    if "zara2" in parts or "zara2" in name or "zara02" in name:
        return "ZARA2"
    if "eth" in parts or "biwi_eth" in name:
        return "ETH"
    return "UNKNOWN"


def inventory_datasets(root: str | Path) -> list[dict[str, Any]]:
    """List candidate real-world trajectory files; does not load or alter them."""
    base = Path(root)
    if not base.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_SUFFIXES):
        parts = {part.lower() for part in path.parts}
        split = next((s for s in ("train", "validation", "val", "test") if s in parts), "unspecified")
        if split == "val":
            split = "validation"
        row: dict[str, Any] = {
            "path": path.as_posix(),
            "dataset": identify_dataset(path),
            "split": split,
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() in {".txt", ".csv", ".tsv"}:
            with path.open(encoding="utf-8", errors="replace") as handle:
                row["nonempty_lines"] = sum(bool(line.strip()) for line in handle)
        rows.append(row)
    return rows


def _observation_tracks(records: Mapping[str, list[tuple[float, float, float]]]):
    """Convert sorted (time, x, y) records into observed trajectory contracts."""
    tracks: dict[str, list[TrajectoryObservation]] = {}
    for agent_id, points in records.items():
        points.sort(key=lambda item: item[0])
        if len(points) < 2:
            continue
        observations: list[TrajectoryObservation] = []
        previous_time = None
        for index, (timestamp, x, y) in enumerate(points):
            if not all(math.isfinite(float(value)) for value in (timestamp, x, y)):
                raise ValueError(f"non-finite value in trajectory for agent {agent_id}")
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError(f"duplicate or non-increasing timestamp for agent {agent_id}")
            observations.append(
                TrajectoryObservation(
                    agent_id=str(agent_id),
                    step_index=index,
                    timestamp=float(timestamp),
                    x=float(x),
                    y=float(y),
                    category=TrajectoryCategory.OBSERVED,
                )
            )
            previous_time = timestamp
        tracks[str(agent_id)] = observations
    return tracks


def load_eth_ucy(
    filepath: str | Path,
    *,
    seconds_per_source_frame: float = 0.04,
    sampling_stride: int = 1,
) -> dict[str, list[TrajectoryObservation]]:
    """Load ETH/UCY rows formatted as frame_id, pedestrian_id, x, y.

    Timestamps are derived from source-frame indices. For a 25 FPS source,
    10-frame annotation gaps therefore become 0.4 seconds.
    """
    if not math.isfinite(seconds_per_source_frame) or seconds_per_source_frame <= 0:
        raise ValueError("seconds_per_source_frame must be a finite positive number")
    if type(sampling_stride) is not int or sampling_stride < 1:
        raise ValueError("sampling_stride must be a positive integer")
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(path)

    raw: dict[str, list[tuple[int, float, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.replace(",", " ").split()
            if not fields:
                continue
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: expected frame_id pedestrian_id x y")
            try:
                frame = int(float(fields[0]))
                agent_id = str(int(float(fields[1])))
                x, y = float(fields[2]), float(fields[3])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric trajectory field") from error
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"{path}:{line_number}: non-finite coordinate")
            raw.setdefault(agent_id, []).append((frame, x, y))

    records: dict[str, list[tuple[float, float, float]]] = {}
    for agent_id, points in raw.items():
        points.sort(key=lambda item: item[0])
        frames = [point[0] for point in points]
        if len(frames) != len(set(frames)):
            raise ValueError(f"{path}: duplicate frame for agent {agent_id}")
        points = points[::sampling_stride]
        if len(points) >= 2:
            first_frame = points[0][0]
            records[agent_id] = [
                ((frame - first_frame) * seconds_per_source_frame, x, y)
                for frame, x, y in points
            ]
    return _observation_tracks(records)


def load_sdd_table(
    filepath: str | Path,
    *,
    seconds_per_frame: float | None = None,
    coordinates_in_metres: bool = False,
) -> dict[str, list[TrajectoryObservation]]:
    """Load a normalized SDD CSV/TSV table.

    Required columns: track_id (or agent_id/ped_id), frame (or timestamp),
    x, y. x/y must already be transformed into world coordinates in metres.
    Raw SDD pixel coordinates are deliberately rejected: apply the scene's
    homography before creating this normalized table.
    """
    if not coordinates_in_metres:
        raise ValueError(
            "SDD x/y must be homography-transformed world coordinates in metres; "
            "pass coordinates_in_metres=True only after verifying that conversion"
        )
    if seconds_per_frame is not None and (
        not math.isfinite(seconds_per_frame) or seconds_per_frame <= 0
    ):
        raise ValueError("seconds_per_frame must be finite and positive")

    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("SDD table must have a header row")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        def choose(*names: str):
            return next((columns[name] for name in names if name in columns), None)
        id_col = choose("track_id", "agent_id", "ped_id", "trackid")
        frame_col = choose("frame", "frame_id", "frameid")
        time_col = choose("timestamp", "time", "time_seconds")
        x_col, y_col = choose("x", "x_world"), choose("y", "y_world")
        if not id_col or not (frame_col or time_col) or not x_col or not y_col:
            raise ValueError(
                "SDD normalized table requires track_id, (frame or timestamp), x, y columns"
            )
        if frame_col and not time_col and seconds_per_frame is None:
            raise ValueError("provide seconds_per_frame when SDD table uses frame indices")

        records: dict[str, list[tuple[float, float, float]]] = {}
        for line_number, row in enumerate(reader, 2):
            try:
                agent_id = str(row[id_col]).strip()
                if not agent_id:
                    raise ValueError("empty track id")
                if time_col:
                    timestamp = float(row[time_col])
                else:
                    timestamp = float(row[frame_col]) * float(seconds_per_frame)
                x, y = float(row[x_col]), float(row[y_col])
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: invalid SDD row: {error}") from error
            records.setdefault(agent_id, []).append((timestamp, x, y))
    return _observation_tracks(records)


def validate_trajectories(
    tracks: Mapping[str, Sequence[TrajectoryObservation]],
) -> dict[str, Any]:
    errors: list[str] = []
    observations = 0
    for agent_id, sequence in tracks.items():
        previous = None
        for observation in sequence:
            observations += 1
            if observation.agent_id != agent_id:
                errors.append(f"{agent_id}: agent id mismatch")
            if not all(
                math.isfinite(float(value))
                for value in (observation.timestamp, observation.x, observation.y)
            ):
                errors.append(f"{agent_id}: non-finite timestamp or coordinate")
            if previous is not None and (
                observation.timestamp <= previous.timestamp
                or observation.step_index <= previous.step_index
            ):
                errors.append(f"{agent_id}: non-increasing timestamp or step index")
            previous = observation
    return {
        "valid": not errors,
        "agents": len(tracks),
        "observations": observations,
        "errors": errors,
    }


def split_by_agent(
    tracks: Mapping[str, Sequence[Any]],
    *,
    seed: int = 42,
    train: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
) -> dict[str, list[str]]:
    ratios = (train, validation, test)
    if any(value < 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8):
        raise ValueError("train/validation/test ratios must be non-negative and sum to 1")
    agent_ids = sorted(agent_id for agent_id, sequence in tracks.items() if len(sequence) >= 2)
    random.Random(seed).shuffle(agent_ids)
    train_end = int(len(agent_ids) * train)
    validation_end = train_end + int(len(agent_ids) * validation)
    return {
        "train": agent_ids[:train_end],
        "validation": agent_ids[train_end:validation_end],
        "test": agent_ids[validation_end:],
    }


def make_windows(
    tracks: Mapping[str, Sequence[TrajectoryObservation]],
    splits: Mapping[str, Sequence[str]],
    *,
    history_length: int = 8,
    prediction_horizon: int = 12,
    stride: int = 1,
    dataset: str = "unknown",
    scene: str = "unknown",
) -> list[DatasetWindow]:
    if min(history_length, prediction_horizon, stride) < 1:
        raise ValueError("history_length, prediction_horizon and stride must be positive")
    windows: list[DatasetWindow] = []
    total_length = history_length + prediction_horizon
    for split_name, agent_ids in splits.items():
        for agent_id in agent_ids:
            sequence = tracks[agent_id]
            for start in range(0, len(sequence) - total_length + 1, stride):
                history = sequence[start : start + history_length]
                future = sequence[start + history_length : start + total_length]
                windows.append(
                    DatasetWindow(
                        dataset=dataset,
                        scene=scene,
                        agent_id=agent_id,
                        split=split_name,
                        reference_timestamp=float(history[-1].timestamp),
                        history=tuple((o.timestamp, o.x, o.y) for o in history),
                        future=tuple((o.timestamp, o.x, o.y) for o in future),
                    )
                )
    return windows


def export_dataset(
    tracks: Mapping[str, Sequence[TrajectoryObservation]],
    output_dir: str | Path,
    *,
    dataset: str,
    scene: str,
    source: str | Path,
    seed: int = 42,
    history_length: int = 8,
    prediction_horizon: int = 12,
    fixed_split: str | None = None,
) -> dict[str, Any]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}")
    validation = validate_trajectories(tracks)
    if not validation["valid"]:
        raise ValueError("invalid trajectories: " + "; ".join(validation["errors"][:5]))
    if fixed_split is not None:
        if fixed_split not in {"train", "validation", "test"}:
            raise ValueError("fixed_split must be train, validation or test")
        splits = {
            "train": sorted(tracks) if fixed_split == "train" else [],
            "validation": sorted(tracks) if fixed_split == "validation" else [],
            "test": sorted(tracks) if fixed_split == "test" else [],
        }
        split_strategy = "preserved_source_partition"
    else:
        splits = split_by_agent(tracks, seed=seed)
        split_strategy = "agent_disjoint_within_scene"

    windows = make_windows(
        tracks,
        splits,
        history_length=history_length,
        prediction_horizon=prediction_horizon,
        dataset=dataset,
        scene=scene,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        destination = output / f"{dataset.lower()}_{scene.lower()}_{split_name}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for window in windows:
                if window.split == split_name:
                    handle.write(json.dumps(asdict(window), allow_nan=False) + "\n")

    manifest = {
        "schema_version": "2.0",
        "data_origin": "real_world_observed",
        "dataset": dataset,
        "scene": scene,
        "source_file": str(source),
        "coordinate_units": "metres",
        "time_units": "seconds",
        "trajectory_category": TrajectoryCategory.OBSERVED.value,
        "split_strategy": split_strategy,
        "source_split": fixed_split,
        "random_seed": seed if fixed_split is None else None,
        "history_length_observations": history_length,
        "prediction_horizon_observations": prediction_horizon,
        "validation": validation,
        "split_agents": splits,
        "window_counts": {
            name: sum(window.split == name for window in windows)
            for name in ("train", "validation", "test")
        },
        "important_note": (
            "ETH/UCY/SDD are real-world pedestrian-motion datasets, not emergency-evacuation "
            "ground truth. Keep benchmark metrics separate from GenEvac evacuation metrics."
        ),
    }
    (output / f"{dataset.lower()}_{scene.lower()}_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest
