"""Dataset and simulation adapters for the GenEvac prediction evaluation subsystem.

Loads trajectory observations from:
1. ETH/UCY raw benchmark files (e.g. data/raw/eth/test/biwi_eth.txt)
2. GenEvac sample files (data/sample/sample_trajectories.json)
3. Running SimulationEngine instances
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from genevac.prediction.data_contract import TrajectoryCategory, TrajectoryObservation
from genevac.prediction.adapter import SimulationTrajectoryAdapter


class DatasetAdapter:
    """Utility to load external datasets and sample files into TrajectoryObservation sequences."""

    @staticmethod
    def load_eth_file(
        filepath: str | Path,
        sampling_stride: int = 1,
        time_per_frame: float = 0.4,
    ) -> dict[str, list[TrajectoryObservation]]:
        """Load ETH/UCY space/tab-separated dataset file.

        Format per line:
            frame_id   ped_id   x_coord   y_coord

        Args:
            filepath: Path to the dataset text file.
            sampling_stride: Subsample frame interval (default 1).
            time_per_frame: Time interval per recorded frame in seconds (default 0.4s for ETH benchmark).

        Returns:
            Dictionary mapping agent_id -> sorted list of TrajectoryObservation instances.
        """
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        raw_records: dict[str, list[tuple[int, float, float]]] = {}

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    frame_idx = int(float(parts[0]))
                    ped_id = str(int(float(parts[1])))
                    x_val = float(parts[2])
                    y_val = float(parts[3])
                except (ValueError, IndexError):
                    continue

                raw_records.setdefault(ped_id, []).append((frame_idx, x_val, y_val))

        trajectories: dict[str, list[TrajectoryObservation]] = {}

        for ped_id, points in raw_records.items():
            # Sort by frame index
            points.sort(key=lambda pt: pt[0])

            # Apply sampling stride
            subsampled = points[::sampling_stride]
            if len(subsampled) < 2:
                continue

            obs_list: list[TrajectoryObservation] = []
            base_frame = subsampled[0][0]

            for idx, (f_idx, x, y) in enumerate(subsampled):
                # Calculate relative timestamp based on frame count
                ts = (f_idx - base_frame) * time_per_frame
                obs = TrajectoryObservation(
                    agent_id=ped_id,
                    step_index=idx,
                    timestamp=round(ts, 4),
                    x=x,
                    y=y,
                    category=TrajectoryCategory.OBSERVED,
                )
                obs_list.append(obs)

            trajectories[ped_id] = obs_list

        return trajectories

    @staticmethod
    def load_sample_json(filepath: str | Path) -> dict[str, list[TrajectoryObservation]]:
        """Load GenEvac sample_trajectories.json file into TrajectoryObservation sequences."""
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"Sample JSON file not found: {filepath}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_obs = data.get("observations", [])
        trajectories: dict[str, list[TrajectoryObservation]] = {}

        for item in raw_obs:
            obs = TrajectoryObservation(
                agent_id=item["agent_id"],
                step_index=item["step_index"],
                timestamp=item["timestamp"],
                x=item["x"],
                y=item["y"],
                category=TrajectoryCategory.SIMULATED,
            )
            trajectories.setdefault(obs.agent_id, []).append(obs)

        for aid in trajectories:
            trajectories[aid].sort(key=lambda o: o.step_index)

        return trajectories


class SimulationEvaluationAdapter:
    """Records a complete simulation run for post-hoc prediction evaluation."""

    def __init__(self, observation_interval: float = 0.4) -> None:
        self.adapter = SimulationTrajectoryAdapter(
            observation_interval=observation_interval,
            max_observation_age=None,
        )
        self.cohort_map: dict[str, str] = {}

    def step(self, engine: Any) -> None:
        """Step the simulation engine and update observations."""
        self.adapter.update_from_simulation(engine)
        for aid in self.adapter.active_agents:
            self.cohort_map[aid] = "active"
        for aid in self.adapter.blocked_agents:
            self.cohort_map[aid] = "blocked"
        for aid in self.adapter.evacuated_agents:
            self.cohort_map[aid] = "evacuated"

    def get_trajectories(self) -> dict[str, tuple[TrajectoryObservation, ...]]:
        """Extract full observation histories for all recorded agents."""
        agent_ids = self.adapter.buffer.get_agent_ids()
        return {aid: self.adapter.buffer.get_observations(aid) for aid in agent_ids}

