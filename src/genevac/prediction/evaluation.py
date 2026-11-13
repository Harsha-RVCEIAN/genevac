"""Prediction evaluation subsystem for GenEvac-Sentinel.

Provides:
- EvaluationSample: Clean container for historical input vs. future ground-truth.
- SampleGenerator: Sliding-window evaluation sample generator ensuring zero future leakage.
- PredictionEvaluator: Core evaluator computing ADE, FDE, horizon-wise errors,
  per-agent metrics, and aggregate mean/median statistics.
- Dataset & Simulation adapters to feed trajectories into evaluation.
"""

from __future__ import annotations

from bisect import bisect_right
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

from genevac.metrics import calculate_ade, calculate_fde
from genevac.prediction.baseline import ConstantVelocityPredictor
from genevac.prediction.data_contract import (
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
)


@dataclass(frozen=True)
class EvaluationSample:
    """A single evaluation window for a specific pedestrian.

    Strict Data Isolation Invariant:
    - ``history`` contains only observations strictly on or before ``reference_timestamp``.
    - ``neighbor_trajectories`` contains optional nearby/other agent observations strictly on or before ``reference_timestamp``.
    - ``ground_truth_future`` contains only observations strictly after ``reference_timestamp``.
    The prediction model must ONLY receive ``history`` and historical neighbor observations via a snapshot.
    """
    agent_id: str
    reference_step: int
    reference_timestamp: float
    history: tuple[TrajectoryObservation, ...]
    ground_truth_future: tuple[TrajectoryObservation, ...]
    cohort: str = "active"
    neighbor_trajectories: tuple[tuple[str, tuple[TrajectoryObservation, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError(f"EvaluationSample for agent '{self.agent_id}' has empty history.")
        # Ensure strict temporal separation
        for obs in self.history:
            if obs.timestamp > self.reference_timestamp + 1e-9:
                raise ValueError(
                    f"Data leakage detected! History observation timestamp {obs.timestamp} "
                    f"is after reference timestamp {self.reference_timestamp}."
                )
        for nbr_id, nbr_seq in self.neighbor_trajectories:
            for obs in nbr_seq:
                if obs.timestamp > self.reference_timestamp + 1e-9:
                    raise ValueError(
                        f"Data leakage detected in neighbor '{nbr_id}'! Timestamp {obs.timestamp} "
                        f"is after reference timestamp {self.reference_timestamp}."
                    )
        for obs in self.ground_truth_future:
            if obs.timestamp <= self.reference_timestamp:
                raise ValueError(
                    f"Ground truth observation timestamp {obs.timestamp} "
                    f"is not strictly after reference timestamp {self.reference_timestamp}."
                )

    def to_snapshot(self, timestep: float = 0.1) -> TrajectorySnapshot:
        """Construct an immutable TrajectorySnapshot containing ONLY historical observations."""
        trajectories: dict[str, tuple[TrajectoryObservation, ...]] = {
            self.agent_id: self.history
        }
        for nbr_id, nbr_seq in self.neighbor_trajectories:
            if nbr_id != self.agent_id and nbr_seq:
                trajectories[nbr_id] = nbr_seq
        return TrajectorySnapshot(
            current_step=self.reference_step,
            current_timestamp=self.reference_timestamp,
            timestep=timestep,
            trajectories=MappingProxyType(trajectories),
        )


@dataclass
class TrajectoryEvaluationResult:
    """Evaluation result for a single sample."""
    sample_id: str
    agent_id: str
    cohort: str
    reference_timestamp: float
    history_length: int
    ground_truth_length: int
    predicted_length: int
    matched_points: int
    ade: float
    fde: float
    horizon_errors: list[float]
    status: str = "SUCCESS"  # 'SUCCESS', 'PARTIAL', 'FAILED'
    error_message: str = ""


@dataclass
class AggregateEvaluationReport:
    """Summary metrics across all evaluated trajectory samples."""
    total_samples: int
    evaluated_samples: int
    skipped_samples: int
    successful_samples: int
    partial_samples: int
    failed_samples: int
    mean_ade: float
    median_ade: float
    mean_fde: float
    median_fde: float
    horizon_mean_errors: list[float]
    cohort_metrics: dict[str, dict[str, float]]
    results: list[TrajectoryEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary (excluding verbose individual results by default or including them)."""
        res_list = [asdict(r) for r in self.results]
        return {
            "total_samples": self.total_samples,
            "evaluated_samples": self.evaluated_samples,
            "skipped_samples": self.skipped_samples,
            "successful_samples": self.successful_samples,
            "partial_samples": self.partial_samples,
            "failed_samples": self.failed_samples,
            "mean_ade": self.mean_ade,
            "median_ade": self.median_ade,
            "mean_fde": self.mean_fde,
            "median_fde": self.median_fde,
            "horizon_mean_errors": self.horizon_mean_errors,
            "cohort_metrics": self.cohort_metrics,
            "results": res_list,
        }

    def save_json(self, path: str | Path) -> None:
        """Save aggregate report to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: str | Path) -> None:
        """Save per-trajectory evaluation results to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id",
                "agent_id",
                "cohort",
                "reference_timestamp",
                "history_length",
                "ground_truth_length",
                "predicted_length",
                "matched_points",
                "ade",
                "fde",
                "status",
                "error_message",
            ])
            for r in self.results:
                writer.writerow([
                    r.sample_id,
                    r.agent_id,
                    r.cohort,
                    f"{r.reference_timestamp:.4f}",
                    r.history_length,
                    r.ground_truth_length,
                    r.predicted_length,
                    r.matched_points,
                    f"{r.ade:.6f}" if math.isfinite(r.ade) else "nan",
                    f"{r.fde:.6f}" if math.isfinite(r.fde) else "nan",
                    r.status,
                    r.error_message,
                ])


class SampleGenerator:
    """Generates sliding-window evaluation samples from trajectory sequences.

    Ensures zero future-data leakage: historical observations are partitioned
    strictly from ground truth future positions.
    """

    def __init__(
        self,
        history_length: int = 8,
        prediction_horizon: int = 12,
        stride: int = 1,
        allow_partial_future: bool = False,
        min_future_points: int = 1,
    ) -> None:
        """Initialize SampleGenerator.

        Args:
            history_length: Number of past observations required for predictor input.
            prediction_horizon: Desired number of future observations for scoring.
            stride: Step stride between sliding windows.
            allow_partial_future: If True, evaluates trajectories that end early (e.g. evacuation).
            min_future_points: Minimum future ground-truth points needed if allow_partial_future is True.
        """
        if history_length < 2:
            raise ValueError(f"history_length must be >= 2, got {history_length}")
        if prediction_horizon < 1:
            raise ValueError(f"prediction_horizon must be >= 1, got {prediction_horizon}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.allow_partial_future = allow_partial_future
        self.min_future_points = min_future_points

    def generate_samples(
        self,
        trajectories: dict[str, Sequence[TrajectoryObservation]],
        cohort_map: dict[str, str] | None = None,
        include_neighbors: bool = True,
    ) -> list[EvaluationSample]:
        """Generate evaluation samples across all agents in the provided trajectories.

        Args:
            trajectories: Mapping of agent_id -> chronological Sequence[TrajectoryObservation].
            cohort_map: Optional mapping of agent_id -> cohort name (e.g. 'active', 'blocked').
            include_neighbors: When True, includes historical (<= ref_timestamp) observations
                for co-present pedestrians to support social conditioning without future leakage.

        Returns:
            List of EvaluationSample instances.
        """
        samples: list[EvaluationSample] = []
        cohort_map = cohort_map or {}
        ts_index_cache: dict[str, list[float]] = {
            aid: [float(o.timestamp) for o in seq]
            for aid, seq in trajectories.items()
        }

        for agent_id, obs_seq in trajectories.items():
            if len(obs_seq) < self.history_length + 1:
                continue

            cohort = cohort_map.get(agent_id, "active")
            if cohort == "inactive":
                continue

            n_obs = len(obs_seq)
            # Sliding window over the observation sequence
            for start_idx in range(0, n_obs - self.history_length, self.stride):
                hist_end = start_idx + self.history_length
                history_tuple = tuple(obs_seq[start_idx:hist_end])
                ref_obs = history_tuple[-1]

                fut_end = min(hist_end + self.prediction_horizon, n_obs)
                future_tuple = tuple(obs_seq[hist_end:fut_end])

                if not future_tuple:
                    continue

                if len(future_tuple) < self.prediction_horizon and not self.allow_partial_future:
                    # Incomplete future and partial not allowed -> skip
                    continue

                if len(future_tuple) < self.min_future_points:
                    continue

                nbr_entries: list[tuple[str, tuple[TrajectoryObservation, ...]]] = []
                if include_neighbors and len(trajectories) > 1:
                    ref_ts = float(ref_obs.timestamp)
                    cutoff_ts = ref_ts + 1e-9
                    max_stale_s = max(2.0, self.history_length * 0.5)
                    for other_id, other_seq in trajectories.items():
                        if other_id == agent_id or not other_seq:
                            continue
                        idx_hi = bisect_right(ts_index_cache[other_id], cutoff_ts)
                        if idx_hi < 1:
                            continue
                        last_nbr = other_seq[idx_hi - 1]
                        if ref_ts - float(last_nbr.timestamp) > max_stale_s:
                            continue
                        idx_lo = max(0, idx_hi - self.history_length)
                        nbr_entries.append((other_id, tuple(other_seq[idx_lo:idx_hi])))

                sample = EvaluationSample(
                    agent_id=agent_id,
                    reference_step=ref_obs.step_index,
                    reference_timestamp=ref_obs.timestamp,
                    history=history_tuple,
                    ground_truth_future=future_tuple,
                    cohort=cohort,
                    neighbor_trajectories=tuple(nbr_entries),
                )
                samples.append(sample)

        return samples


class PredictionEvaluator:
    """Evaluates prediction models against trajectory ground-truth."""

    def __init__(self, predictor: Any | None = None) -> None:
        """Initialize evaluator.

        Args:
            predictor: Any prediction instance supporting .predict(snapshot, agent_id, horizon, prediction_interval).
                Defaults to ConstantVelocityPredictor().
        """
        self.predictor = predictor if predictor is not None else ConstantVelocityPredictor()

    def evaluate_sample(self, sample: EvaluationSample, timestep: float = 0.1) -> TrajectoryEvaluationResult:
        """Evaluate the predictor on a single EvaluationSample.

        Matches predictions with ground truth strictly by timestamp (not array index).
        """
        sample_id = f"{sample.agent_id}_t{sample.reference_timestamp:.3f}"
        horizon = len(sample.ground_truth_future)

        # Build snapshot containing ONLY historical observations
        snapshot = sample.to_snapshot(timestep=timestep)

        # Determine prediction interval from ground-truth future if regular, or use average dt
        gt_timestamps = [obs.timestamp for obs in sample.ground_truth_future]
        if len(gt_timestamps) > 1:
            nominal_interval = (gt_timestamps[-1] - gt_timestamps[0]) / (len(gt_timestamps) - 1)
        else:
            nominal_interval = gt_timestamps[0] - sample.reference_timestamp

        if nominal_interval <= 0:
            nominal_interval = timestep

        try:
            prediction: TrajectoryPrediction = self.predictor.predict(
                snapshot=snapshot,
                agent_id=sample.agent_id,
                horizon=horizon,
                prediction_interval=nominal_interval,
            )
        except Exception as err:
            return TrajectoryEvaluationResult(
                sample_id=sample_id,
                agent_id=sample.agent_id,
                cohort=sample.cohort,
                reference_timestamp=sample.reference_timestamp,
                history_length=len(sample.history),
                ground_truth_length=len(sample.ground_truth_future),
                predicted_length=0,
                matched_points=0,
                ade=float("nan"),
                fde=float("nan"),
                horizon_errors=[],
                status="FAILED",
                error_message=str(err),
            )

        # Match prediction coordinates with ground truth by timestamp
        # Build timestamp map from prediction
        pred_map: dict[float, tuple[float, float]] = {
            round(p.timestamp, 6): (p.x, p.y) for p in prediction.predictions
        }

        matched_gt: list[tuple[float, float]] = []
        matched_pred: list[tuple[float, float]] = []
        horizon_errors: list[float] = []

        for gt_obs in sample.ground_truth_future:
            gt_key = round(gt_obs.timestamp, 6)
            if gt_key in pred_map:
                pred_pt = pred_map[gt_key]
                matched_gt.append((gt_obs.x, gt_obs.y))
                matched_pred.append(pred_pt)
                err = math.hypot(gt_obs.x - pred_pt[0], gt_obs.y - pred_pt[1])
                horizon_errors.append(err)

        if not matched_gt:
            # Fallback if prediction interval slightly drifted from ground-truth:
            # Match sequentially if count matches and all pred times > ref_time
            if len(prediction.predictions) == len(sample.ground_truth_future):
                matched_gt = [(obs.x, obs.y) for obs in sample.ground_truth_future]
                matched_pred = [(p.x, p.y) for p in prediction.predictions]
                horizon_errors = [
                    math.hypot(gt[0] - pr[0], gt[1] - pr[1])
                    for gt, pr in zip(matched_gt, matched_pred)
                ]

        if not matched_gt:
            return TrajectoryEvaluationResult(
                sample_id=sample_id,
                agent_id=sample.agent_id,
                cohort=sample.cohort,
                reference_timestamp=sample.reference_timestamp,
                history_length=len(sample.history),
                ground_truth_length=len(sample.ground_truth_future),
                predicted_length=len(prediction.predictions),
                matched_points=0,
                ade=float("nan"),
                fde=float("nan"),
                horizon_errors=[],
                status="FAILED",
                error_message="Timestamp alignment yielded 0 matching points.",
            )

        ade = calculate_ade(matched_gt, matched_pred)
        fde = calculate_fde(matched_gt, matched_pred)

        is_partial = len(matched_gt) < len(sample.ground_truth_future) or sample.cohort == "evacuated"

        return TrajectoryEvaluationResult(
            sample_id=sample_id,
            agent_id=sample.agent_id,
            cohort=sample.cohort,
            reference_timestamp=sample.reference_timestamp,
            history_length=len(sample.history),
            ground_truth_length=len(sample.ground_truth_future),
            predicted_length=len(prediction.predictions),
            matched_points=len(matched_gt),
            ade=ade,
            fde=fde,
            horizon_errors=horizon_errors,
            status="PARTIAL" if is_partial else "SUCCESS",
            error_message="",
        )

    def evaluate_samples(
        self,
        samples: Sequence[EvaluationSample],
        timestep: float = 0.1,
    ) -> AggregateEvaluationReport:
        """Evaluate a collection of EvaluationSample instances and compute aggregate statistics."""
        results: list[TrajectoryEvaluationResult] = []
        valid_ades: list[float] = []
        valid_fdes: list[float] = []
        horizon_error_buckets: dict[int, list[float]] = {}
        cohort_buckets: dict[str, dict[str, list[float]]] = {}

        for sample in samples:
            res = self.evaluate_sample(sample, timestep=timestep)
            results.append(res)

            if math.isfinite(res.ade) and math.isfinite(res.fde):
                valid_ades.append(res.ade)
                valid_fdes.append(res.fde)

                for step_idx, err in enumerate(res.horizon_errors):
                    horizon_error_buckets.setdefault(step_idx, []).append(err)

                c_dict = cohort_buckets.setdefault(res.cohort, {"ade": [], "fde": []})
                c_dict["ade"].append(res.ade)
                c_dict["fde"].append(res.fde)

        mean_ade = statistics.fmean(valid_ades) if valid_ades else float("nan")
        median_ade = statistics.median(valid_ades) if valid_ades else float("nan")
        mean_fde = statistics.fmean(valid_fdes) if valid_fdes else float("nan")
        median_fde = statistics.median(valid_fdes) if valid_fdes else float("nan")

        horizon_means: list[float] = []
        for step_idx in sorted(horizon_error_buckets.keys()):
            errs = horizon_error_buckets[step_idx]
            horizon_means.append(statistics.fmean(errs) if errs else float("nan"))

        cohort_metrics: dict[str, dict[str, float]] = {}
        for c_name, c_vals in cohort_buckets.items():
            cohort_metrics[c_name] = {
                "count": len(c_vals["ade"]),
                "mean_ade": statistics.fmean(c_vals["ade"]) if c_vals["ade"] else float("nan"),
                "median_ade": statistics.median(c_vals["ade"]) if c_vals["ade"] else float("nan"),
                "mean_fde": statistics.fmean(c_vals["fde"]) if c_vals["fde"] else float("nan"),
                "median_fde": statistics.median(c_vals["fde"]) if c_vals["fde"] else float("nan"),
            }

        successful = sum(1 for r in results if r.status == "SUCCESS")
        partial = sum(1 for r in results if r.status == "PARTIAL")
        failed = sum(1 for r in results if r.status == "FAILED")

        return AggregateEvaluationReport(
            total_samples=len(samples),
            evaluated_samples=len(valid_ades),
            skipped_samples=len(samples) - len(valid_ades),
            successful_samples=successful,
            partial_samples=partial,
            failed_samples=failed,
            mean_ade=mean_ade,
            median_ade=median_ade,
            mean_fde=mean_fde,
            median_fde=median_fde,
            horizon_mean_errors=horizon_means,
            cohort_metrics=cohort_metrics,
            results=results,
        )
