"""Scenario comparison runner and orchestration pipeline for GenEvac-Sentinel Phase 06.

Executes declared scenarios non-invasively using Pavan's SimulationEngine,
extracts Phase 02/05 metrics per run, validates comparability across scenarios,
and generates auditable aggregate study results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from genevac.environment import EnvironmentMap, Exit, Obstacle
from genevac.experiments.comparability import ComparabilityDecision, ComparabilityGate, ComparabilityResult
from genevac.experiments.scenario import (
    AggregateMetricResult,
    ComparisonDefinition,
    PerRunMetricResult,
    RunRecord,
    RunStatus,
    ScenarioDefinition,
    TerminationReason,
    aggregate_per_run_results,
)
from genevac.metrics import (
    calculate_area_density,
    calculate_clearance_time,
    calculate_minimum_separation,
)
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.simulation import SimulationEngine


@dataclass(frozen=True)
class MetricDelta:
    """Quantitative comparison of a metric between a variant and baseline."""
    metric_id: str
    baseline_mean: Optional[float]
    variant_mean: Optional[float]
    absolute_delta: Optional[float]
    percentage_delta: Optional[float]
    comparability: ComparabilityResult


@dataclass(frozen=True)
class ComparisonStudyResult:
    """Complete, self-contained result of a scenario comparison study."""
    comparison: ComparisonDefinition
    scenarios: Dict[str, ScenarioDefinition]
    run_records: Tuple[RunRecord, ...]
    per_run_results: Tuple[PerRunMetricResult, ...]
    aggregates: Dict[Tuple[str, str], AggregateMetricResult]  # (scenario_id, metric_id) -> Aggregate
    comparability_results: Dict[Tuple[str, str], ComparabilityResult]  # (variant_id, metric_id) -> Result
    metric_deltas: Dict[Tuple[str, str], MetricDelta]  # (variant_id, metric_id) -> Delta
    executed_at: str
    reproducibility_notes: str
    repetition_sample_type: str = "deterministic_duplicate"


class ScenarioComparisonRunner:
    """Orchestrator for multi-scenario comparative evacuation experiments."""

    @classmethod
    def execute_comparison(
        cls,
        comparison: ComparisonDefinition,
        scenarios: Sequence[ScenarioDefinition],
        repetitions: int = 1,
    ) -> ComparisonStudyResult:
        """Execute a formal comparison study across provided scenario definitions.

        Args:
            comparison: ComparisonDefinition specifying research question, baseline, and factors.
            scenarios: Sequence of ScenarioDefinition objects (must contain baseline and variants).
            repetitions: Number of runs per scenario (governed by repetition_policy).

        Returns:
            ComparisonStudyResult containing raw run records, per-run metrics, comparability audits,
            and deterministic aggregates.
        """
        scenario_map = {s.scenario_id: s for s in scenarios}
        if comparison.baseline_scenario_id not in scenario_map:
            raise ValueError(f"Baseline scenario '{comparison.baseline_scenario_id}' missing from provided scenarios.")
        for s_id in comparison.scenario_ids:
            if s_id not in scenario_map:
                raise ValueError(f"Declared scenario '{s_id}' missing from provided scenarios.")

        all_run_records: List[RunRecord] = []
        all_per_run_results: List[PerRunMetricResult] = []

        # 1. Execute runs for each scenario
        for s_id in comparison.scenario_ids:
            sc = scenario_map[s_id]
            for rep_idx in range(repetitions):
                run_id = f"{s_id}_run_{rep_idx + 1:02d}"
                run_rec, metrics = cls._execute_single_run(sc, run_id, rep_idx)
                all_run_records.append(run_rec)
                all_per_run_results.extend(metrics)

        # 2. Deterministic Aggregation (Per-run first)
        aggregates: Dict[Tuple[str, str], AggregateMetricResult] = {}
        for s_id in comparison.scenario_ids:
            for m_id in comparison.dependent_metrics:
                unit = cls._get_metric_unit(m_id)
                agg = aggregate_per_run_results(
                    results=all_per_run_results,
                    run_records=all_run_records,
                    scenario_id=s_id,
                    metric_id=m_id,
                    unit=unit,
                )
                aggregates[(s_id, m_id)] = agg

        # 3. Audit Comparability across baseline and variants
        base_sc = scenario_map[comparison.baseline_scenario_id]
        comparability_results: Dict[Tuple[str, str], ComparabilityResult] = {}
        for s_id in comparison.scenario_ids:
            if s_id == comparison.baseline_scenario_id:
                continue
            var_sc = scenario_map[s_id]

            for m_id in comparison.dependent_metrics:
                unit = cls._get_metric_unit(m_id)
                base_agg = aggregates[(comparison.baseline_scenario_id, m_id)]
                var_agg = aggregates[(s_id, m_id)]

                base_data_avail = (base_agg.eligible_run_count > 0 and base_agg.mean is not None)
                var_data_avail = (var_agg.eligible_run_count > 0 and var_agg.mean is not None)

                base_meta = {
                    "environment_bounds": (0.0, 0.0, base_sc.engine_config.get("width", 50.0), base_sc.engine_config.get("height", 20.0)),
                    "agent_count": base_sc.agent_count,
                    "simulated_duration": base_sc.engine_config.get("max_steps", 100) * base_sc.engine_config.get("timestep", 0.1),
                    "unit": unit,
                    "data_available": base_data_avail,
                    "status": "VALID" if base_data_avail else "UNAVAILABLE",
                    "exit_events_available": False if m_id == "exit_throughput" else True,
                }
                var_meta = {
                    "environment_bounds": (0.0, 0.0, var_sc.engine_config.get("width", 50.0), var_sc.engine_config.get("height", 20.0)),
                    "agent_count": var_sc.agent_count,
                    "simulated_duration": var_sc.engine_config.get("max_steps", 100) * var_sc.engine_config.get("timestep", 0.1),
                    "unit": unit,
                    "data_available": var_data_avail,
                    "status": "VALID" if var_data_avail else "UNAVAILABLE",
                    "exit_events_available": False if m_id == "exit_throughput" else True,
                }

                comp_res = ComparabilityGate.evaluate_metric_comparability(
                    metric_id=m_id,
                    baseline_meta=base_meta,
                    variant_meta=var_meta,
                    comparability_policy=comparison.comparability_policy,
                )
                comparability_results[(s_id, m_id)] = comp_res

        # 4. Compute Baseline Deltas
        metric_deltas: Dict[Tuple[str, str], MetricDelta] = {}
        for s_id in comparison.scenario_ids:
            if s_id == comparison.baseline_scenario_id:
                continue
            for m_id in comparison.dependent_metrics:
                base_agg = aggregates[(comparison.baseline_scenario_id, m_id)]
                var_agg = aggregates[(s_id, m_id)]
                comp_res = comparability_results[(s_id, m_id)]

                b_mean = base_agg.mean
                v_mean = var_agg.mean

                if b_mean is not None and v_mean is not None and comp_res.is_comparable:
                    abs_delta = v_mean - b_mean
                    pct_delta = (abs_delta / b_mean * 100.0) if not math.isclose(b_mean, 0.0) else None
                else:
                    abs_delta = None
                    pct_delta = None

                metric_deltas[(s_id, m_id)] = MetricDelta(
                    metric_id=m_id,
                    baseline_mean=b_mean,
                    variant_mean=v_mean,
                    absolute_delta=abs_delta,
                    percentage_delta=pct_delta,
                    comparability=comp_res,
                )

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        has_determ_reps = any(agg.is_deterministic_repetition for agg in aggregates.values() if agg.eligible_run_count > 1)
        if has_determ_reps and repetitions > 1:
            sample_type = "deterministic_duplicate"
            repro_notes = (
                f"Executed {repetitions} repetitions per scenario. "
                "All multi-run metrics produced zero variance (std_dev = 0.0) due to deterministic simulation kinematics. "
                "These are deterministic duplicates and must not be treated as independent stochastic samples."
            )
        else:
            sample_type = "single_or_stochastic"
            repro_notes = f"Executed {repetitions} repetitions per scenario. Config fingerprinting enforced."

        return ComparisonStudyResult(
            comparison=comparison,
            scenarios=scenario_map,
            run_records=tuple(all_run_records),
            per_run_results=tuple(all_per_run_results),
            aggregates=aggregates,
            comparability_results=comparability_results,
            metric_deltas=metric_deltas,
            executed_at=now_iso,
            reproducibility_notes=repro_notes,
            repetition_sample_type=sample_type,
        )

    @classmethod
    def _execute_single_run(
        cls,
        scenario: ScenarioDefinition,
        run_id: str,
        rep_idx: int,
    ) -> Tuple[RunRecord, List[PerRunMetricResult]]:
        """Run a single execution of a scenario using the simulation engine without mutating engine state."""
        start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 0. Validate Configuration before execution
        is_valid, validation_errors = scenario.validate_configuration()
        if not is_valid:
            failure_msg = "; ".join(validation_errors)
            run_rec = RunRecord(
                run_id=run_id,
                scenario_id=scenario.scenario_id,
                scenario_version=scenario.version,
                started_at=start_time_iso,
                ended_at=start_time_iso,
                seed=scenario.seed,
                initial_agent_count=scenario.agent_count,
                final_step=0,
                simulated_duration=0.0,
                termination_reason=TerminationReason.ERROR,
                run_status=RunStatus.FAILED,
                failure_reason=failure_msg,
                config_fingerprint=scenario.compute_fingerprint(),
                source_provenance={"engine": "genevac.simulation.SimulationEngine", "version": "1.0.0"},
            )
            failed_metrics = [
                PerRunMetricResult(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    metric_id=m_id,
                    display_name=m_id,
                    value=None,
                    unit=cls._get_metric_unit(m_id),
                    status="FAILED",
                    reason=failure_msg,
                )
                for m_id in ("total_elapsed_time", "clearance_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput")
            ]
            return run_rec, failed_metrics

        cfg = scenario.engine_config

        # 1. Build Environment
        width = float(cfg.get("width", 50.0))
        height = float(cfg.get("height", 20.0))
        env = EnvironmentMap(width=width, height=height)

        for ex in cfg.get("exits", []):
            env.add_exit(Exit(id=ex["id"], position=tuple(ex["position"]), width=float(ex.get("width", 2.0))))
        for obs in cfg.get("obstacles", []):
            env.add_obstacle(Obstacle(id=obs["id"], bounds=tuple(obs["bounds"])))

        # 2. Build Simulation Engine
        max_steps = int(cfg.get("max_steps", 100))
        timestep = float(cfg.get("timestep", 0.1))
        seed = scenario.seed if scenario.seed is not None else 42
        engine = SimulationEngine(environment=env, max_steps=max_steps, timestep=timestep, seed=seed)

        # 3. Add Pedestrians from initial conditions
        peds_cfg = cfg.get("pedestrians", [])
        if peds_cfg:
            for p_data in peds_cfg:
                engine.add_pedestrian(
                    Pedestrian(
                        id=p_data["id"],
                        position=tuple(p_data["position"]),
                        velocity=tuple(p_data.get("velocity", (0.0, 0.0))),
                        desired_speed=float(p_data.get("desired_speed", 1.34)),
                        radius=float(p_data.get("radius", 0.3)),
                        target_exit=p_data.get("target_exit", ""),
                        status=PedestrianStatus(p_data.get("status", "active")),
                    )
                )
        else:
            # Deterministic default placement if none explicitly configured
            for i in range(scenario.agent_count):
                engine.add_pedestrian(
                    Pedestrian(
                        id=f"agent_{i+1:03d}",
                        position=(5.0 + (i % 10) * 2.0, 5.0 + (i // 10) * 2.0),
                        velocity=(1.0, 0.0),
                        desired_speed=1.34,
                        radius=0.3,
                        target_exit="exit_1",
                        status=PedestrianStatus.ACTIVE,
                    )
                )

        initial_count = len(engine.pedestrians)

        # 4. Execute stepping
        termination = TerminationReason.COMPLETED
        failure_msg = ""
        run_status = RunStatus.SUCCESS

        try:
            while not engine.is_finished():
                engine.step()

            if engine.current_step >= engine.max_steps:
                active_remaining = [p for p in engine.pedestrians if p.status == PedestrianStatus.ACTIVE]
                if active_remaining:
                    termination = TerminationReason.MAX_STEPS_REACHED
        except Exception as e:
            run_status = RunStatus.FAILED
            termination = TerminationReason.ERROR
            failure_msg = str(e)

        end_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        final_step = engine.current_step
        sim_duration = final_step * timestep

        run_rec = RunRecord(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.version,
            started_at=start_time_iso,
            ended_at=end_time_iso,
            seed=seed,
            initial_agent_count=initial_count,
            final_step=final_step,
            simulated_duration=sim_duration,
            termination_reason=termination,
            run_status=run_status,
            failure_reason=failure_msg,
            config_fingerprint=scenario.compute_fingerprint(),
            source_provenance={"engine": "genevac.simulation.SimulationEngine", "version": "1.0.0"},
        )

        # 5. Compute Phase 02 per-run metrics
        metrics: List[PerRunMetricResult] = []
        if run_status == RunStatus.SUCCESS:
            # Metric: Total Elapsed Simulation Time (Distinct from evacuation clearance time)
            metrics.append(
                PerRunMetricResult(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    metric_id="total_elapsed_time",
                    display_name="Total Elapsed Simulation Time",
                    value=sim_duration,
                    unit="seconds",
                    status="VALID",
                    time_window=(0.0, sim_duration),
                    source_step_range=(0, final_step),
                    valid_sample_count=1,
                )
            )

            # Metric: Evacuation Clearance Time (Only VALID if all agents evacuated)
            evacuated_peds = [p for p in engine.pedestrians if p.status == PedestrianStatus.EVACUATED]
            if len(evacuated_peds) == initial_count and initial_count > 0:
                clearance_val = sim_duration
                clearance_status = "VALID"
                clearance_reason = ""
            else:
                clearance_val = None
                clearance_status = "UNAVAILABLE"
                clearance_reason = (
                    f"Evacuation incomplete: {len(evacuated_peds)}/{initial_count} evacuated before simulation end."
                )

            metrics.append(
                PerRunMetricResult(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    metric_id="clearance_time",
                    display_name="Evacuation Clearance Time",
                    value=clearance_val,
                    unit="seconds",
                    status=clearance_status,
                    reason=clearance_reason,
                    time_window=(0.0, sim_duration),
                    source_step_range=(0, final_step),
                    valid_sample_count=1 if clearance_val is not None else 0,
                )
            )

            # Metric: Mean Pedestrian Speed
            speeds = [math.hypot(p.velocity[0], p.velocity[1]) for p in engine.pedestrians]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            metrics.append(
                PerRunMetricResult(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    metric_id="mean_speed",
                    display_name="Mean Pedestrian Speed",
                    value=avg_speed,
                    unit="m/s",
                    status="VALID",
                    time_window=(0.0, sim_duration),
                    source_step_range=(0, final_step),
                    valid_sample_count=len(speeds),
                )
            )

            # Metric: Area Density
            positions = [p.position for p in engine.pedestrians]
            density_val = calculate_area_density(positions, (0.0, 0.0, width, height))
            metrics.append(
                PerRunMetricResult(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    metric_id="area_density",
                    display_name="Pedestrian Area Density",
                    value=density_val,
                    unit="ped/m²",
                    status="VALID",
                    time_window=(0.0, sim_duration),
                    source_step_range=(0, final_step),
                    valid_sample_count=1,
                )
            )

            # Metric: Minimum Separation
            positions = [p.position for p in engine.pedestrians]
            min_sep = calculate_minimum_separation(positions)
            metrics.append(
                PerRunMetricResult(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    metric_id="minimum_separation",
                    display_name="Minimum Pedestrian Separation",
                    value=min_sep,
                    unit="metres",
                    status="VALID",
                    time_window=(0.0, sim_duration),
                    source_step_range=(0, final_step),
                    valid_sample_count=len(positions),
                )
            )

            # Metric: Exit Throughput (Discrete exit events not emitted by engine -> UNAVAILABLE)
            metrics.append(
                PerRunMetricResult(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    metric_id="exit_throughput",
                    display_name="Exit Throughput Rate",
                    value=None,
                    unit="ped/s",
                    status="UNAVAILABLE",
                    reason="Authoritative discrete exit events not emitted by engine.",
                    valid_sample_count=0,
                )
            )
        else:
            # Record failed metric entries
            for m_id in ("total_elapsed_time", "clearance_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput"):
                metrics.append(
                    PerRunMetricResult(
                        run_id=run_id,
                        scenario_id=scenario.scenario_id,
                        metric_id=m_id,
                        display_name=m_id,
                        value=None,
                        unit=cls._get_metric_unit(m_id),
                        status="FAILED",
                        reason=failure_msg,
                    )
                )

        return run_rec, metrics

    @staticmethod
    def _get_metric_unit(metric_id: str) -> str:
        units = {
            "total_elapsed_time": "seconds",
            "clearance_time": "seconds",
            "mean_speed": "m/s",
            "area_density": "ped/m²",
            "minimum_separation": "metres",
            "exit_throughput": "ped/s",
            "ade": "metres",
            "fde": "metres",
            "path_length": "metres",
        }
        return units.get(metric_id, "")
