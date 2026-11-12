"""Experiment export and reporting module for GenEvac-Sentinel Phase 07.

Provides robust, auditable export pipelines for Phase 06 scenario comparison studies:
- Part A: Structured CSV exports for Scenario Inventory, Per-Run Metrics, and Aggregate Results.
- Part B: Portable JSON export with full schema specification and null-safe value representation.
- Part C: Truthful reproducibility metadata distinguishing configured from engine-consumed parameters.
- Part D: Complete 11-section Markdown experiment report generation.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from genevac.experiments.comparability import ComparabilityDecision, ComparabilityResult
from genevac.experiments.presentation import (
    generate_metric_comparison_table,
    generate_per_run_detail_table,
    generate_scenario_inventory_table,
    generate_scientific_interpretation,
)
from genevac.experiments.runner import ComparisonStudyResult, MetricDelta
from genevac.experiments.scenario import (
    AggregateMetricResult,
    ComparisonDefinition,
    PerRunMetricResult,
    RunRecord,
    RunStatus,
    ScenarioDefinition,
)

SCHEMA_VERSION = "1.0.0"
SOFTWARE_VERSION = "1.0.0"


def _format_float(val: Optional[float], decimals: int = 6) -> str:
    """Format float cleanly or return empty string for None/non-finite."""
    if val is None or not math.isfinite(val):
        return ""
    return f"{val:.{decimals}f}"


def export_scenario_inventory_csv(
    study: ComparisonStudyResult,
    filepath: Optional[Union[Path, str]] = None,
) -> str:
    """Export the scenario inventory as a standardized CSV table.

    Columns:
    - experiment_id
    - scenario_id
    - scenario_name
    - scenario_version
    - scenario_description
    - role (BASELINE or VARIANT)
    - agent_count
    - environment_id
    - room_width
    - room_height
    - obstacle_count
    - exit_count
    - max_steps
    - timestep
    - seed
    - stopping_condition
    - config_fingerprint
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    headers = [
        "experiment_id",
        "scenario_id",
        "scenario_name",
        "scenario_version",
        "scenario_description",
        "role",
        "agent_count",
        "environment_id",
        "room_width",
        "room_height",
        "obstacle_count",
        "exit_count",
        "max_steps",
        "timestep",
        "seed",
        "stopping_condition",
        "config_fingerprint",
    ]
    writer.writerow(headers)

    exp_id = study.comparison.comparison_id if study.comparison else ""
    base_id = study.comparison.baseline_scenario_id if study.comparison else ""

    for s_id in sorted(study.scenarios.keys()):
        sc = study.scenarios[s_id]
        role = "BASELINE" if s_id == base_id else "VARIANT"
        cfg = sc.engine_config or {}

        w = cfg.get("width")
        w_str = _format_float(w, 2) if isinstance(w, (int, float)) else ""
        h = cfg.get("height")
        h_str = _format_float(h, 2) if isinstance(h, (int, float)) else ""

        obstacles = cfg.get("obstacles", [])
        obs_count = len(obstacles) if isinstance(obstacles, list) else 0

        exits = cfg.get("exits", [])
        exit_count = len(exits) if isinstance(exits, list) else 0

        max_steps = cfg.get("max_steps", "")
        ts = cfg.get("timestep")
        ts_str = _format_float(ts, 4) if isinstance(ts, (int, float)) else ""

        seed_str = str(sc.seed) if sc.seed is not None else ""
        fp = sc.compute_fingerprint()

        writer.writerow([
            exp_id,
            sc.scenario_id,
            sc.name,
            sc.version,
            sc.description,
            role,
            sc.agent_count,
            sc.environment_id,
            w_str,
            h_str,
            obs_count,
            exit_count,
            max_steps,
            ts_str,
            seed_str,
            sc.stopping_condition,
            fp,
        ])

    csv_content = output.getvalue()
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_content, encoding="utf-8")

    return csv_content


def export_per_run_metrics_csv(
    study: ComparisonStudyResult,
    filepath: Optional[Union[Path, str]] = None,
) -> str:
    """Export per-run metric results as a standardized CSV table (one row per run and metric).

    Columns:
    - experiment_id
    - scenario_id
    - run_id
    - run_status
    - metric_name
    - metric_value
    - metric_unit
    - availability_status
    - unavailable_reason
    - repetition_sample_type
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    headers = [
        "experiment_id",
        "scenario_id",
        "run_id",
        "run_status",
        "metric_name",
        "metric_value",
        "metric_unit",
        "availability_status",
        "unavailable_reason",
        "repetition_sample_type",
    ]
    writer.writerow(headers)

    exp_id = study.comparison.comparison_id if study.comparison else ""
    runs_by_id = {r.run_id: r for r in study.run_records}
    seen_combinations = set()

    # 1. Write computed per-run results
    sorted_results = sorted(
        study.per_run_results,
        key=lambda x: (x.scenario_id, x.run_id, x.metric_id),
    )

    for res in sorted_results:
        run_rec = runs_by_id.get(res.run_id)
        run_status = run_rec.run_status.value if run_rec else "UNKNOWN"

        if res.status == "VALID" and res.value is not None and math.isfinite(res.value):
            val_str = _format_float(res.value, 6)
            unavail_reason = ""
            status_str = "VALID"
        else:
            val_str = ""
            status_str = "UNAVAILABLE"
            unavail_reason = res.reason or (run_rec.failure_reason if run_rec and run_rec.run_status != RunStatus.SUCCESS else "Measurement unavailable")

        writer.writerow([
            exp_id,
            res.scenario_id,
            res.run_id,
            run_status,
            res.metric_id,
            val_str,
            res.unit,
            status_str,
            unavail_reason,
            study.repetition_sample_type,
        ])
        seen_combinations.add((res.run_id, res.metric_id))

    # 2. Account for failed or missing runs where metrics were not generated
    if study.comparison and study.comparison.dependent_metrics:
        for run_rec in sorted(study.run_records, key=lambda x: x.run_id):
            if run_rec.run_status != RunStatus.SUCCESS:
                for m_id in study.comparison.dependent_metrics:
                    if (run_rec.run_id, m_id) not in seen_combinations:
                        writer.writerow([
                            exp_id,
                            run_rec.scenario_id,
                            run_rec.run_id,
                            run_rec.run_status.value,
                            m_id,
                            "",
                            "",
                            "UNAVAILABLE",
                            f"Run failed early: {run_rec.failure_reason or run_rec.run_status.value}",
                            study.repetition_sample_type,
                        ])
                        seen_combinations.add((run_rec.run_id, m_id))

    csv_content = output.getvalue()
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_content, encoding="utf-8")

    return csv_content


def export_aggregate_comparison_csv(
    study: ComparisonStudyResult,
    filepath: Optional[Union[Path, str]] = None,
) -> str:
    """Export aggregate scenario comparison results as a standardized CSV table.

    Columns:
    - experiment_id
    - scenario_id
    - metric_name
    - mean
    - standard_deviation
    - minimum
    - maximum
    - sample_count
    - comparability_decision
    - availability_status
    - unavailable_reason
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    headers = [
        "experiment_id",
        "scenario_id",
        "metric_name",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "sample_count",
        "comparability_decision",
        "availability_status",
        "unavailable_reason",
    ]
    writer.writerow(headers)

    exp_id = study.comparison.comparison_id if study.comparison else ""
    base_id = study.comparison.baseline_scenario_id if study.comparison else ""

    # Sort aggregates deterministically by scenario_id, metric_id
    for (s_id, m_id), agg in sorted(study.aggregates.items(), key=lambda x: (x[0][0], x[0][1])):
        if agg.eligible_run_count > 0 and agg.mean is not None and math.isfinite(agg.mean):
            mean_str = _format_float(agg.mean, 6)
            std_str = _format_float(agg.std_dev, 6)
            min_str = _format_float(agg.min_val, 6)
            max_str = _format_float(agg.max_val, 6)
            avail_status = "VALID"
            unavail_reason = ""
        else:
            mean_str = ""
            std_str = ""
            min_str = ""
            max_str = ""
            avail_status = "UNAVAILABLE"
            unavail_reason = "; ".join(agg.exclusion_reasons) if agg.exclusion_reasons else "Data unavailable"

        # Comparability decision
        if s_id == base_id:
            comp_decision = "BASELINE"
        else:
            comp_res = study.comparability_results.get((s_id, m_id))
            comp_decision = comp_res.decision.value if comp_res else "NOT_EVALUATED"

        writer.writerow([
            exp_id,
            s_id,
            m_id,
            mean_str,
            std_str,
            min_str,
            max_str,
            agg.eligible_run_count,
            comp_decision,
            avail_status,
            unavail_reason,
        ])

    csv_content = output.getvalue()
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_content, encoding="utf-8")

    return csv_content


def extract_reproducibility_metadata(study: ComparisonStudyResult) -> Dict[str, Any]:
    """Extract full reproducibility metadata from study execution structures.

    Rules:
    - Truthfully reports that the inspected Pavan simulation engine kinematics are deterministic and do not consume PRNG seeds.
    - Accurately classifies multi-run identical outputs as deterministic duplicates.
    - Preserves canonical SHA-256 configuration fingerprints.
    """
    comp = study.comparison
    scenario_map = study.scenarios

    runs_by_status = {
        RunStatus.SUCCESS.value: 0,
        RunStatus.PARTIAL.value: 0,
        RunStatus.FAILED.value: 0,
    }
    for r in study.run_records:
        runs_by_status[r.run_status.value] = runs_by_status.get(r.run_status.value, 0) + 1

    configured_seeds = {s_id: sc.seed for s_id, sc in sorted(scenario_map.items())}
    timesteps = {s_id: (sc.engine_config or {}).get("timestep", 0.1) for s_id, sc in sorted(scenario_map.items())}
    max_steps = {s_id: (sc.engine_config or {}).get("max_steps", 100) for s_id, sc in sorted(scenario_map.items())}
    agent_counts = {s_id: sc.agent_count for s_id, sc in sorted(scenario_map.items())}
    fingerprints = {s_id: sc.compute_fingerprint() for s_id, sc in sorted(scenario_map.items())}

    return {
        "schema_version": SCHEMA_VERSION,
        "software_version": SOFTWARE_VERSION,
        "experiment_id": comp.comparison_id if comp else "",
        "execution_timestamp": study.executed_at,
        "baseline_scenario_id": comp.baseline_scenario_id if comp else "",
        "scenario_ids": list(comp.scenario_ids) if comp else [],
        "run_ids": [r.run_id for r in sorted(study.run_records, key=lambda x: x.run_id)],
        "total_runs_attempted": len(study.run_records),
        "runs_by_status": runs_by_status,
        "configured_seeds": configured_seeds,
        "seed_used_by_engine": False,
        "seed_usage_note": (
            "As inspected in the current Pavan SimulationEngine (genevac.simulation.engine), "
            "the engine's kinematics are purely deterministic and do not consume PRNG seeds during physics stepping. "
            "Supplied seeds are recorded as scenario intent only."
        ),
        "repetition_sample_type": study.repetition_sample_type,
        "reproducibility_notes": study.reproducibility_notes,
        "engine_timesteps": timesteps,
        "engine_max_steps": max_steps,
        "initial_agent_counts": agent_counts,
        "scenario_config_fingerprints": fingerprints,
    }


def export_experiment_json(
    study: ComparisonStudyResult,
    filepath: Optional[Union[Path, str]] = None,
    indent: int = 2,
) -> str:
    """Export complete scenario comparison experiment as a portable, valid JSON structure.

    Features:
    - Explicit null representation for missing / unavailable numeric values.
    - Pure primitive data types (no unsafe object serialization).
    - Preserves all scenario configurations, run records, per-run metrics, aggregates, and comparability decisions.
    """
    comp = study.comparison

    # 1. Scenarios map
    scenarios_data = {}
    for s_id, sc in sorted(study.scenarios.items()):
        scenarios_data[s_id] = {
            "scenario_id": sc.scenario_id,
            "name": sc.name,
            "version": sc.version,
            "description": sc.description,
            "environment_id": sc.environment_id,
            "environment_version": sc.environment_version,
            "agent_count": sc.agent_count,
            "engine_config": sc.engine_config,
            "seed": sc.seed,
            "stopping_condition": sc.stopping_condition,
            "max_duration": sc.max_duration,
            "config_fingerprint": sc.compute_fingerprint(),
        }

    # 2. Run records
    run_records_data = []
    for r in sorted(study.run_records, key=lambda x: x.run_id):
        run_records_data.append({
            "run_id": r.run_id,
            "scenario_id": r.scenario_id,
            "scenario_version": r.scenario_version,
            "started_at": r.started_at,
            "ended_at": r.ended_at,
            "seed": r.seed,
            "initial_agent_count": r.initial_agent_count,
            "final_step": r.final_step,
            "simulated_duration": r.simulated_duration,
            "termination_reason": r.termination_reason.value if hasattr(r.termination_reason, "value") else str(r.termination_reason),
            "run_status": r.run_status.value if hasattr(r.run_status, "value") else str(r.run_status),
            "failure_reason": r.failure_reason,
            "config_fingerprint": r.config_fingerprint,
        })

    # 3. Per-run metrics
    per_run_data = []
    for res in sorted(study.per_run_results, key=lambda x: (x.scenario_id, x.run_id, x.metric_id)):
        per_run_data.append({
            "run_id": res.run_id,
            "scenario_id": res.scenario_id,
            "metric_id": res.metric_id,
            "display_name": res.display_name,
            "value": res.value if (res.value is not None and math.isfinite(res.value)) else None,
            "unit": res.unit,
            "status": res.status,
            "reason": res.reason,
            "time_window": list(res.time_window),
            "population": res.population,
            "valid_sample_count": res.valid_sample_count,
            "missing_sample_count": res.missing_sample_count,
        })

    # 4. Aggregates
    aggregates_data = []
    for (s_id, m_id), agg in sorted(study.aggregates.items(), key=lambda x: (x[0][0], x[0][1])):
        aggregates_data.append({
            "scenario_id": agg.scenario_id,
            "metric_id": agg.metric_id,
            "unit": agg.unit,
            "attempted_run_count": agg.attempted_run_count,
            "successful_run_count": agg.successful_run_count,
            "partial_run_count": agg.partial_run_count,
            "failed_run_count": agg.failed_run_count,
            "eligible_run_count": agg.eligible_run_count,
            "excluded_run_count": agg.excluded_run_count,
            "unavailable_count": agg.unavailable_count,
            "mean": agg.mean if (agg.mean is not None and math.isfinite(agg.mean)) else None,
            "median": agg.median if (agg.median is not None and math.isfinite(agg.median)) else None,
            "std_dev": agg.std_dev if (agg.std_dev is not None and math.isfinite(agg.std_dev)) else None,
            "min_val": agg.min_val if (agg.min_val is not None and math.isfinite(agg.min_val)) else None,
            "max_val": agg.max_val if (agg.max_val is not None and math.isfinite(agg.max_val)) else None,
            "aggregation_method": agg.aggregation_method,
            "contributing_run_ids": list(agg.contributing_run_ids),
            "is_single_run": agg.is_single_run,
            "is_deterministic_repetition": agg.is_deterministic_repetition,
            "inclusion_reasons": list(agg.inclusion_reasons),
            "exclusion_reasons": list(agg.exclusion_reasons),
        })

    # 5. Comparability evaluations
    comparability_data = []
    for (v_id, m_id), c_res in sorted(study.comparability_results.items(), key=lambda x: (x[0][0], x[0][1])):
        comparability_data.append({
            "variant_scenario_id": v_id,
            "metric_id": m_id,
            "decision": c_res.decision.value if hasattr(c_res.decision, "value") else str(c_res.decision),
            "is_comparable": c_res.is_comparable,
            "reason_codes": list(c_res.reason_codes),
            "explanation": c_res.explanation,
            "caveats": list(c_res.caveats),
        })

    # 6. Metric deltas
    deltas_data = []
    for (v_id, m_id), delta in sorted(study.metric_deltas.items(), key=lambda x: (x[0][0], x[0][1])):
        deltas_data.append({
            "variant_scenario_id": v_id,
            "metric_id": m_id,
            "baseline_mean": delta.baseline_mean if (delta.baseline_mean is not None and math.isfinite(delta.baseline_mean)) else None,
            "variant_mean": delta.variant_mean if (delta.variant_mean is not None and math.isfinite(delta.variant_mean)) else None,
            "absolute_delta": delta.absolute_delta if (delta.absolute_delta is not None and math.isfinite(delta.absolute_delta)) else None,
            "percentage_delta": delta.percentage_delta if (delta.percentage_delta is not None and math.isfinite(delta.percentage_delta)) else None,
            "comparability_decision": delta.comparability.decision.value if hasattr(delta.comparability.decision, "value") else str(delta.comparability.decision),
        })

    full_payload = {
        "schema_version": SCHEMA_VERSION,
        "software_version": SOFTWARE_VERSION,
        "experiment_identifier": comp.comparison_id if comp else "",
        "comparison_metadata": {
            "title": comp.title if comp else "",
            "research_question": comp.research_question if comp else "",
            "baseline_scenario_id": comp.baseline_scenario_id if comp else "",
            "scenario_ids": list(comp.scenario_ids) if comp else [],
            "independent_variable": comp.independent_variable if comp else "",
            "dependent_metrics": list(comp.dependent_metrics) if comp else [],
            "controlled_factors": comp.controlled_factors if comp else {},
            "seed_policy": comp.seed_policy if comp else "",
            "notes": comp.notes if comp else "",
        },
        "executed_at": study.executed_at,
        "scenarios": scenarios_data,
        "run_records": run_records_data,
        "per_run_metrics": per_run_data,
        "aggregate_metrics": aggregates_data,
        "comparability_evaluations": comparability_data,
        "metric_deltas": deltas_data,
        "reproducibility_metadata": extract_reproducibility_metadata(study),
    }

    json_str = json.dumps(full_payload, indent=indent, ensure_ascii=False)
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_str, encoding="utf-8")

    return json_str


def generate_experiment_report(
    study: ComparisonStudyResult,
    filepath: Optional[Union[Path, str]] = None,
) -> str:
    """Generate a comprehensive, scientifically disciplined 11-section experiment report in Markdown.

    Sections:
    1. Experiment Overview
    2. Scenario Inventory
    3. Execution Summary
    4. Per-Run Results
    5. Aggregate Metric Comparisons
    6. Comparability Decisions
    7. Scientific Interpretation
    8. Unavailable Metrics and Reasons
    9. Failed Runs and Reasons
    10. Reproducibility Metadata
    11. Limitations & Boundaries
    """
    comp = study.comparison
    base_id = comp.baseline_scenario_id if comp else "None"
    repro_meta = extract_reproducibility_metadata(study)

    lines: List[str] = [
        f"# Experiment Evaluation Report: {comp.title if comp else 'Scenario Comparison'}",
        "",
        f"- **Experiment Identifier:** `{comp.comparison_id if comp else 'N/A'}`",
        f"- **Schema Version:** `{SCHEMA_VERSION}`",
        f"- **Execution Timestamp:** `{study.executed_at}`",
        f"- **Baseline Scenario:** `{base_id}`",
        "",
        "---",
        "",
        "## 1. Experiment Overview",
        "",
        f"- **Research Question:** {comp.research_question if comp else 'N/A'}",
        f"- **Independent Variable:** `{comp.independent_variable if comp else 'N/A'}`",
        f"- **Dependent Metrics:** {', '.join(f'`{m}`' for m in comp.dependent_metrics) if comp and comp.dependent_metrics else 'None'}",
        f"- **Controlled Factors:** {', '.join(f'`{k}={v}`' for k, v in comp.controlled_factors.items()) if comp and comp.controlled_factors else 'None declared'}",
        f"- **Repetition Policy:** {study.repetition_sample_type}",
        "",
        "---",
        "",
        "## 2. Scenario Inventory",
        "",
        generate_scenario_inventory_table(study),
        "",
        "---",
        "",
        "## 3. Execution Summary",
        "",
        f"- **Total Runs Attempted:** {repro_meta['total_runs_attempted']}",
        f"- **Successful Runs:** {repro_meta['runs_by_status'].get('SUCCESS', 0)}",
        f"- **Partial Runs:** {repro_meta['runs_by_status'].get('PARTIAL', 0)}",
        f"- **Failed Runs:** {repro_meta['runs_by_status'].get('FAILED', 0)}",
        "",
        generate_per_run_detail_table(study),
        "",
        "---",
        "",
        "## 4. Per-Run Results",
        "",
    ]

    # Section 4 Table
    if study.per_run_results:
        lines.extend([
            "| Run ID | Scenario | Metric | Value | Unit | Status | Sample Count (Valid/Missing) | Diagnostics |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for res in sorted(study.per_run_results, key=lambda x: (x.scenario_id, x.run_id, x.metric_id)):
            val_str = f"{res.value:.4f}" if (res.value is not None and math.isfinite(res.value)) else "UNAVAILABLE"
            samp_str = f"{res.valid_sample_count}/{res.missing_sample_count}"
            reason_str = res.reason or "None"
            lines.append(
                f"| `{res.run_id}` | `{res.scenario_id}` | `{res.metric_id}` | {val_str} | {res.unit} | {res.status} | {samp_str} | {reason_str} |"
            )
    else:
        lines.append("*No per-run metric results recorded.*")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Aggregate Metric Comparisons",
        "",
        generate_metric_comparison_table(study),
        "",
        "---",
        "",
        "## 6. Comparability Decisions",
        "",
    ])

    # Section 6 Table
    if study.comparability_results:
        lines.extend([
            "| Variant Scenario | Metric | Decision | Comparable? | Reason Codes | Caveats & Explanations |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for (v_id, m_id), c_res in sorted(study.comparability_results.items(), key=lambda x: (x[0][0], x[0][1])):
            is_comp_str = "YES" if c_res.is_comparable else "NO"
            reasons_str = ", ".join(c_res.reason_codes) if c_res.reason_codes else "None"
            caveats_str = "; ".join(c_res.caveats) if c_res.caveats else c_res.explanation
            lines.append(
                f"| `{v_id}` | `{m_id}` | **{c_res.decision.value}** | {is_comp_str} | `{reasons_str}` | {caveats_str} |"
            )
    else:
        lines.append("*No cross-scenario comparisons evaluated.*")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Scientific Interpretation",
        "",
        generate_scientific_interpretation(study),
        "",
        "---",
        "",
        "## 8. Unavailable Metrics and Reasons",
        "",
    ])

    # Section 8
    unavail_found = False
    for (s_id, m_id), agg in sorted(study.aggregates.items(), key=lambda x: (x[0][0], x[0][1])):
        if agg.unavailable_count > 0 or agg.mean is None:
            unavail_found = True
            reasons = "; ".join(agg.exclusion_reasons) if agg.exclusion_reasons else "Data absent from simulation outputs"
            lines.append(f"- **Metric `{m_id}` (Scenario `{s_id}`):** UNAVAILABLE. Reason: {reasons}")

    if not unavail_found:
        lines.append("*All evaluated metrics yielded valid numeric observations across all runs.*")

    lines.extend([
        "",
        "---",
        "",
        "## 9. Failed Runs and Reasons",
        "",
    ])

    # Section 9
    failed_runs = [r for r in study.run_records if r.run_status != RunStatus.SUCCESS]
    if failed_runs:
        for fr in failed_runs:
            lines.append(
                f"- **Run `{fr.run_id}` (`{fr.scenario_id}`):** Status `{fr.run_status.value}` at step {fr.final_step}. "
                f"Termination: `{fr.termination_reason.value}`. Reason: {fr.failure_reason or 'No diagnostic message'}"
            )
    else:
        lines.append("*All attempted simulation runs completed successfully with zero runtime failures.*")

    lines.extend([
        "",
        "---",
        "",
        "## 10. Reproducibility Metadata",
        "",
        f"- **Repetition Policy & Sample Type:** `{repro_meta['repetition_sample_type']}`",
        f"- **Seed Engine Usage:** `seed_used_by_engine={repro_meta['seed_used_by_engine']}`. *{repro_meta['seed_usage_note']}*",
        f"- **Scenario Configuration Fingerprints (SHA-256):**",
    ])
    for s_id, fp in repro_meta["scenario_config_fingerprints"].items():
        lines.append(f"  - `{s_id}`: `{fp}`")

    lines.extend([
        f"- **Notes:** {repro_meta['reproducibility_notes']}",
        "",
        "---",
        "",
        "## 11. Limitations & Boundaries",
        "",
        "- **Directly Measured vs. Derived:** Physical trajectories, step durations, and boundary collisions are measured directly; densities, clearance times, and mean velocities are mathematically derived.",
        "- **No Composite Scoring:** GenEvac-Sentinel strictly rejects artificial composite scoring, weighting schemes, or opaque ranking formulas.",
        "- **Zero Inferred Evacuations:** Agents missing from snapshot observations are marked `not_observed` and are never assumed evacuated.",
        "- **Simulation Engine Independence:** Person B analytics maintain a strictly read-only relationship with Pavan's simulation engine.",
    ])

    report_content = "\n".join(lines) + "\n"
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report_content, encoding="utf-8")

    return report_content


def export_all_experiment_artifacts(
    study: ComparisonStudyResult,
    output_dir: Union[Path, str],
) -> Dict[str, Path]:
    """Export all five standardized experiment artifacts into the specified directory.

    Artifacts:
    1. scenario_inventory.csv
    2. per_run_metrics.csv
    3. aggregate_comparison.csv
    4. experiment_result.json
    5. experiment_report.md

    Returns:
        Mapping from artifact identifier to its created Path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {
        "scenario_inventory_csv": out_path / "scenario_inventory.csv",
        "per_run_metrics_csv": out_path / "per_run_metrics.csv",
        "aggregate_comparison_csv": out_path / "aggregate_comparison.csv",
        "experiment_json": out_path / "experiment_result.json",
        "experiment_report_md": out_path / "experiment_report.md",
    }

    export_scenario_inventory_csv(study, artifacts["scenario_inventory_csv"])
    export_per_run_metrics_csv(study, artifacts["per_run_metrics_csv"])
    export_aggregate_comparison_csv(study, artifacts["aggregate_comparison_csv"])
    export_experiment_json(study, artifacts["experiment_json"])
    generate_experiment_report(study, artifacts["experiment_report_md"])

    return artifacts

