"""Presentation, table formatting, and interpretation reporting for Phase 06."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from genevac.experiments.runner import ComparisonStudyResult


def generate_scenario_inventory_table(study: ComparisonStudyResult) -> str:
    """Generate a clean Markdown table summarizing all scenarios in the study."""
    lines = [
        "| Scenario ID | Name (Ver) | Role | Agents | Environment | Key Configuration | Fingerprint | Runs (Succ/Att) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    base_id = study.comparison.baseline_scenario_id

    for s_id, sc in sorted(study.scenarios.items()):
        role = "**BASELINE**" if s_id == base_id else "VARIANT"
        w = sc.engine_config.get("width", 50.0)
        h = sc.engine_config.get("height", 20.0)
        env_str = f"{w:.1f}m × {h:.1f}m"
        fp_short = sc.compute_fingerprint()[:10]

        # Count runs
        runs = [r for r in study.run_records if r.scenario_id == s_id]
        succ = sum(1 for r in runs if r.run_status.value == "SUCCESS")
        att = len(runs)
        run_str = f"{succ}/{att}"

        key_diffs = []
        if sc.engine_config.get("obstacles"):
            key_diffs.append(f"{len(sc.engine_config['obstacles'])} obstacles")
        if sc.engine_config.get("exits"):
            key_diffs.append(f"{len(sc.engine_config['exits'])} exits")
        if not key_diffs:
            key_diffs.append("standard open layout")
        diff_str = ", ".join(key_diffs)

        lines.append(
            f"| `{sc.scenario_id}` | {sc.name} ({sc.version}) | {role} | {sc.agent_count} | {env_str} | {diff_str} | `{fp_short}` | {run_str} |"
        )

    return "\n".join(lines)


def generate_metric_comparison_table(study: ComparisonStudyResult) -> str:
    """Generate a comparative metric summary table with baseline deltas and comparability audits."""
    base_id = study.comparison.baseline_scenario_id
    lines = [
        "| Metric | Unit | Baseline Mean | Variant Scenario | Variant Mean | Absolute Delta (Δ) | Relative Delta (%Δ) | Comparability | Caveats / Diagnostics |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for m_id in study.comparison.dependent_metrics:
        base_agg = study.aggregates.get((base_id, m_id))
        if base_agg is None or base_agg.mean is None:
            base_mean_str = "UNAVAILABLE"
        else:
            if base_agg.std_dev is not None:
                base_mean_str = f"{base_agg.mean:.3f} ± {base_agg.std_dev:.3f}"
            else:
                base_mean_str = f"{base_agg.mean:.3f} (N=1)"

        for v_id in study.comparison.scenario_ids:
            if v_id == base_id:
                continue

            var_agg = study.aggregates.get((v_id, m_id))
            delta = study.metric_deltas.get((v_id, m_id))
            comp_res = study.comparability_results.get((v_id, m_id))

            if var_agg is None or var_agg.mean is None:
                var_mean_str = "UNAVAILABLE"
            else:
                if var_agg.std_dev is not None:
                    var_mean_str = f"{var_agg.mean:.3f} ± {var_agg.std_dev:.3f}"
                else:
                    var_mean_str = f"{var_agg.mean:.3f} (N=1)"

            if delta and delta.absolute_delta is not None:
                abs_str = f"{delta.absolute_delta:+.3f}"
                pct_str = f"{delta.percentage_delta:+.1f}%" if delta.percentage_delta is not None else "N/A"
            else:
                abs_str = "N/A"
                pct_str = "N/A"

            comp_str = comp_res.decision.value if comp_res else "UNKNOWN"
            caveats_str = "; ".join(comp_res.caveats) if comp_res and comp_res.caveats else (comp_res.explanation if comp_res else "")

            unit_str = base_agg.unit if base_agg else ""
            lines.append(
                f"| `{m_id}` | {unit_str} | {base_mean_str} | `{v_id}` | {var_mean_str} | {abs_str} | {pct_str} | **{comp_str}** | {caveats_str} |"
            )

    return "\n".join(lines)


def generate_per_run_detail_table(study: ComparisonStudyResult) -> str:
    """Generate a detailed audit table of all individual simulation executions."""
    lines = [
        "| Run ID | Scenario ID | Final Step | Simulated Duration | Run Status | Termination Reason | Failure Details |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in sorted(study.run_records, key=lambda x: x.run_id):
        fail = r.failure_reason if r.failure_reason else "None"
        lines.append(
            f"| `{r.run_id}` | `{r.scenario_id}` | {r.final_step} | {r.simulated_duration:.1f}s | {r.run_status.value} | {r.termination_reason.value} | {fail} |"
        )

    return "\n".join(lines)


def generate_scientific_interpretation(study: ComparisonStudyResult) -> str:
    """Generate an objective, scientifically disciplined interpretation report."""
    comp = study.comparison
    base_id = comp.baseline_scenario_id

    sections = [
        "### Scientific Interpretation & Study Limitations",
        f"- **Research Question:** {comp.research_question}",
        f"- **Independent Variable:** `{comp.independent_variable}`",
        f"- **Controlled Factors:** {', '.join(f'`{k}={v}`' for k, v in comp.controlled_factors.items()) if comp.controlled_factors else 'None declared'}",
        "",
        "#### 1. Observed Differences:",
    ]

    for (v_id, m_id), delta in sorted(study.metric_deltas.items()):
        if delta.absolute_delta is not None:
            direction = "increased" if delta.absolute_delta > 0 else "decreased"
            sections.append(
                f"- For scenario `{v_id}`, `{m_id}` {direction} by {abs(delta.absolute_delta):.3f} {study.aggregates[(v_id, m_id)].unit} "
                f"({delta.percentage_delta:+.1f}%) compared to baseline `{base_id}`."
            )
        else:
            sections.append(
                f"- For scenario `{v_id}`, `{m_id}` delta could not be computed (decision: {delta.comparability.decision.value})."
            )

    sections.extend([
        "",
        "#### 2. Plausible Explanations & Confounders:",
        "- Differences in clearance time and crowd density reflect physical changes in layout geometry and agent kinematic bottlenecks.",
        "- Because the simulation dynamics are deterministic, variation across runs with identical initial conditions is 0.0 unless stochastic initial agent generators are used.",
        "",
        "#### 3. What Cannot Be Concluded:",
        "- **No Composite Ranking:** Scenarios are not ranked using composite weighted scores or artificial indexes.",
        "- **No Unverified Causal Attribution:** In multi-factor exploratory scenarios, observed changes cannot be attributed solely to a single parameter without isolating each factor independently.",
        "- **ERSI / Delta Score:** Intentionally excluded due to absence of authoritative mathematical specifications in repository scope.",
    ])

    return "\n".join(sections)
