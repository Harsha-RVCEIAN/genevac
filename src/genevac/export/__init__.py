"""CSV and JSON exporters for trajectory results and simulation metrics."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Re-export Phase 07 experiment-level export functionality
from genevac.experiments.export import (
    export_aggregate_comparison_csv,
    export_all_experiment_artifacts,
    export_experiment_json,
    export_per_run_metrics_csv,
    export_scenario_inventory_csv,
    extract_reproducibility_metadata,
    generate_experiment_report,
)


def export_to_json(data: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_to_csv(rows: List[Dict[str, Any]], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "export_to_json",
    "export_to_csv",
    "export_scenario_inventory_csv",
    "export_per_run_metrics_csv",
    "export_aggregate_comparison_csv",
    "export_experiment_json",
    "extract_reproducibility_metadata",
    "generate_experiment_report",
    "export_all_experiment_artifacts",
]
