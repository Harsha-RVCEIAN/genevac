"""Experiment configuration schema validation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_experiment_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate an experiment configuration dictionary against required criteria.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        (is_valid, list of error messages).
    """
    errors: List[str] = []

    if not isinstance(config, dict):
        return False, ["Configuration root must be a dictionary."]

    exp_id = config.get("experiment_id") or config.get("name")
    if not exp_id or not str(exp_id).strip():
        errors.append("experiment_id or name must be a non-empty string.")

    repetitions = config.get("repetitions", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        errors.append(f"repetitions must be a positive integer, got {repetitions!r}")

    scenarios = config.get("scenarios")
    if scenarios is not None:
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            errors.append("scenarios must be a non-empty list of scenario dicts.")

    base_seed = config.get("base_seed", 42)
    if not isinstance(base_seed, int) or base_seed < 0:
        errors.append(f"base_seed must be a non-negative integer, got {base_seed!r}")

    return (len(errors) == 0, errors)

