---
trigger: always_on
---

# GenEvac-Sentinel Core Project Rules

## Project ownership

- Pavan owns **Module B Part 1: the simulation engine**.
- Harsha owns **Module B Part 2: metrics, heatmaps, charts, exports, scenario comparisons, baseline trajectory prediction, MID investigation, experiments, evaluation, integration, and research presentation**.
- Module A is owned by other teammates and is outside this agent's implementation scope.
- Do not confuse Module B Part 2 analytics/prediction work with ownership of the simulation engine.

## Scope control

- Work only on the task and phase explicitly approved by Harsha.
- Do not expand scope or implement adjacent modules “for completeness.”
- Do not alter another teammate's files or public interfaces without approval.
- Treat `pavan-part1-reference/`, if present, as read-only reference material.

## Existing implementation first

Prior notes identify these candidate files; verify they exist and inspect them before relying on them:
- `src/genevac/prediction/data_contract.py`
- `src/genevac/prediction/__init__.py`
- `src/genevac/prediction/adapter.py`
- `src/genevac/prediction/baseline.py`
- `src/genevac/prediction/dataset_adapter.py`
- `src/genevac/prediction/evaluation.py`

Reuse working code. Do not rewrite, duplicate, rename, or delete it merely to match a proposed architecture. Historical reports and test counts are not current verification.

## Truthfulness

Separate verified facts, assumptions, and unresolved questions. Cite exact repository paths/symbols for technical claims. Never claim code was run, tests passed, or integration worked unless actually executed in the current environment.
