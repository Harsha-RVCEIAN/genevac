# Genevac

> AI-assisted pedestrian evacuation simulation, trajectory prediction, and performance analytics.

## Overview

**Genevac** is a modular simulation and analytics framework designed to model, predict, and analyze pedestrian dynamics during emergency evacuations. It integrates realistic physical environment modeling, rule/agent-based pedestrian behaviors, machine learning-driven trajectory prediction, and spatial-temporal evacuation analytics.

---

## Project Structure

```text
genevac/
├── src/genevac/              # Main application source
│   ├── simulation/           # Environment, pedestrian movement & simulation loop
│   ├── environment/          # Map, exits, obstacles & walkable areas
│   ├── pedestrians/          # Pedestrian state, attributes & behavior models
│   ├── prediction/           # AI prediction models & trajectory history buffer
│   ├── metrics/              # Evacuation performance & prediction accuracy metrics
│   ├── analytics/            # Congestion detection, statistics & heatmaps
│   ├── experiments/          # Scenario configurations, execution & batch runs
│   ├── visualization/        # Live rendering, trajectories, heatmaps & charts
│   ├── export/               # CSV / JSON trajectory and metric exporters
│   ├── config.py             # Global application configuration
│   └── main.py               # Main CLI and application entry point
├── tests/                    # Automated unit and integration tests
│   ├── test_simulation.py
│   ├── test_pedestrians.py
│   ├── test_prediction.py
│   ├── test_metrics.py
│   ├── test_heatmaps.py
│   ├── test_experiments.py
│   └── test_export.py
├── data/                     # Simulation and scenario data
│   ├── sample/               # Sample inputs and test fixtures
│   ├── scenarios/            # Evacuation scenario configuration files
│   └── processed/            # Derived outputs (gitignored)
├── docs/                     # Project documentation
│   ├── architecture.md       # High-level architecture & design
│   ├── data_model.md         # Data contracts for pedestrians and trajectories
│   ├── prediction.md         # AI prediction model specifications
│   ├── evaluation.md         # Evaluation protocols and metrics
│   ├── experiments.md        # Experimental methodology
│   └── setup.md              # Installation and development setup
├── scripts/                  # Developer and execution utilities
│   ├── run_simulation.py     # Execute a single simulation scenario
│   ├── run_experiments.py    # Batch experiment runner
│   └── evaluate_predictions.py # Evaluate prediction model accuracy
├── pyproject.toml            # Project packaging specification
├── requirements.txt          # Python dependencies
├── .env.example              # Template environment variables
└── .gitignore                # Git ignore rules
```

---

## Quickstart

### 1. Installation

```bash
# Clone and enter the repository
git clone <repo-url>
cd genevac

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

### 2. Run a Simulation

```bash
python scripts/run_simulation.py --scenario data/scenarios/sample_hallway.json
```

### 3. Run Experiments & Evaluations

```bash
python scripts/run_experiments.py
python scripts/evaluate_predictions.py
```

### 4. Run Tests

```bash
pytest
```
GenEvac-Sentinel — Module B Person B Implementation Plan
This folder contains the staged implementation plan for Person B / Module B Part 2.
Ownership
Pavan / Person A in the supplied schedule: Module B Part 1 — simulation engine.
Harsha / Person B in the supplied schedule: Module B Part 2 — metrics, heatmaps, charts, exports, scenario comparisons, baseline trajectory prediction, MID investigation, experiments, evaluation, and research presentation.
Module A: owned by other teammates. It is outside this implementation scope.
How to use these files
Read `00_EXECUTION_ORDER.md` first. It defines the phase order and approval gates. Then use the corresponding phase document. Do not ask the agent to execute all phases in one pass.
The plan is designed to audit and reuse the current repository implementation first. Existing prediction/evaluation code must not be rewritten merely to match a new document.
Important evidence rule
Earlier project notes reported a 158-test passing result and described existing prediction files. Those are historical reports, not proof about the current checkout. Antigravity must inspect the actual branch, source, tests, and Pavan's current public simulation API before making claims.
Suggested working arrangement
Work on Harsha's own branch/worktree.
Treat `pavan-part1-reference/`, if present, as read-only.
Do not merge, rebase, cherry-pick, commit, push, or modify another person's branch without explicit approval.
Stop at each phase gate and wait for Harsha's approval.