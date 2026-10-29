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
