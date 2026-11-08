# GenEvac-Sentinel

An educational 2D continuous evacuation simulation and trajectory prediction research platform.

---

## Current Status: Person A Complete — Ready for Person B Handover

All four Person A development and verification phases are completed and fully verified:
- **Phase 1: Codebase Audit and Stabilization** — Completed & Approved
- **Phase 2: End-to-End Simulation Verification** — Completed & Approved
- **Phase 3: Documentation and Interface Finalization** — Completed & Approved
- **Phase 4: Final Packaging and Person B Handover** — Completed & Verified (286/286 tests passing)

---

## Documentation Suite

The project includes an exhaustive technical documentation suite under [`docs/`](docs/):

| Document | Description |
| :--- | :--- |
| **[Person B Handover Specification](docs/PERSON_B_HANDOVER.md)** | **Primary Handover Blueprint**: Ownership boundaries, verified observation schema, runnable blueprint, evaluation guidelines, and verification commands. |
| **[Documentation Master Index](docs/DOCUMENTATION_INDEX.md)** | Central navigation index across all 11 architectural, physical, and interface documents. |
| **[Project Overview](docs/PROJECT_OVERVIEW.md)** | High-level system purpose, educational prototype scope, sequence diagrams, and module responsibilities. |
| **[Architecture Specification](docs/ARCHITECTURE.md)** | Modular design, unidirectional data flow, class hierarchies, and immutability guarantees. |
| **[Building Specification](docs/BUILDING_SPECIFICATION.md)** | Authoritative $40\times 30\text{ m}$ coordinates, 4 rooms, central corridor, 4 doorways, 3 exits, and 4 static obstacles. |
| **[Simulation Engine Mechanics](docs/SIMULATION_ENGINE.md)** | Drift-free clock, two-tiered routing (canonical + visibility graph A*), 17-angle deflection kinematics, and exit crossing dynamics. |
| **[Pedestrian Model](docs/PEDESTRIAN_MODEL.md)** | `PedestrianState`, discrete statuses (`ACTIVE`, `EVACUATED`, `BLOCKED`, `INACTIVE`), and rolling FIFO trajectory history ($40.0\text{ s}$). |
| **[Simulation Observation Contract](docs/SIMULATION_OBSERVATION_CONTRACT.md)** | Exact snapshot schema, field names, SI units, top-left coordinate frame, and deep immutability. |
| **[Person B Integration Guide](docs/PERSON_B_INTEGRATION_GUIDE.md)** | Practical snapshot ingestion workflow, constant-velocity baseline predictor example, and FDE evaluation. |
| **[Visualization Guide](docs/VISUALIZATION_GUIDE.md)** | Pygame top-down display, aspect-ratio preserving viewport, interactive controls, and headless execution. |
| **[Known Limitations](docs/KNOWN_LIMITATIONS.md)** | Educational prototype notice, narrow-passage standstills, missing `run_id`, upstream Pygame warning, and flat 2D scope. |

---

## Quick Start

### 1. Environment Setup
```bash
# Clone or enter repository
cd genevac

# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install runtime and testing dependencies
pip install -r requirements-dev.txt
```

### 2. Run Test Suite
```bash
# Execute the full automated test suite (286 tests)
pytest -v
```

### 3. Launch 2D Visualization
```bash
# Launch default 40x30m building with 8 active evacuees
PYTHONPATH=src python -m genevac.visualization.app

# Launch loaded from a custom building JSON map
PYTHONPATH=src python -m genevac.visualization.app --json maps/building_main.json

# Specify custom window size and physics timestep
PYTHONPATH=src python -m genevac.visualization.app --width 1400 --height 900 --dt 0.05
```

#### Interactive Keyboard Controls
- **`SPACE`**: Pause / Resume simulation clock and movement.
- **`R`**: Deterministic reset (re-instantiates fresh population cohort).
- **`ESC` / `Q`**: Cleanly exit application.

### 4. Run Headless Smoke Test (CI/CD)
```bash
SDL_VIDEODRIVER=dummy PYTHONPATH=src python -c "
from genevac.visualization.app import SimulationApp
app = SimulationApp(width=800, height=600)
app.run(max_frames=30)
print('Headless OK')
"
```

---

## Architecture Summary

```
genevac/
├── maps/                   # JSON floor plans and layout configurations (schema "1.0")
├── src/genevac/
│   ├── environment/        # 2D Geometry, WalkableAreas, Walls, Doors, Exits, Obstacles, JSON Serialization
│   ├── pedestrian/         # PedestrianState, Velocity2D, TrajectoryHistory, PedestrianRegistry
│   ├── simulation/         # SimulationEngine, SimulationClock, Movement, Events, Observation Snapshots
│   └── visualization/      # SimulationApp, Viewport, EnvironmentRenderer (Pygame)
└── tests/                  # Automated pytest suite (286 tests across 12 test modules)
```

---

## Simulation-to-Prediction Interface (Person B)

Person B consumes deeply frozen snapshots (`SimulationObservationSnapshot`) emitted by Person A's engine at **$2.5\text{ Hz}$** ($0.4\text{ s}$ cadence).

### Minimal Ingestion Pattern
```python
from genevac.environment.builder import create_initial_building
from genevac.pedestrian.models import PedestrianState, Point
from genevac.simulation.engine import SimulationEngine

# 1. Create environment and initial population cohort
building = create_initial_building()
cohort = [
    PedestrianState(id="agent_1", position=Point(8.0, 6.0), current_area_id="room_1", target_exit_id="exit_left", preferred_speed=1.3)
]

# 2. Pass cohort into SimulationEngine constructor (runs validation and routing)
engine = SimulationEngine(building=building, registry=cohort, dt=0.1, observation_interval=0.4)

# 3. Advance simulation and consume snapshots
for step in range(40):
    snapshot = engine.step()
    if snapshot is not None:
        print(f"Observation tick {snapshot.step_count} (t={snapshot.timestamp:.2f}s): {snapshot.active_count} active agents")
        for ped in snapshot.pedestrians:
            print(f"  {ped.pedestrian_id}: pos=({ped.x:.2f}, {ped.y:.2f}), vel=({ped.vx:.2f}, {ped.vy:.2f})")
```

For full details, consult [`docs/PERSON_B_HANDOVER.md`](docs/PERSON_B_HANDOVER.md).