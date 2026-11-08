# GenEvac-Sentinel Test Suite (`tests/`)

This directory contains the comprehensive automated test suite for Person A's implementation of GenEvac-Sentinel.

## Test Suite Overview

The test suite contains **286 test cases** with 100% passing status across Python 3.11:

| Test File | Focus & Coverage |
| :--- | :--- |
| `test_environment.py` | Geometry primitives, line intersection, building validation, and door/exit constraints. |
| `test_environment_serialization.py` | JSON round-trip serialization, schema versioning (`"1.0"`), and boundary handling. |
| `test_pedestrian_models.py` | Pedestrian state invariants, physical radius clearance, trajectory history FIFO queues, and status enums. |
| `test_pedestrian_registry.py` | Stable IDs, duplicate rejection, deterministic insertion-order iteration, and dynamic count synchronization. |
| `test_simulation_clock.py` | Integer step progression, drift-free timestamp derivation, and observation stride checks. |
| `test_simulation_engine.py` | Simulation lifecycle, initialization clearance checks, pause/resume, reset determinism, and exit closure events. |
| `test_simulation_events.py` | Structured discrete simulation events (`exit_closed`, `no_route`, `evacuation`). |
| `test_simulation_movement.py` | Visibility graph generation, A* search, 17-angle deflection kinematics, continuous collision avoidance, and deadlock safe halting. |
| `test_simulation_observation.py` | Immutable observation snapshots, frozen dataclass guarantees, field types, and SI units. |
| `test_smoke.py` | Package import verification and smoke testing. |
| `test_visualization.py` | Viewport scaling, aspect ratio preservation, coordinate transforms, and read-only rendering calls. |
| `test_integration.py` | Multi-agent end-to-end evacuation scenarios, corridor congestion, obstacle avoidance, and exit closures. |

## Running Tests

### Standard Test Execution
```bash
# Run all tests
.venv/bin/pytest

# Run with verbose output
.venv/bin/pytest -v
```

### Running Specific Test Categories
```bash
# Run simulation engine and movement tests
.venv/bin/pytest tests/test_simulation_engine.py tests/test_simulation_movement.py

# Run environment tests
.venv/bin/pytest tests/test_environment.py tests/test_environment_serialization.py

# Run end-to-end integration tests
.venv/bin/pytest tests/test_integration.py
```

