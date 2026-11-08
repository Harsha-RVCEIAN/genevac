# GenEvac-Sentinel: Documentation Index

Welcome to the comprehensive documentation suite for **GenEvac-Sentinel** (Person A implementation).

This index organizes all architecture specifications, domain models, simulation mechanics, and interface contracts for developers, researchers, and downstream integration partners.

---

## 1. Documentation Navigator

| Document | Target Audience | Summary |
| :--- | :--- | :--- |
| **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** | All Audiences | Educational prototype purpose, scope boundaries, Person A vs Person B ownership, and high-level architecture. |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Developers & Architects | Module organization, dependency flow, immutability guarantees, and decoupling principles. |
| **[BUILDING_SPECIFICATION.md](BUILDING_SPECIFICATION.md)** | Simulation & Environment Engineers | Exact 2D building geometry, coordinates, walls, doorways, exits, obstacles, and JSON schemas. |
| **[SIMULATION_ENGINE.md](SIMULATION_ENGINE.md)** | Simulation Engineers | Simulation clock, lifecycle, 17-angle deflection kinematics, A* navigation, and exit crossing dynamics. |
| **[PEDESTRIAN_MODEL.md](PEDESTRIAN_MODEL.md)** | Simulation & ML Engineers | Pedestrian domain models, physical radius, status lifecycle transitions, FIFO trajectory history, and population registry. |
| **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** | UI Developers & Users | Pygame visualization app, viewport coordinate scaling, interactive keybindings, rendering layers, and headless testing. |
| **[SIMULATION_OBSERVATION_CONTRACT.md](SIMULATION_OBSERVATION_CONTRACT.md)** | Person B & Integration Engineers | Complete schema specification for observation snapshots, field definitions, SI units, and immutability guarantees. |
| **[PERSON_B_INTEGRATION_GUIDE.md](PERSON_B_INTEGRATION_GUIDE.md)** | Person B (Prediction / AI Engineers) | Integration workflow, observation ingestion examples, baseline predictor sample script, and integration checklist. |
| **[PERSON_B_HANDOVER.md](PERSON_B_HANDOVER.md)** | Person B & Technical Leads | Authoritative handover package: ownership boundaries, exact verified snapshot contract, runnable blueprint, evaluation guidelines, and verification commands. |
| **[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)** | Project Leads & Auditors | Educational prototype notice, documented accepted limitations (deadlocks, lack of run_id, upstream warning), and verified capabilities. |

---

## 2. Quick Navigation by Role

### For Person B (ML / Prediction Engineers)
Begin with:
1. **[PERSON_B_HANDOVER.md](PERSON_B_HANDOVER.md)** — Master handover specification, boundaries, verified observation contract, and execution demo (`examples/person_b_integration_demo.py`).
2. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** — High-level architecture, ownership scope, and simulation truth invariants.
3. **[SIMULATION_OBSERVATION_CONTRACT.md](SIMULATION_OBSERVATION_CONTRACT.md)** — Exhaustive field definitions, SI units, and immutability guarantees.
4. **[PERSON_B_INTEGRATION_GUIDE.md](PERSON_B_INTEGRATION_GUIDE.md)** — Practical ingestion workflow, constant-velocity baseline example, and checklist.

### For Simulation Core Developers
Begin with:
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — Architectural patterns and module structure.
2. **[SIMULATION_ENGINE.md](SIMULATION_ENGINE.md)** — Step execution flow, routing, and collision kinematics.
3. **[PEDESTRIAN_MODEL.md](PEDESTRIAN_MODEL.md)** — Agent state management and registry invariants.
4. **[BUILDING_SPECIFICATION.md](BUILDING_SPECIFICATION.md)** — Environment coordinate space and wall definitions.

### For Visualization & Testing Engineers
Begin with:
1. **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** — Running GUI visualization, keybindings, and headless testing modes.
2. **[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)** — Accepted edge-case behaviors and test fixture guidelines.

