# System Architecture

## 1. High-Level Overview

Genevac is designed with a modular, decoupled architecture where each package encapsulates a distinct responsibility in the simulation and evaluation lifecycle.

```text
+--------------------------------------------------------------+
|                        CLI / Scripts                         |
|   (main.py, run_simulation.py, run_experiments.py)          |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|                      Simulation Engine                       |
|   - Environment (Walkable areas, Obstacles, Exits)           |
|   - Pedestrians (Position, Velocity, Social Forces/Behaviors)|
|   - Simulation Loop (Timestepping, Collision Resolution)     |
+---------------+------------------------------+---------------+
                |                              |
                v                              v
+---------------+--------------+  +------------+---------------+
|     Prediction Engine        |  |     Analytics & Metrics    |
| - Trajectory History Buffer  |  | - Clearance Time & Flow   |
| - AI Predictors (ML/DL/Heur) |  | - Congestion & Density    |
| - Multi-step Forecasts       |  | - Spatial Heatmaps        |
+---------------+--------------+  +------------+---------------+
                |                              |
                +---------------+--------------+
                                |
                                v
+-------------------------------+------------------------------+
|                   Visualization & Export                     |
|  - Real-time 2D View  - Trajectory Plots  - CSV/JSON Export   |
+--------------------------------------------------------------+
```

## 2. Core Modules

1. **`simulation/`**: Controls the discrete-time update loop. Integrates forces, updates pedestrian states, and interacts with the physical layout.
2. **`environment/`**: Defines geometric boundaries, barriers/obstacles, doorways, and destination exits.
3. **`pedestrians/`**: Implements individual agents, speed profiles, reaction times, stress parameters, and behavioral steering.
4. **`prediction/`**: Stores historical trajectory snippets and predicts future agent trajectories using AI models.
5. **`metrics/`**: Calculates evacuation performance (clearance time, flow rate) and prediction fidelity (ADE, FDE).
6. **`analytics/`**: Computes spatial grid density, bottleneck detection, and heatmaps.
7. **`visualization/`**: Generates visual representations of simulation states and analytics.
8. **`export/`**: Serializes simulation runs into standardized CSV and JSON formats.
