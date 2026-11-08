# GenEvac-Sentinel: Visualization Architecture and User Guide

## 1. Overview and Purpose

The visualization subsystem (`src/genevac/visualization/`) provides a 2D top-down graphical presentation of the GenEvac-Sentinel simulation. Built on Pygame, it renders building floor plans, architectural boundaries, obstacles, doorways, emergency exits, and live pedestrian agents.

The visualization architecture maintains a clear separation between rendering and execution control:
- **Rendering Layer (`EnvironmentRenderer`, `Viewport`)**: Strictly **read-only**. It queries `SimulationEngine` and `Building` models purely to draw frames and display HUD metrics. It cannot mutate simulation state or influence pedestrian physics.
- **Application Controller Layer (`SimulationApp`)**: Manages the Pygame event loop, window display, and interactive user inputs. It legitimately controls the simulation execution lifecycle by invoking public engine methods (`engine.step()`, pause toggling, and deterministic reset via population recreation).

---

## 2. Visualization Components

The package consists of three primary modules:

```mermaid
flowchart TD
    subgraph Controller["Application Controller Layer"]
        App[SimulationApp<br/>src/genevac/visualization/app.py]
    end

    subgraph SimulationCore["Authoritative Simulation State"]
        Engine[SimulationEngine]
    end

    subgraph ReadOnlyView["Read-Only Rendering Layer"]
        Viewport[Viewport<br/>src/genevac/visualization/coordinates.py]
        Renderer[EnvironmentRenderer<br/>src/genevac/visualization/renderer.py]
    end

    App -->|Drives physics via engine.step() / reset| Engine
    App --> Viewport
    App --> Renderer
    Renderer -->|Queries coordinates & bounds| Viewport
    Renderer -.->|Reads state (read-only)| Engine
```

### 2.1 Viewport Coordinate Transformation (`coordinates.py`)
- Maps 2D continuous world coordinates (metres, top-left origin) to screen pixel coordinates.
- **Aspect Ratio Preservation**: Computes `scale = min(scale_x, scale_y)` so physical dimensions are never stretched or distorted.
- **Centering & Margins**: Centers the building within the available window, applying a uniform margin ($50\text{px}$) and dedicated header ($60\text{px}$) and footer ($50\text{px}$) offsets.
- **Methods**:
  - `world_to_screen_point(pt) -> (sx, sy)`
  - `world_to_screen_rect(rect) -> (sx, sy, sw, sh)`
  - `world_to_screen_dim(meters) -> pixels`
  - `screen_to_world_point((sx, sy)) -> Point(wx, wy)`

### 2.2 Environment Renderer (`renderer.py`)
Renders the complete architectural scene and simulation actors using a curated high-contrast architectural palette:
- **Floor & Corridors**: Light room floors (`#EBF0F6`) with bright corridors (`#F8FAFC`).
- **Walls**: Solid exterior walls (`#1E293B`) and interior partitions (`#334155`) with physical $0.2\text{m}$ thickness and openings for doors/exits.
- **Obstacles**: Rich amber warning blocks (`#D97706`) with darker structural borders.
- **Portals**: Vibrant cyan for interior doorways (`#0EA5E9`) and high-visibility emerald green for safety exits (`#22C55E`).
- **Pedestrians**:
  - `ACTIVE`: Electric cyan circle (`#38BDF8`) with white core dot and velocity vector heading arrow.
  - `EVACUATED`: Emerald green circle (`#22C55E`).
  - `BLOCKED`: Coral red circle (`#EF4444`) with dark red outline (`#991B1B`).
  - `INACTIVE`: Slate grey circle (`#94A3B8`).
  - **ID Badges**: Dark badge pill (`#0F172A`) with light text indicating the pedestrian's stable ID.
- **Simulation HUD**: Top banner displaying real-time metrics (Elapsed Time, Timestep, Step Counter, Active/Evacuated/Blocked counts, and Engine Status).

### 2.3 Simulation Application (`app.py`)
The interactive orchestrator managing the Pygame lifecycle, event pump, wall-clock timing, and simulation execution:
- **Stepping**: Drives physics advances by invoking `engine.step()` inside the fixed-timestep accumulator loop.
- **Pause & Resume**: Intercepts keyboard events (`SPACE`) to halt/resume the fixed-timestep accumulation without losing residual delta time.
- **Deterministic Reset**: Intercepts keyboard events (`R`) to instantiate fresh, unmutated pedestrian states via `_population_factory()` and reinitialize a clean `SimulationEngine` instance.
- **Read-Only Rendering Invocation**: Passes the engine state to `EnvironmentRenderer.render()` for display on the Pygame surface.

---

## 3. Decoupled Fixed-Timestep Accumulator

To prevent rendering frame rate variations from corrupting deterministic simulation physics, `SimulationApp` implements a decoupled accumulator loop:

1. **Delta Clamping**: Clamps frame delta $\Delta t_{\text{wall}} \le 0.25\text{ s}$ to prevent "spiral of death" lag spikes.
2. **Accumulation**: `sim_accumulator += dt_wall`.
3. **Step Catch-Up Ceiling**: Executes at most 5 simulation steps ($\Delta t = 0.1\text{ s}$) per rendered frame:
   ```python
   steps_taken = 0
   while self._sim_accumulator >= self._dt and steps_taken < self._max_catchup_steps:
       self._engine.step()
       self._sim_accumulator -= self._dt
       steps_taken += 1
   ```
4. **Residual Time Retention**: Fractional leftover time remains in the accumulator for the next frame.

---

## 4. User Controls & Keybindings

When running the interactive visualization window:

| Key | Action | Description |
| :--- | :--- | :--- |
| `SPACE` | **Pause / Resume** | Toggles simulation clock stepping on/off. HUD updates to display `PAUSED`. |
| `R` | **Reset Simulation** | Re-instantiates a fresh population cohort via `population_factory` and resets the engine deterministically to step 0. |
| `ESC` / `Q` | **Quit** | Gracefully closes the Pygame window and terminates the application. |
| Window Close | **Quit** | Closes the application via standard OS window close button. |

---

## 5. Execution Modes and Commands

### 5.1 Interactive GUI Window

Run the simulation visualization using the virtual environment Python:

```bash
# Launch default 40x30m building with 8 pedestrians
.venv/bin/python3 -m genevac.visualization.app

# Launch with custom window dimensions and custom timestep
.venv/bin/python3 -m genevac.visualization.app --width 1400 --height 900 --dt 0.05

# Launch with a custom building JSON model
.venv/bin/python3 -m genevac.visualization.app --json path/to/building.json
```

### 5.2 Headless Execution (Automated Testing / CI)

For continuous integration, Docker containers, or environments without a hardware display, Pygame can be run headlessly using the `dummy` SDL video driver:

```bash
# Run headless via environment variable
SDL_VIDEODRIVER=dummy .venv/bin/python3 -m genevac.visualization.app --steps 100
```

Programmatic headless execution in Python:

```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from genevac.visualization.app import SimulationApp

app = SimulationApp(width=800, height=600)
app.run(max_frames=60)  # Runs 60 frames headlessly and exits cleanly
```

---

## 6. Rendering Performance and Safety Guarantees

1. **Non-blocking Rendering**: Rendering queries engine state via immutable getters and registry iterators.
2. **Crash Resilience**: Font initialization is guarded with fallbacks (`pygame.font.SysFont` $\to$ default font).
3. **Deterministic Reset**: Pressing `R` invokes `_population_factory()` rather than cloning mutated entities, guaranteeing identical replay trajectories across resets.

