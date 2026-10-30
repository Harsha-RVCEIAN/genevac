# Phase A: Architecture Reconnaissance & Boundary Verification

## 1. Verified Working-Tree State

- **Repository Root**: `C:\Users\harsh\projects\genevac`
- **Active Git Branch**: `harsha`
- **Working-Tree Status**: Clean, up to date with `origin/harsha`.
- **Target Branch Availability**: Confirmed (`harsha` is checked out; remote branch `origin/harsha` exists).
- **Existing User Commits Preserved**: Commit `f6eaed5` ("inital state before building") includes raw datasets under `data/raw/` and research materials under `research/`.

---

## 2. Answers to the 20 Architecture Questions

### Q1. Where is `SimulationEngine` defined?
- **File**: [`src/genevac/simulation/__init__.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py#L8-L34)
- **Class**: [`SimulationEngine`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py#L8)

### Q2. How does the simulation advance?
- **Mechanism**: Via [`SimulationEngine.step()`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py#L21-L24). Each invocation increments `self.current_step` by 1 and iterates over `self.pedestrians`, calling [`ped.update_position(self.timestep)`](file:///c:/Users/harsh/projects/genevac/src/genevac/pedestrians/__init__.py#L24-L29).
- **Full Run**: [`SimulationEngine.run()`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py#L31-L34) loops `while not self.is_finished(): self.step()`.

### Q3. How is simulation time represented?
- **Internal Representation**: Discretized step count `self.current_step: int` (starts at `0`).
- **Real Time**: Calculated as $t = \text{current\_step} \times \text{timestep}$ (seconds), as reflected in [`calculate_clearance_time()`](file:///c:/Users/harsh/projects/genevac/src/genevac/metrics/__init__.py#L24-L26).

### Q4. Is the timestep configurable? What is its actual default?
- **Configurability**: Yes, passed as parameter `timestep: float` in [`SimulationEngine.__init__`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py#L9) and configured in [`SimulationConfig.timestep`](file:///c:/Users/harsh/projects/genevac/src/genevac/config.py#L10).
- **Default Value**: `0.1` seconds (10 Hz).

### Q5. How is the step counter updated?
- **Code**: `self.current_step += 1` inside [`SimulationEngine.step()`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py#L22).

### Q6. What attributes represent pedestrian identity, position, and velocity?
- **Class**: [`Pedestrian`](file:///c:/Users/harsh/projects/genevac/src/genevac/pedestrians/__init__.py#L15-L29) in `src/genevac/pedestrians/__init__.py`.
- **Identity**: `id: str`
- **Position**: `position: Tuple[float, float]`
- **Velocity**: `velocity: Tuple[float, float]`
- **Additional Attributes**: `desired_speed: float = 1.34` (m/s), `radius: float = 0.3` (m), `target_exit: str = ""`, `status: PedestrianStatus = PedestrianStatus.ACTIVE`.

### Q7. What coordinate system is used?
- **Format**: 2D Cartesian coordinates `(x, y)` in continuous space.
- **Update Rule**: `new_x = position[0] + velocity[0] * dt`, `new_y = position[1] + velocity[1] * dt`.

### Q8. Are positions documented as metres, pixels, or another unit?
- **Documentation**: Documented as **meters** in [`docs/data_model.md`](file:///c:/Users/harsh/projects/genevac/docs/data_model.md#L8) (`"Cartesian coordinates (x, y) in meters"`) and in visualization labels (`"X (m)"`, `"Y (m)"` in [`src/genevac/visualization/__init__.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/visualization/__init__.py#L14-L15)).

### Q9. Which units are documented, and which are actually enforced by code?
- **Documented**: Position (meters), velocity (meters/second), time (seconds), radius (meters).
- **Enforced by Code**: **None**. The physics update uses raw arithmetic on Python floats. There is no runtime unit validation, dimensional check, or type-safe quantity wrapper.

### Q10. Does a trajectory-history implementation already exist?
- **Implementation**: Yes, [`TrajectoryHistoryBuffer`](file:///c:/Users/harsh/projects/genevac/src/genevac/prediction/__init__.py#L7-L18) in `src/genevac/prediction/__init__.py`.

### Q11. How much history does it retain?
- **Length**: Configured by `window_size: int = 20` (default 20 timesteps). At $dt = 0.1$ s, this represents 2.0 seconds of history.
- **Storage**: Managed using `collections.deque(maxlen=self.window_size)`.

### Q12. How are observations sampled or appended?
- **Method**: [`record(agent_id: str, position: Tuple[float, float])`](file:///c:/Users/harsh/projects/genevac/src/genevac/prediction/__init__.py#L11-L14). Appends directly to the agent's deque.

### Q13. Are timestamps stored, or only positions?
- **Storage**: **Only positions** `(x, y)` are currently stored in `TrajectoryHistoryBuffer`. Timestamps, simulation steps, and observation validity flags are **not** stored.

### Q14. How are pedestrians removed or identified across simulation steps?
- **Identification**: Identified by persistent string `id`.
- **Removal**: In [`SimulationEngine`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py), pedestrians are stored in `self.pedestrians: List[Pedestrian]`. Agents are **not** popped or deleted from the list; instead, their `status` is transitioned to `PedestrianStatus.EVACUATED` or `INCAPACITATED`. Inactive agents stop updating position (`update_position()` returns early).

### Q15. Does an existing predictor already exist?
- **Implementation**: Yes, [`LinearTrajectoryPredictor`](file:///c:/Users/harsh/projects/genevac/src/genevac/prediction/__init__.py#L21-L36) in `src/genevac/prediction/__init__.py`.

### Q16. What inputs, output shapes, and assumptions does it use?
- **Inputs**: `history: List[Tuple[float, float]]`, `horizon: int = 10`.
- **Output**: `predictions: List[Tuple[float, float]]` of length `horizon`.
- **Assumptions**: Calculates linear velocity displacement from the last two observations:
  $$\Delta x = \text{history}[-1][0] - \text{history}[-2][0], \quad \Delta y = \text{history}[-1][1] - \text{history}[-2][1]$$
  Extrapolates uniformly for `horizon` steps. If fewer than 2 history points exist, repeats the last position. Assumes a constant timestep between history points matching the desired forecast step.

### Q17. Which tests cover the relevant code?
- [`tests/test_simulation.py`](file:///c:/Users/harsh/projects/genevac/tests/test_simulation.py): `test_simulation_step`
- [`tests/test_pedestrians.py`](file:///c:/Users/harsh/projects/genevac/tests/test_pedestrians.py): `test_pedestrian_movement`, `test_inactive_pedestrian_does_not_move`
- [`tests/test_prediction.py`](file:///c:/Users/harsh/projects/genevac/tests/test_prediction.py): `test_history_buffer`, `test_linear_predictor`
- [`tests/test_metrics.py`](file:///c:/Users/harsh/projects/genevac/tests/test_metrics.py): `test_metrics_calculation` (ADE, FDE, clearance time)
- [`tests/test_heatmaps.py`](file:///c:/Users/harsh/projects/genevac/tests/test_heatmaps.py): `test_heatmap_density`
- [`tests/test_experiments.py`](file:///c:/Users/harsh/projects/genevac/tests/test_experiments.py): `test_experiment_runner`
- [`tests/test_export.py`](file:///c:/Users/harsh/projects/genevac/tests/test_export.py): `test_export_json_and_csv`

### Q18. Which dependencies and Python versions are currently declared?
- **Specification**: [`pyproject.toml`](file:///c:/Users/harsh/projects/genevac/pyproject.toml) specifies `requires-python = ">=3.10"`.
- **Declared Dependencies**:
  - `numpy>=1.24.0`
  - `scipy>=1.10.0`
  - `pandas>=2.0.0`
  - `matplotlib>=3.7.0`
  - `pydantic>=2.0.0`
  - `pyyaml>=6.0`
  - `tqdm>=4.65.0`
  - Optional / Dev: `pytest>=7.4.0`, `pytest-cov>=4.1.0`, `black>=23.7.0`, `ruff>=0.0.280`.

### Q19. Which files are shared with Person A and must remain untouched?
- `src/genevac/simulation/*` ([`__init__.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/simulation/__init__.py))
- `src/genevac/environment/*` ([`__init__.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/environment/__init__.py))
- `src/genevac/pedestrians/*` ([`__init__.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/pedestrians/__init__.py))
- `src/genevac/config.py` ([`config.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/config.py))
- `src/genevac/main.py` ([`main.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/main.py))
- `tests/test_simulation.py` ([`test_simulation.py`](file:///c:/Users/harsh/projects/genevac/tests/test_simulation.py))
- `tests/test_pedestrians.py` ([`test_pedestrians.py`](file:///c:/Users/harsh/projects/genevac/tests/test_pedestrians.py))

### Q20. What is the minimum existing information available for a future prediction model?
1. Pedestrian ID (`str`)
2. Current Position `(x, y)` (`Tuple[float, float]` in meters)
3. Trajectory History (list of up to 20 past `(x, y)` positions)
4. Nominal Timestep duration ($dt = 0.1$ s)

