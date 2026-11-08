# GenEvac-Sentinel: Known Limitations and Scope

## 1. Project Purpose and Operational Scope

GenEvac-Sentinel is an **educational evacuation simulation and AI trajectory prediction research prototype**.

> [!WARNING] **Not a Certified Life-Safety System**
> This software is designed strictly for academic coursework, algorithmic research, and educational demonstration. It is **NOT** a certified life-safety system, building code compliance tool, or architectural fire safety validation engine. It must never be used for real-world emergency egress design or physical life-safety planning.

---

## 2. Accepted System Limitations

Through rigorous end-to-end verification and regression testing, the following four specific limitations have been documented and accepted for the Person A baseline:

### 2.1 Head-on Symmetrical Deadlocks in Narrow Passages
- **Description**: When two pedestrians with identical physical radii ($r = 0.25\text{m}$) approach each other in a direct head-on trajectory along the centerline of a narrow corridor or doorway where the total width is $< 4r$ ($< 1.0\text{m}$), neither agent can deflect laterally without intersecting a wall.
- **Engine Behavior**: Collision safety takes absolute precedence. Both agents decelerate to $v = 0.0\text{ m/s}$ and enter a safe, stationary halt rather than penetrating each other's body volume or tunneling through walls.
- **Accepted Trade-off**: The engine prefers a safe physical standstill over unphysical collision or clipping. Social yielding conventions (e.g. stepping aside or reverse negotiation) are not implemented in this prototype.

### 2.2 Snapshot ID Run-Locality & Absence of Run Identifier
- **Description**: `SimulationObservationSnapshot` contains `snapshot_id` formatted as `f"snap_{step_count:06d}"` (e.g., `"snap_000004"`), `step_count`, and `timestamp`. It is monotonic and unique **only within a single continuous simulation run**.
- **Reset Repetition**: Upon engine reset or when launching a new simulation instance, `step_count` resets to `0`, regenerating identical snapshot IDs (`"snap_000000"`, `"snap_000004"`). The schema does not contain an independent `run_id` or `session_id`.
- **Downstream Impact**: In isolation, snapshot IDs are not globally unique across distinct runs or Monte Carlo iterations.
- **Accepted Solution**: Downstream consumers (Person B) must track execution sessions externally, either by prefixing output file paths, organizing files into run-specific directories, or wrapping snapshot ingestion in run-aware containers.

### 2.3 Upstream Pygame `pkg_resources` Deprecation Warning
- **Description**: Upon initializing Pygame 2.6.1, the Python runtime emits an upstream deprecation warning:
  `UserWarning: pkg_resources is deprecated as an API. See setuptools docs.`
- **Root Cause**: An internal import within the third-party Pygame distribution referencing deprecated `pkg_resources` instead of `importlib.resources`.
- **Status**: Benign. It does not affect simulation physics, mathematical determinism, or test execution.

### 2.4 Single-Floor 2D Plane Boundary
- **Description**: GenEvac-Sentinel operates strictly in a continuous 2D Euclidean plane ($X, Y$).
- **Excluded Features**: Multi-floor vertical egress (stairwells, ramps, elevators), dynamic environmental hazards (fire spread, toxic smoke propagation, oxygen depletion), and non-linear panic behaviors (stampedes, pushing forces, crush injuries) are outside the prototype scope.

---

## 3. Verified Robust Capabilities

To ensure clarity, the following capabilities are fully verified and robust:

| Subsystem | Capability Status | Verification Evidence |
| :--- | :--- | :--- |
| **Drift-Free Clock** | $\Delta t = 0.1\text{ s}$ exactly, no cumulative floating-point drift | Tested across 1,000+ continuous steps |
| **Collision Avoidance** | Wall penetration, corner cutting, and agent overlap rejected | Verified by 17-angle deflection kinematics |
| **Deterministic Replay** | Identical inputs produce bitwise identical trajectories | Validated across automated multi-run tests |
| **Observation Immutability** | Frozen dataclasses prevent downstream mutations | Tested via `FrozenInstanceError` assertions |
| **Indefinite Retention** | Evacuated agents remain in registry and snapshots | Validated across full evacuation scenarios |
| **Dynamic Consistency** | $N = N_{\text{act}} + N_{\text{evac}} + N_{\text{blk}} + N_{\text{inact}}$ always holds | Enforced via `__post_init__` validation |

---

## 4. Considerations for Future Development (Post-Handover)

For future versions beyond the educational prototype:
1. **Reciprocal Velocity Obstacles (RVO2)**: Implement cooperative reciprocal velocity obstacles to resolve narrow passage deadlocks through collaborative collision-avoidance cones.
2. **Dynamic Route Recalculation**: Allow agents to dynamically re-evaluate alternative exits when an active exit becomes congested or blocked during an ongoing evacuation.
3. **Session Metadata**: Extend `SimulationObservationSnapshot` with a configurable `run_id: str` field upon schema version bump.

