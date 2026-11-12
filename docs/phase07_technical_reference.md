# Phase 07 Technical Reference: Baseline Trajectory Prediction & Original Evacuation MID

**Document Version:** 2.0.0  
**Phase:** 07 — Baseline Trajectory Prediction & Original Evacuation MID  
**Subsystem:** Module B Part 2 (Person B / Harsha)  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary & Subsystem Architecture

Phase 07 delivers a comprehensive, scientifically disciplined trajectory prediction subsystem for GenEvac-Sentinel comprising:
1. **Kinematic Baseline (`ConstantVelocityPredictor`):** A deterministic, timestamp-aware velocity extrapolation model that enforces strict contract guarantees, prevents cumulative numerical drift, and handles irregular observation intervals and missing steps.
2. **Original Evacuation MID (`EvacuationMIDPredictor`):** An independently formulated stochastic Motion Indeterminacy Diffusion approach designed from first principles for emergency evacuation environments. It models tactical multi-modal branching via conditional reverse diffusion guided by exit attraction, obstacle repulsion, kinematic inertia, and physical boundary projection.
3. **Integration Layer:** Complete read-only compatibility with `TrajectorySnapshot`, `TrajectoryHistoryBuffer`, `EnvironmentMap`, `PredictionEvaluator`, and Phase 06 scenario-comparison pipelines.

```
                  [ Simulation Engine (Person A) / Trajectory History ]
                                            │
                                            ▼
                                   [ TrajectorySnapshot ]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
     [ ConstantVelocityPredictor ]                    [ EvacuationMIDPredictor ]
     - Deterministic Kinematic Baseline               - Original Stochastic Diffusion Model
     - Timestamp-Aware Extrapolation                  - Conditioning: Velocity, Exits, Obstacles, Crowd
     - Single-Path Output                             - Evacuation Guidance Field (Attraction/Repulsion)
     - Drift-Free Formula                             - Multi-Sample Generation (K Plausible Paths)
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                               [ PredictionEvaluator ]
                    - ADE (Average Displacement Error)
                    - FDE (Final Displacement Error)
                    - Stochastic Metrics: Best-of-K ADE/FDE, Mean ADE/FDE
                    - Sample Diversity & Walkable Space Validity
```

---

## 2. Part A — Baseline Trajectory Prediction

### 2.1 Kinematic Derivation & Drift Avoidance
Given the two most recent valid observations $o_{\text{prev}} = (x_{\text{prev}}, y_{\text{prev}}, t_{\text{prev}})$ and $o_{\text{last}} = (x_{\text{last}}, y_{\text{last}}, t_{\text{last}})$ with physical time delta $\Delta t = t_{\text{last}} - t_{\text{prev}} > 0$:
$$v_x = \frac{x_{\text{last}} - x_{\text{prev}}}{\Delta t}, \quad v_y = \frac{y_{\text{last}} - y_{\text{prev}}}{\Delta t}$$
For future steps $k \in \{1, \dots, H\}$ at interval $\delta > 0$, future timestamp is $t_{\text{fut}}(k) = t_{\text{ref}} + k \cdot \delta$:
$$x_{\text{fut}}(k) = x_{\text{last}} + v_x \cdot (t_{\text{fut}}(k) - t_{\text{last}}), \quad y_{\text{fut}}(k) = y_{\text{last}} + v_y \cdot (t_{\text{fut}}(k) - t_{\text{last}})$$

Direct projection from $o_{\text{last}}$ rather than sequential accumulation eliminates numerical floating-point drift.

### 2.2 Error Preconditions & Contract Guardrails
The predictor enforces strict defensive checks:
- Rejects sequences with $< 2$ observations (`TrajectoryValidationError`).
- Rejects non-positive or non-finite horizons, prediction intervals, or time deltas.
- Rejects stale observations exceeding `max_observation_age`.
- Rejects future observations where $t_{\text{obs}} > t_{\text{ref}} + 10^{-9}\text{ s}$.
- Batch execution (`predict_batch`) isolates individual failures into per-agent validation errors without corrupting valid agent predictions.

---

## 3. Part B — Original Evacuation Motion Indeterminacy Diffusion (MID)

### 3.1 Problem Definition: Motion Indeterminacy in Evacuation
In an emergency evacuation, pedestrian trajectories exhibit physical and behavioral multi-modality. Rather than viewing variations as random noise, GenEvac-Sentinel treats motion indeterminacy as the **tactical uncertainty of human decision-making** under spatial and social constraints:
- Choosing between alternative exits when one is congested.
- Navigating around central obstacles (detouring left vs. right).
- Decelerating or waiting at narrow bottleneck corridors.

### 3.2 16-Dimensional Conditioning Vector ($C$) & Canonical Normalization
The diffusion process is conditioned on real, verified data extracted from `TrajectorySnapshot` and `EnvironmentMap`:
1. **Motion Kinematics:** Instantaneous velocity $(v_x, v_y)$, speed $s$, unit heading $\mathbf{u}_v$, acceleration $(a_x, a_y)$.
2. **Exit Goal Context:** Exit displacement $(dx_E, dy_E)$, distance $d_E$, exit unit vector $\mathbf{u}_E$.
3. **Crowd Context:** Local crowd density $\rho_{\text{local}}$ ($R = 3.0\text{ m}$), nearest neighbor separation $d_{\min}$.
4. **Environment Geometry:** Distance to nearest room wall $d_{\text{wall}}$, distance to nearest obstacle bounding box $d_{\text{obs}}$.

All 16 features are strictly ordered and normalized via canonical scales (`FEATURE_NORMALIZATION_SCALES`) ensuring scale invariance across physical units (e.g. velocities scaled by 2.0 m/s, distances by 50.0 m, crowd density by 4.0 ped/m²). Both training and inference consume the identical normalized conditioning vector.

If `target_exit_id` is explicitly specified but absent from `environment.exits`, a `TrajectoryValidationError` is raised immediately. Silent fallback or exit substitution is strictly prohibited.

### 3.3 Forward Diffusion & Training Loss Masking
Future relative trajectory displacements $\Delta Y_0 \in \mathbb{R}^{H \times 2}$ are corrupted across $N$ steps:
$$q(\Delta Y_n \mid \Delta Y_0) = \mathcal{N}\left(\Delta Y_n;\, \sqrt{\bar{\alpha}_n} \Delta Y_0,\, (1 - \bar{\alpha}_n) \mathbf{I}\right)$$
where $\beta_1, \dots, \beta_N$ follow a linear variance schedule, $\alpha_n = 1 - \beta_n$, and $\bar{\alpha}_n = \prod_{i=1}^n \alpha_i$.

During training (`fit` and `train_step`), unobserved future steps are handled via an explicit binary loss mask without silent point duplication:
$$\mathcal{L} = \frac{\sum_{b=1}^B \sum_{h=1}^H M_{b, h} \|\hat{\epsilon}_{b, h} - \epsilon_{b, h}\|_2^2}{2 \sum_{b=1}^B \sum_{h=1}^H M_{b, h} + 10^{-8}}$$

### 3.4 Reverse Sampling with Per-Sample Evacuation Guidance Field
Starting from pure Gaussian noise $\Delta Y_N \sim \mathcal{N}(0, \mathbf{I})$, reverse diffusion proceeds iteratively. To ensure physical and behavioral validity, each generated sample $k \in \{0, \dots, K-1\}$ is augmented by an **independent per-sample Evacuation Guidance Field** $\mathbf{g}_{\text{evac}}^{(k)}$ evaluated on that sample's coordinates:
$$\Delta Y_{\text{final}}^{(k)} = \Delta Y_{\text{drift}} + 0.25 \cdot \Delta Y_{\text{diff}}^{(k)} + 0.15 \cdot \mathbf{g}_{\text{evac}}\left( \Delta Y_{\text{diff}}^{(k)},\, C,\, \text{Env} \right)$$

The guidance field evaluates:
- **Exit Attraction:** $\mathbf{g}_{\text{exit}} = w_{\text{exit}} \cdot \mathbf{u}_E$
- **Deterministic Obstacle Repulsion & Escape:**
  - Outside obstacles ($d > 0$): repulsive force away from the nearest obstacle surface.
  - Inside obstacles ($d = 0$): computes distance to all 4 bounding box faces and emits an outward escape unit vector toward the closest face with deterministic tie-breaking.
  - Force capping: All guidance forces are capped at $F_{\max} = 5.0\text{ m/s}^2$ ensuring strict numerical stability.
- **Kinematic Alignment:** $\mathbf{g}_{\text{kin}}(k) = w_{\text{kin}} \cdot \exp(-k \lambda) \cdot \mathbf{u}_v$
- **Boundary Clamping:** Clamps coordinates to navigable bounds $[r_{\text{ped}}, W - r_{\text{ped}}] \times [r_{\text{ped}}, H - r_{\text{ped}}]$.
- **Isolated RNG:** All sampling uses local `torch.Generator(device)` instances, preventing global random state mutation.
- **Horizon Validation:** Supports and slices for any $1 \le H \le H_{\text{cfg}}$; rejects non-positive or oversized horizons.

### 3.5 Neural Denoising Architecture (`EvacuationMIDNetwork`)
Implemented in PyTorch for CPU inference:
- Sinusoidal step embedding + MLP projection.
- Conditioning encoder MLP (16 $\to$ 64 dimensions with LayerNorm and SiLU).
- Trajectory linear projection (Horizon $\times$ 2 $\to$ 64).
- Residual MLP backbone with LayerNorm and SiLU (128 hidden units).
- Zero-initialized linear head outputting estimated noise $\hat{\epsilon} \in \mathbb{R}^{H \times 2}$.

---

## 4. Part C — Evaluation Protocols

### 4.1 Evaluation Measures & Physical Validity
1. **Average Displacement Error (ADE):**
   $$\text{ADE} = \frac{1}{H} \sum_{k=1}^H \|p_k^{\text{gt}} - p_k^{\text{pred}}\|_2$$
2. **Final Displacement Error (FDE):**
   $$\text{FDE} = \|p_H^{\text{gt}} - p_H^{\text{pred}}\|_2$$
3. **Oracle Best-of-K ADE ($\text{ADE}_{\min}$):** Minimum ADE achieved across $K$ sampled trajectories:
   $$\text{ADE}_{\min} = \min_{1 \le i \le K} \text{ADE}(Y^{(i)}, Y^{\text{gt}})$$
   Explicitly labeled as an oracle metric to avoid misleading claims of single-path accuracy.
4. **Expected Mean-of-K ADE ($\text{ADE}_{\text{mean}}$):** Expected ADE across all $K$ samples without oracle selection:
   $$\text{ADE}_{\text{mean}} = \frac{1}{K} \sum_{i=1}^K \text{ADE}(Y^{(i)}, Y^{\text{gt}})$$
5. **Sample Diversity ($\text{Div}$):** Average pairwise displacement divergence across generated paths:
   $$\text{Div} = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \frac{1}{H} \sum_{h=1}^H \|p_h^{(i)} - p_h^{(j)}\|_2$$
6. **Point Walkability Rate:** Fraction of generated waypoints residing within room bounds and outside obstacles:
   $$\text{Val}_{\text{point}} = \frac{\sum_{i=1}^K \sum_{h=1}^H \mathbb{I}\left( \text{is\_walkable}(p_h^{(i)}) \right)}{K \cdot H}$$
7. **Segment Collision-Free Rate:** Fraction of sampled paths where no consecutive line segment $(p_h, p_{h+1})$ intersects any obstacle bounding box, verified via the Liang-Barsky parametric clipping algorithm (`line_segment_intersects_box`).
8. **Mean Prediction Caveat:** The arithmetic point-wise mean across diverse multi-modal trajectory clusters is documented as a statistical summary only; in non-convex environments, an arithmetic mean path can penetrate obstacles even when all individual sampled paths are collision-free.
9. **Evaluation Audit Diagnostics:** Evaluators record `evaluated_samples`, `failed_samples`, and `skipped_samples` with specific failure reasons, ensuring zero silent dropping.

---

## 5. Part D — Training Feasibility & Status Declaration

### 5.1 Training Data Feasibility Audit
- `data/sample/sample_trajectories.json`: 3 pedestrians, 26 total observations.
- `data/scenarios/sample_hallway.json`: 1 scenario, deterministic straight movement.
- `data/raw/eth/`, `hotel/`: Non-evacuation outdoor sidewalk datasets without emergency exit bottlenecks.
- Upstream simulation engine: Deterministic kinematics with zero stochastic variance.

### 5.2 Status Declaration
- **Declared Status:** **Implemented Prototype with Genuine Architecture and Verification Pipeline.**
- A complete training pipeline (`fit` / `train_step`) is implemented and verified using PyTorch Adam optimization with explicit binary loss masking for unobserved future steps.
- The model tracks its training provenance explicitly via `is_trained`, `training_sample_count`, `training_epochs_completed`, and `is_trained_model`.
- The model is verified on synthetic and contract datasets, with honest disclosure that large-scale evacuation generalization awaits future multi-scenario emergency dataset collection.

---

## 6. Verification Evidence

### 6.1 Test Suite Breakdown
```text
Full Test Suite: python -m pytest tests/ -v
Result: 232 passed, 0 failed, 1 warning in 5.01s

Phase 07 Dedicated Test Suites:
1. tests/test_baseline_trajectory_phase07.py (13 tests) ........... PASSED (0.10s)
   - Straight-line diagonal and negative velocities
   - Stationary pedestrians
   - Changing velocity with ADE/FDE scoring
   - Irregular intervals and skipped steps
   - Insufficient history rejection (0 and 1 observations)
   - Invalid dt and future observation rejection
   - Stale observation enforcement
   - Missing agent batch prediction error isolation
   - Run reset and isolation
   - Multiple pedestrian batch predictions
   - Deterministic reproducibility
   - Zero simulation mutation

2. tests/test_mid_phase07.py (11 tests) ........................... PASSED (4.51s)
   - Feature ordering and canonical normalization (16 features)
   - Finite guidance and numerical stability (capped F_max = 5.0)
   - Invalid target exit rejection (TrajectoryValidationError, zero silent fallback)
   - Isolated RNG without global side effects (torch.Generator & numpy rng)
   - Obstacle repulsion inside box & corner escape normal vectors
   - Per-sample guidance independence (independent evaluation per trajectory)
   - Segment collision detection (Liang-Barsky line clipping)
   - Stochastic evaluation reporting (oracle Best-of-K, Mean ADE, Diversity, Validity)
   - Trained vs. untrained model provenance tracking
   - Training pipeline with loss masking (no silent point duplication)
   - Horizon validation (1 <= H <= H_cfg slicing) and parameter checks

Regression Baseline:
All 208 prior tests across Phases 01-06 and Export continue to pass with zero regressions.
```
