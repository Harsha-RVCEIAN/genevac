# Phase 07 Technical Specification: GenEvac-Sentinel Evacuation Motion Indeterminacy Diffusion (MID)

**Document Version:** 2.0.0  
**Phase:** 07 — Stochastic Trajectory Prediction & Evacuation MID  
**Subsystem:** Module B Part 2 (Person B / Harsha)  
**Status:** IMPLEMENTED PROTOTYPE WITH VERIFIED DIFFUSION & GUIDANCE PIPELINE  

---

## 1. Executive Summary & Originality Declaration

This document specifies the original, project-specific **Motion Indeterminacy Diffusion (MID)** architecture developed independently for the GenEvac-Sentinel emergency evacuation simulator.

### Originality Statement
In strict compliance with repository safety and scientific integrity guidelines:
- **No external proprietary or researcher-specific repositories** (such as Guiteanpai's `research/MID/` or Trajectron++) were consulted, copied, reproduced, or renamed for this implementation.
- All mathematical formulations, conditioning representations, guidance vector fields, network architectures, and evaluation protocols presented here were **designed independently from first principles** specifically for constrained emergency evacuation environments.
- Established general diffusion concepts (Sohl-Dickstein et al. 2015; Ho et al. 2020) are formally cited where employed, and explicitly distinguished from our project-specific evacuation guidance and conditioning mechanisms.

---

## 2. Problem Formulation: Motion Indeterminacy in Evacuation

### 2.1 Indeterminacy vs. Prediction Error
Traditional deterministic predictors (such as constant-velocity or single-path regressors) model trajectory forecasting as a regression problem: $\hat{Y} = f(X)$. In doing so, they conflate **aleatoric motion indeterminacy** with **model error**:
- **Prediction Error:** The numerical discrepancy between a deterministic point estimate and a single realized ground truth.
- **Motion Indeterminacy:** The genuine physical and behavioral multi-modality inherent to human evacuation decision-making. When faced with smoke, blocked exits, corridor congestion, or narrow doors, an evacuee does not follow an unalterable single path; they make tactical choices between plausible alternatives (e.g., detouring left vs. right around a central pillar, waiting for a bottleneck to clear, or accelerating toward an unobstructed peripheral exit).

Arbitrary random perturbations (e.g. adding Gaussian noise to a constant-velocity line) do **not** constitute a diffusion model. Random jitter violates temporal correlation, produces dynamically impossible zig-zag paths, ignores environmental boundaries, and fails to represent tactical uncertainty.

### 2.2 Prediction Target & Uncertainty Evolution
Let $X_{1:T_{\text{obs}}} = (p_1, p_2, \dots, p_{T_{\text{obs}}}) \in \mathbb{R}^{T_{\text{obs}} \times 2}$ denote the observed historical positions of a pedestrian up to reference timestamp $t_{\text{ref}} = t_{T_{\text{obs}}}$.

The target is to forecast the sequence of $H$ future displacement positions:
$$\Delta Y = \left( \Delta p_1, \Delta p_2, \dots, \Delta p_H \right) \in \mathbb{R}^{H \times 2}, \quad \text{where } \Delta p_k = p_{T_{\text{obs}}+k} - p_{T_{\text{obs}}}$$
The absolute future trajectory is recovered via:
$$Y = p_{T_{\text{obs}}} + \Delta Y$$

In an evacuation, the spatial uncertainty distribution $p(\Delta p_k \mid X, \text{Context})$ expands nonlinearly over the prediction horizon $k \in \{1, \dots, H\}$. At small horizons ($k \approx 1$), momentum dominates and variance is tight; at large horizons ($k \approx H$), tactical branching (route choice, obstacle avoidance) produces broad multi-modal distributions.

---

## 3. Evacuation Context & Conditioning Representation

### 3.1 16-Dimensional Conditioning Vector ($C$)
Conditioning features are extracted directly from `TrajectorySnapshot` and `EnvironmentMap` with the exact 16-element ordering:

| Index | Feature Description | Physical Unit | Normalization Scale | Normalized Range |
| :---: | :--- | :--- | :---: | :---: |
| 0 | Instantaneous velocity $v_x$ | $\text{m/s}$ | $2.0$ | $[-1.0, 1.0]$ |
| 1 | Instantaneous velocity $v_y$ | $\text{m/s}$ | $2.0$ | $[-1.0, 1.0]$ |
| 2 | Estimated speed $s = \|\mathbf{v}\|_2$ | $\text{m/s}$ | $2.0$ | $[0.0, 1.0]$ |
| 3 | Heading unit vector $u_x$ | dimensionless | $1.0$ | $[-1.0, 1.0]$ |
| 4 | Heading unit vector $u_y$ | dimensionless | $1.0$ | $[-1.0, 1.0]$ |
| 5 | Acceleration $a_x$ (last 3 points) | $\text{m/s}^2$ | $2.0$ | $[-1.0, 1.0]$ |
| 6 | Acceleration $a_y$ (last 3 points) | $\text{m/s}^2$ | $2.0$ | $[-1.0, 1.0]$ |
| 7 | Target exit displacement $dx_E$ | $\text{m}$ | $50.0$ | $[-1.0, 1.0]$ |
| 8 | Target exit displacement $dy_E$ | $\text{m}$ | $50.0$ | $[-1.0, 1.0]$ |
| 9 | Target exit distance $d_E$ | $\text{m}$ | $50.0$ | $[0.0, 1.0]$ |
| 10 | Target exit unit heading $u_{Ex}$ | dimensionless | $1.0$ | $[-1.0, 1.0]$ |
| 11 | Target exit unit heading $u_{Ey}$ | dimensionless | $1.0$ | $[-1.0, 1.0]$ |
| 12 | Local crowd density $\rho_{\text{local}}$ ($R=3\text{m}$) | $\text{ped/m}^2$ | $4.0$ | $[0.0, 1.0]$ |
| 13 | Nearest neighbor distance $d_{\min}$ | $\text{m}$ | $50.0$ | $[0.0, 1.0]$ |
| 14 | Distance to nearest room wall $d_{\text{wall}}$ | $\text{m}$ | $50.0$ | $[0.0, 1.0]$ |
| 15 | Distance to nearest obstacle $d_{\text{obs}}$ | $\text{m}$ | $50.0$ | $[0.0, 1.0]$ |

### 3.2 Canonical Feature Normalization
Both training (`fit`) and inference (`predict_samples`) consume the identical `normalized_vector` computed by `EvacuationConditioning`, ensuring scale invariance and preventing high-magnitude geometric distances (e.g. 50 m) from overwhelming unit-scale headings and velocities.

### 3.3 Strict Exit & Precondition Handling
- **Target Exit Validation:** If `target_exit_id` is explicitly specified, it must exist in `environment.exits`. If absent, `TrajectoryValidationError` is raised. **Silent substitution of exits is strictly prohibited.**
- **Missing Environment Fallback:** When `environment=None`, exits default to $(0,0)$, density to $0$, and boundary distances to safe nominal defaults (50.0 m) without crashing.

---

## 4. Mathematical Model: Diffusion & Per-Sample Guidance

### 4.1 Forward Diffusion Process
The forward process corrupts ground-truth displacement trajectories $\Delta Y_0 \in \mathbb{R}^{H \times 2}$ over $N$ discrete diffusion steps:
$$q(\Delta Y_n \mid \Delta Y_0) = \mathcal{N}\left(\Delta Y_n;\, \sqrt{\bar{\alpha}_n} \Delta Y_0,\, (1 - \bar{\alpha}_n) \mathbf{I}\right)$$
$$\Delta Y_n = \sqrt{\bar{\alpha}_n} \Delta Y_0 + \sqrt{1 - \bar{\alpha}_n} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### 4.2 Reverse Diffusion & Per-Sample Guidance Architecture
The reverse diffusion network $\epsilon_\theta(\Delta Y_n, n, C)$ estimates denoising noise updates.
Sampling proceeds through reverse diffusion to produce candidate displacement trajectories $\Delta Y^{(k)}$.

To ensure physical and behavioral validity, each generated sample $k \in \{0, \dots, K-1\}$ is augmented by an **independent per-sample Evacuation Guidance Field** $\mathbf{g}_{\text{evac}}^{(k)}$:
$$\Delta Y_{\text{final}}^{(k)} = \Delta Y_{\text{drift}} + 0.25 \cdot \Delta Y_{\text{diff}}^{(k)} + 0.15 \cdot \mathbf{g}_{\text{evac}}\left( \Delta Y_{\text{diff}}^{(k)},\, C,\, \text{Env} \right)$$

#### Component Forces:
1. **Exit Goal Attraction:**
   $$\mathbf{g}_{\text{exit}} = w_{\text{exit}} \cdot \mathbf{u}_E$$
2. **Deterministic Obstacle Repulsion & Escape Field:**
   For a point $(x, y)$ relative to obstacle $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$:
   - **Outside ($d > 0$):** Repulsion away from closest surface point:
     $$\mathbf{g}_{\text{obs}} = \min\left( F_{\max},\, \frac{w_{\text{obs}}}{(d + 0.1)^2} \right) \cdot \frac{p - p_{\text{surface}}}{\|p - p_{\text{surface}}\|_2 + \epsilon}$$
   - **Inside ($d = 0$):** Point is inside obstacle bounding box. The model computes distance to all 4 boundary faces ($d_{\text{left}}, d_{\text{right}}, d_{\text{bottom}}, d_{\text{top}}$) and emits a deterministic outward escape vector toward the nearest face with deterministic tie-breaking.
   - **Force Capping:** All guidance forces are capped at $F_{\max} = 5.0\text{ m/s}^2$ for strict numerical stability.
3. **Kinematic Inertia Consistency:**
   $$\mathbf{g}_{\text{kin}}(k) = w_{\text{kin}} \cdot \mathbf{u}_v \cdot \exp\left( -k \cdot \lambda_{\text{decay}} \right)$$
4. **Physical Room Boundary Clamping:**
   Boundary enforcement is kept distinct from obstacle repulsion. Coordinates are projected into navigable room bounds:
   $$\hat{x} \in [r_{\text{ped}}, W - r_{\text{ped}}], \quad \hat{y} \in [r_{\text{ped}}, H - r_{\text{ped}}]$$

### 4.3 Deterministic Isolated RNG
Sampling uses isolated `torch.Generator(device)` instances seeded per call. **Global Python, NumPy, or PyTorch random number generator states are never modified**, preventing side effects on unrelated project suites.

---

## 5. Neural Network Architecture: `EvacuationMIDNetwork`

```
Inputs:
  Trajectory Noise: ΔY_n (Batch, Horizon, 2)
  Diffusion Step:   n    (Batch, 1)
  Conditioning:     C    (Batch, 16)
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
[ Trajectory] [Step Embed] [Condition]
[ Linear    ] [ Sinusoid ] [ MLP     ]
[ Layer     ] [ + MLP    ] [ 16->64   ]
   │           │           │
   └───────────┼───────────┘
               ▼
     [ Concatenated Latent ] (Dim = 64 + 64 + 64 = 192)
               │
               ▼
     [ Residual MLP Block 1 ] (Hidden Dim = 128, SiLU, LayerNorm)
               │
               ▼
     [ Residual MLP Block 2 ] (Hidden Dim = 128, SiLU, LayerNorm)
               │
               ▼
     [ Output Projection Head ] (Linear -> Horizon x 2)
               │
               ▼
   Estimated Noise: ε_hat (Batch, Horizon, 2)
```

---

## 6. Training Pipeline with Loss Masking

When training on real or synthetic trajectory samples:
1. **No Silent Point Duplication:** Incomplete future ground-truth sequences are **not** padded by repeating the last observed point.
2. **Explicit Loss Masking:** A binary tensor mask $M \in \{0, 1\}^{B \times H \times 1}$ is constructed:
   $$\mathcal{L} = \frac{\sum_{b=1}^B \sum_{h=1}^H M_{b, h} \|\hat{\epsilon}_{b, h} - \epsilon_{b, h}\|_2^2}{2 \sum_{b=1}^B \sum_{h=1}^H M_{b, h} + 10^{-8}}$$
   Unobserved future steps receive zero loss weight.
3. **Skipped Sample Diagnostics:** Samples with invalid history ($< 2$ points) or missing ground-truth are recorded in `skipped_samples` with explicit diagnostic messages.

---

## 7. Multi-Sample Physical Validity & Evaluation Protocols

To avoid misleading aggregate claims, evaluation distinguishes:
1. **Point Walkability:** Fraction of generated points satisfying `environment.is_walkable(x, y)`.
2. **Segment Collision Free:** Evaluates whether consecutive trajectory segments $(p_k, p_{k+1})$ intersect any obstacle bounding box using parametric line-clipping (`line_segment_intersects_box`).
3. **Oracle Metrics:** `oracle_best_of_k_ade` and `oracle_best_of_k_fde` are explicitly labeled as oracle metrics.
4. **Expected Metric:** `expected_mean_of_k_ade` reports average sample error without cherry-picking.
5. **Sample Diversity:** Pairwise trajectory divergence $\text{Div}$.
6. **Mean Prediction Caveat:** `mean_prediction` is documented as an arithmetic point-wise mean. In multi-modal distributions, the arithmetic mean may pass through obstacles even if individual samples are collision-free.
