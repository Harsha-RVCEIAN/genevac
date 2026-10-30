# GenEvac MID Adaptation Design & Architecture Specification

## 1. Executive Summary

This document specifies the architectural design for **GenEvac-Sentinel's independent, native MID-based trajectory prediction system**.

Rather than porting or vendoring the legacy Gutianpei/MID codebase, we adapt the mathematically sound **Motion Indeterminacy Diffusion** principles into a clean, modern PyTorch implementation native to the `genevac` package. This architecture strictly adheres to the Part 2 trajectory data contract, respects Person A's ownership of the simulation engine, and supports both CPU execution and modern Python 3.10+ environments.

---

## 2. Methodology-to-GenEvac Adaptation Table

The following table details our technical decisions across every major component of the MID framework:

| # | Original MID Component | Purpose in Original MID | Relevant Source Implementation | GenEvac Requirement | Our Intended Implementation | Decision | Reason & Technical Risks |
|---|---|---|---|---|---|---|---|
| **1** | **Trajectory Input Representation** | Node state vector: $(x, y, v_x, v_y, a_x, a_y)$. | `environment/node.py`<br>`data_structures.py` | Consumes `TrajectoryObservation` records: $(x, y)$ in metres at step indices with timestamps. | Compute finite-difference velocity $\Delta x, \Delta y$; construct tensor $(B, T_{obs}, 4)$. | **Adapt** | *Reason*: Eliminates dependency on Trajectron dynamic node classes while retaining kinematic velocity state. |
| **2** | **Coordinate Preprocessing** | Scene-level zero-centering (`x -= x.mean()`) and scaling. | `process_data.py:133-145` | Top-left origin $(0, 0)$, positive right/downward in metres. | Local agent-centric coordinate normalization relative to current position $p_0$: $\tilde{p}_t = p_t - p_0$. | **Adapt** | *Reason*: Scene-mean centering does not generalize across varying evacuation scenarios; agent-centric centering is translation-invariant and numerically stable. |
| **3** | **Temporal Preprocessing** | Fixed $dt = 0.4\text{ s}$ (downsampled from 25 fps video). | `process_data.py:123` | Discrete simulation with engine-authored timestamps ($dt = 0.1\text{ s}$). | Native support for $dt = 0.1\text{ s}$; temporal normalization verifying $\Delta t \approx \text{timestep}$. | **Adapt** | *Reason*: Avoids artificial downsampling within the simulation loop; directly predicts 10 Hz simulation ticks. |
| **4** | **Observation History** | Fixed 8 timesteps (3.2 seconds at 2.5 Hz). | `mid.py:58-61`<br>`get_traj_hypers.py` | Configurable history window (default 20 steps = 1.9s or 2.0s at 10 Hz). | Configurable encoder input horizon $T_{obs}$ (default 20 steps); zero-padding with explicit boolean validity mask `history_valid_mask`. | **Adapt** | *Reason*: Preserves actual observations without data fabrication; velocities computed only from valid timestamps. |
| **5** | **Future Prediction Horizon** | Fixed 12 timesteps (4.8 seconds at 2.5 Hz). | `mid.py:60` | Configurable forecast horizon $H$ (default 20 to 30 steps = 2.0s to 3.0s). | Parameterized prediction horizon $H$; denoiser operates over sequence length $H$. | **Adapt** | *Reason*: Evacuation analysis requires flexible forecast horizons aligned with doorway and exit choke points. |
| **6** | **Social Interaction Representation** | Dynamic scene graph with spatio-temporal graph attention (Trajectron++). | `models/encoders/trajectron.py`<br>`graph_attention.py` | Interaction among neighboring evacuees avoiding collisions and bottlenecks. | Lightweight Spatial-Distance Multi-Head Attention layer over active neighbor vectors within radius $R$. | **Replace** | *Reason*: The monolithic Trajectron graph is brittle, requires `dill` serialization, and causes severe CPU latency. A clean PyTorch attention module is intended to provide interaction-aware conditioning inspired by MID's interaction-conditioning objective in $<2\text{ ms (untested target)}$. Note: This is a lightweight adaptation, not architecturally equivalent to the original Trajectron++ graph-based approach. |
| **7** | **Diffusion / Noise Process** | DDPM forward Markov chain with linear beta schedule ($T=100$). | `models/diffusion.py:9-84`<br>`class VarianceSchedule` | Controlled diffusion adding and removing Gaussian noise from future paths. | Clean `VarianceSchedule` supporting linear and cosine schedules with closed-form sampling. | **Preserve** | *Reason*: The forward noise process $\sqrt{\bar{\alpha}_t} y_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ is theoretically optimal and completely portable. |
| **8** | **Denoising Network Backbone** | `TransformerConcatLinear` combining context with time embedding. | `models/diffusion.py:172-215` | Multi-step temporal denoiser predicting $\hat{\epsilon}$. | FiLM / `ConcatSquashLinear` modulation + PyTorch `TransformerEncoder` predicting future displacement noise. | **Preserve** | *Reason*: Combining FiLM conditioning with self-attention across future timesteps effectively models temporal trajectory continuity. |
| **9** | **Training Objective** | MSE loss on predicted noise $\epsilon_\theta$: $\|\epsilon - \hat{\epsilon}\|_2^2$. | `models/diffusion.py:83` | Direct supervision on noise estimation. | Standard MSE loss on predicted noise $\epsilon_\theta$. | **Preserve** | *Reason*: Simple, stable convergence; avoids complex adversarial min-max optimization. |
| **10** | **Inference Sampler** | 100-step DDPM or 2-to-5 step fast DDIM sampling. | `models/diffusion.py:86-127` | Real-time and quasi-online inference with low CPU latency. | DDIM fast sampler with configurable stride ($S=20$ or $50$, corresponding to 5 or 2 steps). | **Preserve** | *Reason*: DDIM reduces 100 forward passes to 2–5 passes, reducing required denoiser evaluations from 100 to 2–5 (actual CPU latency untested). |
| **11** | **Multi-Modal Output** | Best-of-$K$ sampling from $K=20$ random initializations $y_T \sim \mathcal{N}(0, I)$. | `models/diffusion.py:88-94`<br>`mid.py:67` | Multiple candidate evacuation paths around bifurcations / obstacles. | Generate $K$ stochastic candidate paths ($K \in [1, 20]$); output distribution with confidence weights. | **Preserve** | *Reason*: Core value of diffusion: captures divergent path choices when escaping through multiple exits. |
| **12** | **Evaluation Protocol** | Minimum ADE/FDE across 20 samples ("Best-of-20"). | `mid.py:86-99`<br>`evaluation/` | Offline ADE/FDE against actual simulation ground truth; latency benchmarks. | Standard ADE, FDE, and Average Wall-Clock Latency metrics. | **Adapt** | *Reason*: Best-of-20 evaluates theoretical capacity; we additionally evaluate mean ADE and collision rate against obstacles. |
| **13** | **Dataset Loading & Storage** | Pickle files created by `dill` containing custom `Environment` objects. | `process_data.py`<br>`mid.py:246-249` | Ingestion of GenEvac simulation trajectory archives. | Native PyTorch `Dataset` parsing GenEvac JSON/CSV simulation logs into sliding observation-target windows. | **Replace** | *Reason*: Eliminates `dill` pickle incompatibility across Python versions; ensures reproducibility and clean data contracts. |

---

### 2.1 Coordinate Transformation Note

The coordinate preprocessing strategies between original MID and GenEvac differ and must never be mixed:

- **Original MID preprocessing**: Scene-mean centering (`x -= x.mean()`) with optional scaling (from `process_data.py:133-145`). All agents in a scene are centered relative to the scene's spatial mean.
- **GenEvac adaptation**: Agent-centric centering relative to $p_0 =$ last observed position:
  $$p_{\text{local}} = p_{\text{global}} - p_0$$
- **Reconstruction**: $p_{\text{global}} = p_0 + p_{\text{local}}$
- **Key difference**: Original MID centers on the scene mean; GenEvac centers per-agent on the last observed position. These are fundamentally different coordinate systems.
- **Rule**: Never mix coordinate systems between training and inference. The training preprocessing and the inference preprocessing must apply identical transformations.

### 2.2 Social Interaction Conditioning Specification

Our lightweight social-attention adaptation is inspired by MID's interaction-conditioning objective, designed for fast execution and numerical stability:

- **Neighbor Selection Radius**: Configurable interaction radius $R$ (default $R = 5.0\text{ m}$).
- **Distance Metric**: Euclidean distance computed from the **latest observed positions** $p_0$ of all active pedestrians.
- **Relative Position Features**: $\Delta p_{ij} = p_{j,0} - p_{i,0}$ in local agent-centric coordinates.
- **Relative Velocity Features**: $\Delta v_{ij} = v_{j,0} - v_{i,0}$ computed only from valid consecutive observations.
- **Neighbor Padding & Masking**: Fixed neighbor capacity $K_{\text{neighbors}}$ (default 8). When fewer than $K_{\text{neighbors}}$ exist within $R$, slots are padded with zeros and marked invalid in a boolean `neighbor_valid_mask`.
- **Zero-Neighbor Behavior**: When no neighbors are within radius $R$, the attention layer uses self-conditioning or zero-masked pooling, guaranteed to produce a clean zero vector without `NaN`s or divisions by zero.
- **Permutation Invariance**: Attention pooling is permutation-invariant with respect to neighbor ordering in the input tensor.
- **Scope Clarification**: This is a lightweight adaptation, not architecturally equivalent to Trajectron++'s dynamic heterogeneous graph.

---

## 3. GenEvac Data Contract Integration

Our native MID implementation strictly adheres to the Part 2 data contracts defined in [`src/genevac/prediction/data_contract.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/prediction/data_contract.py).

### 3.1 Input Contract: `TrajectorySnapshot`
The predictor consumes read-only observations without interacting with internal simulation engine state:

```python
snapshot: TrajectorySnapshot
# snapshot.current_step: int
# snapshot.current_timestamp: float
# snapshot.timestep: float
# snapshot.trajectories: Mapping[str, Tuple[TrajectoryObservation, ...]]
```

- **Zero Simulation Mutation**: Calling prediction models never alters `Pedestrian.position`, `Pedestrian.velocity`, or `SimulationEngine.current_step`.
- **Top-Left Coordinate Origin**: Input coordinates $(x, y)$ are in metres, where $+X$ is right and $+Y$ is downward.

### 3.2 Output Contract: `TrajectoryPrediction`
The prediction results are returned in dedicated, immutable data structures:

```python
prediction: TrajectoryPrediction
# prediction.agent_id: str
# prediction.source_step_index: int       # Simulation step of the input snapshot
# prediction.source_timestamp: float      # Physical timestamp of the input snapshot
# prediction.predictions: Tuple[PredictedPosition, ...]
#   - PredictedPosition(timestamp=t_1, x=x_1, y=y_1) where t_1 > source_timestamp
# prediction.category = TrajectoryCategory.PREDICTED
```

- **Anti-Staleness Enforcement**: Every prediction result permanently binds `source_step_index` and `source_timestamp`. Downstream analytics reject predictions if the simulation clock has advanced past the validity threshold.
- **Strict Data Isolation**: Predicted positions are typed as `PredictedPosition` and categorized as `PREDICTED`. They are **never** appended into the authoritative `TrajectoryHistoryBuffer`.

---

## 4. Independent GenEvac MID Architecture & Module Plan

To ensure modularity and maintainability, the native MID implementation will be organized under `src/genevac/prediction/mid/`:

```text
src/genevac/prediction/
├── __init__.py                # Re-exports contracts, buffer, baseline, and MID predictor
├── data_contract.py           # Authoritative Part 2 data contracts & validation (FROZEN)
├── baseline.py                # Pure kinematic Constant Velocity baseline
├── mid/                       # Native GenEvac MID Subsystem
│   ├── __init__.py            # Module exports (GenEvacMIDPredictor, MIDConfig)
│   ├── config.py              # Hyperparameters (diffusion steps, dimensions, horizons)
│   ├── preprocessing.py       # Snapshot-to-tensor normalization & local frame transformation
│   ├── dataset.py             # PyTorch Dataset for slicing simulation logs into (X_obs, Y_tgt)
│   ├── encoders.py            # Past trajectory GRU encoder & Social Multi-Head Attention
│   ├── diffusion.py           # VarianceSchedule, forward q-sampling, and training loss
│   ├── denoiser.py            # FiLM-conditioned Transformer temporal denoiser
│   ├── sampler.py             # DDIM fast sampler (2-5 steps) & DDPM stochastic sampler
│   └── predictor.py           # High-level GenEvacMIDPredictor implementing ITrajectoryPredictor
```

### Component Breakdown

1. **`preprocessing.py`**:
   - Converts `TrajectorySnapshot` into standardized PyTorch tensors:
     - `history_tensor`: $(B, T_{obs}, 4)$ representing local displacements $(x - x_0, y - y_0, v_x, v_y)$.
     - `neighbor_tensor`: $(B, K_{neighbors}, T_{obs}, 2)$ containing relative positions of nearest agents.
     - `mask`: Boolean validity mask for agents with short histories.
2. **`encoders.py`**:
   - `TrajectoryEncoder`: Bidirectional GRU mapping individual history $(B, T_{obs}, 4) \to (B, d_h)$.
   - `SocialInteractionEncoder`: Spatial Multi-Head Attention computing interaction vectors among nearby agents within radius $R=5.0\text{ m}$.
   - Merges features into conditioning context $c \in \mathbb{R}^{B \times d_{ctx}}$.
3. **`diffusion.py`**:
   - Implements `VarianceSchedule` ($\beta_1=10^{-4}$ to $\beta_T=0.05$).
   - Computes closed-form noisy targets: $y_t = \sqrt{\bar{\alpha}_t} y_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
   - Computes MSE loss between true noise $\epsilon$ and denoiser output $\hat{\epsilon}$.
4. **`denoiser.py`**:
   - Combines diffusion time embedding $[\beta_t, \sin(\beta_t), \cos(\beta_t)]$ with context $c$ via `ConcatSquashLinear`.
   - Passes conditioned future trajectories through 3 PyTorch `TransformerEncoderLayer` modules.
5. **`sampler.py`**:
   - Executes DDIM reverse iterations with configurable stride (e.g. 5 steps).
   - Generates $K$ future trajectory candidates from $y_T \sim \mathcal{N}(0, I)$.
   - Reconstructs global world coordinates: $p_{pred} = p_0 + \sum \Delta \tilde{p}$.

---

## 5. Training Data Strategy & Leakage Prevention

### 5.1 Training Data Sources
1. **GenEvac Simulation Archives (Primary)**:
   - Automated batch executions of diverse evacuation layouts (hallways, corner turns, multi-door rooms, bottleneck exits) across varying crowd densities (10 to 100 agents).
   - Captures true evacuation dynamics (stress velocity, wall repulsion, exit convergence).
2. **ETH/UCY Benchmark Datasets (Secondary / Pretraining)**:
   - Used exclusively for offline baseline validation and academic benchmarking.
   - Evaluated using an explicit coordinate adapter ($dt = 0.4\text{ s}$, zero-mean coordinates).

### 5.2 Slicing Simulation Logs into Training Samples
- Trajectory logs are sliced using a sliding window:
  - Observation window: $T_{obs} = 20$ steps (2.0s).
  - Target future window: $T_{pred} = 30$ steps (3.0s).
- **Leakage Prevention**:
  - Trajectory slicing is partitioned at the **Scenario / Simulation Run level** (e.g., 70% of scenario seeds for training, 15% for validation, 15% for held-out testing).
  - Sliding windows from the same scenario run are **never** split across training and test sets.

### 5.3 Handling Edge Cases
- **Newly Spawned Pedestrians ($< T_{obs}$ observations)**:
  - Zero-padded with explicit validity mask. Padded frames are set to zero displacement and excluded from velocity computation via a boolean history_valid_mask tensor. The model must distinguish padded frames from real observations.
- **Evacuating / Disappearing Pedestrians**:
  - If a pedestrian reaches an exit during the future horizon, their target trajectory is masked past the exit step. The training loss is computed only over valid active timesteps.

