# Phase B: MID Research Implementation Review

## 1. Research Overview & Theoretical Foundations

### 1.1 Citation & Provenance
- **Paper**: *"Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"*, CVPR 2022 (Oral).
- **Authors**: Tianpei Gu, Guangyi Chen, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou, Jiwen Lu (Tsinghua University).
- **Official Repository**: `https://github.com/Gutianpei/MID.git`
- **Inspected Commit**: `3b61674`

### 1.2 Core Thesis: Motion Indeterminacy Diffusion (MID)
Human motion exhibits multi-modality and indeterminacy, especially at path bifurcations, doorways, and obstacle bypasses. Prior methods (e.g., Social GAN, Trajectron++) rely on one-shot latent variable mappings ($z \sim \mathcal{N}(0, I)$), which often suffer from mode collapse, posterior collapse, or unrealistic sample diversity.

MID formulates trajectory prediction as a **progressive reverse diffusion process**:
1. **Forward Process ($q$)**: Progressively adds Gaussian noise to future trajectories over $T$ steps (default $T=100$), transforming structured ground-truth trajectories $y_0$ into pure isotropic noise $y_T \sim \mathcal{N}(0, I)$.
2. **Reverse Process ($p_\theta$)**: A parameterized neural network progressively denoises $y_T \to y_0$ conditioned on:
   - Historical motion states $x_{1:T_{obs}}$.
   - Social interaction embeddings derived from neighboring pedestrians.
   - Spatial environment context.
3. **DDIM Fast Sampling (2023 update)**: Denoising Diffusion Implicit Models (Song et al.) allow skipping steps non-Markovianly, reducing reverse sampling to 2 to 5 steps while preserving trajectory quality.

---

## 2. Input, Output, and Representation Conventions

### 2.1 Spatial-Temporal Assumptions
- **History Length ($T_{obs}$)**: 8 timesteps (equivalent to 3.2 seconds at 2.5 Hz).
- **Prediction Horizon ($T_{pred}$)**: 12 timesteps (equivalent to 4.8 seconds at 2.5 Hz).
- **Sampling Interval ($dt$)**: 0.4 seconds (2.5 frames per second, downsampled from 25 fps raw video).
- **Spatial Coordinates**: Rectified 2D ground-plane Cartesian coordinates $(x, y)$ in meters. In ETH/UCY datasets, coordinates are zero-centered per scene.

### 2.2 Tensor Shapes & Dimensions
- **Input History Tensor**: $(B, T_{obs}, D_{state})$, where $D_{state}=6$ (position $(x, y)$, velocity $(v_x, v_y)$, acceleration $(a_x, a_y)$).
- **Prediction Tensor**: $(B, K, T_{pred}, 2)$, where:
  - $B$: Batch size (number of target agents).
  - $K$: Number of sampled candidate trajectories (default $K=20$).
  - $T_{pred} = 12$: Future steps.
  - $2$: Predicted future Cartesian coordinates $(\hat{x}, \hat{y})$.
- **Evaluation Metric**: Minimum ADE/FDE across the $K=20$ candidate predictions ("Best-of-20").

---

## 3. Preprocessing Pipeline (`process_data.py`)

1. **Raw Ingestion**: Reads tab-separated `.txt` tracking files (`frame_id`, `track_id`, `x`, `y`).
2. **Subsampling**: Divides frame IDs by 10 (`frame_id // 10`) to achieve 2.5 Hz downsampling.
3. **Coordinate Centering**: Subtracts mean coordinates per scene (`data['pos_x'] -= data['pos_x'].mean()`).
4. **Kinematic Derivatives**: Computes velocity and acceleration via finite differences:
   $$v_t = \frac{x_t - x_{t-1}}{dt}, \quad a_t = \frac{v_t - v_{t-1}}{dt}$$
5. **Standardization**: Applies z-score normalization using dataset statistics:
   - Position: mean = 0, std = 1
   - Velocity: mean = 0, std = 2
   - Acceleration: mean = 0, std = 1
6. **Data Structure Assembly**: Constructs Trajectron++ `Environment`, `Scene`, and `Node` graph objects and pickles them using `dill`.

---

## 4. Architectural Analysis: Adopt vs. Discard

| Architectural Component | Assessment for GenEvac-Sentinel | Recommendation |
|---|---|---|
| **Diffusion Formulation (DDPM/DDIM)** | High theoretical value for modeling crowd bifurcation around bottlenecks and exits. | **Adopt Conceptually**: Design a lightweight 2-step reverse diffusion model or latent diffusion predictor for Module B. |
| **History & Horizon ($8 \to 12$)** | Standard benchmark setting ($3.2\text{ s} \to 4.8\text{ s}$). Our simulator runs at $dt=0.1$ s (10 Hz). | **Adapt**: Map 10 Hz simulation observations (e.g. 20 steps = 2.0 s) to forecast horizon (e.g. 30 steps = 3.0 s). |
| **Monolithic Trajectron++ Graph Dependency** | Overly complex dynamic scene graph requiring `dill` serialization, legacy `torch.nn` modules, and extensive boilerplate. | **Discard**: Implement a clean PyTorch tensor-based interaction module (e.g. lightweight Multi-Head Attention or Graph Convolution). |
| **Unmodified Gutianpei/MID Codebase** | Legacy Python 3.7 / PyTorch 1.7.1 code, hardcoded CUDA device calls, unhandled script bugs, absent weights. | **Quarantine**: Retain strictly in `research/MID/` for reference; do **not** vendor into `src/genevac/`. |

