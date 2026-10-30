# Phase H: Model Options & Strategic Recommendation

## 1. Candidate Architectural Approaches

We evaluate three distinct trajectory prediction paradigms for GenEvac-Sentinel:

### Approach A: Pure Kinematic Constant Velocity (CV) Baseline
- **Description**: Closed-form linear extrapolation based solely on the historical displacement vector of each individual pedestrian:
  $$\Delta x = x_t - x_{t-1}, \quad \Delta y = y_t - y_{t-1}, \quad \hat{p}_{t+h} = (x_t + h \cdot \Delta x, \; y_t + h \cdot \Delta y)$$
- **Strict Boundary**: Relies **exclusively** on observed trajectory points. It does **not** consume exit locations, destination coordinates, obstacle geometry, or wall maps.
- **Why this separation is essential**: Adding exit attraction and wall repulsion into a baseline model turns it into a movement planner rather than a trajectory predictor. Keeping the baseline purely kinematic provides an honest, leak-free benchmark against which learned models can be compared.
- **Empirical CPU Performance**: $0.025\text{ ms}$ for 10 agents, $0.143\text{ ms}$ for 50 agents ($2.86\text{ µs/agent}$). Zero external dependencies.

### Approach B: Lightweight Learned Model (Sequence-to-Sequence CPU GRU / TCN)
- **Description**: Modern neural network consisting of a trajectory history encoder (e.g. 2-layer GRU or 1D Temporal Convolutional Network) predicting multi-step $\Delta x, \Delta y$ velocity offsets.
- **Input Scope**: Standardized historical displacement tensors $(B, T_{obs}, 2)$. Can be optionally extended in later iterations to take neighboring agent interaction grids.
- **Empirical CPU Performance**: Measured on host CPU (Intel Arc family, PyTorch 2.4.1+cpu):
  - 10 agents: $2.55\text{ ms}$ model forward pass + $0.17\text{ ms}$ tensor conversion = $2.72\text{ ms}$ total.
  - 50 agents: $9.84\text{ ms}$ model forward pass + $0.27\text{ ms}$ tensor conversion = $10.11\text{ ms}$ total.
  - 100 agents: $9.58\text{ ms}$ model forward pass + $0.48\text{ ms}$ tensor conversion = $10.06\text{ ms}$ total.
  - 250 agents: $15.61\text{ ms}$ model forward pass + $1.19\text{ ms}$ tensor conversion = $16.80\text{ ms}$ total.
- **Verdict**: Highly feasible for online or quasi-online prediction at 1–2 Hz query frequency, or offline post-processing.

### Approach C: MID-Inspired Motion Indeterminacy Diffusion (DDIM Fast Sampling)
- **Description**: Generative reverse diffusion model predicting future trajectories by iteratively denoising Gaussian noise conditioned on historical states and social embeddings.
- **Computational Limitation on CPU**: Full 100-step DDPM requires $> 1500\text{ ms}$ per batch on CPU. Even a 2-step DDIM model requires multiple forward passes through a heavy Transformer backbone, resulting in estimated latencies $> 80 - 150\text{ ms}$ on CPU.
- **Verdict**: Reserved strictly for offline academic evaluation against ETH/UCY benchmarks. Not suitable for active simulation loops without GPU acceleration.

---

## 2. Empirically Grounded Evaluation Matrix

| Criterion | Approach A: Kinematic Constant Velocity | Approach B: Lightweight Learned GRU (CPU) | Approach C: MID Diffusion (DDPM/DDIM) |
|---|---|---|---|
| **Architectural Nature** | Pure Trajectory Predictor | Pure Trajectory Predictor | Stochastic Generative Predictor |
| **Environmental Coupling** | None (History only) | None (History only; optional social grid) | None (History + dynamic social graph) |
| **Movement Planning Risk** | **Zero** | **Zero** | **Zero** |
| **Data Requirements** | None | Moderate ($10^3 - 10^4$ sequences) | High ($10^5$ sequences) |
| **Training Time** | None (instantaneous) | Fast ($\approx 10 - 20$ mins on CPU) | Expensive ($\ge 12 - 24$ hours on GPU) |
| **Pretrained Weights** | N/A (analytic) | Can be trained cleanly in-repo | **None** (must train from scratch) |
| **Measured Latency (50 agents, CPU)** | **0.14 ms** | **10.11 ms** (including tensor conversion) | $> 100\text{ ms}$ (DDIM) / $> 1.5\text{ s}$ (DDPM) |
| **Measured Latency (100 agents, CPU)**| **0.26 ms** | **10.06 ms** (including tensor conversion) | $> 200\text{ ms}$ (DDIM) / $> 3.0\text{ s}$ (DDPM) |
| **Simulator Compatibility (10 Hz)** | Native | Native (or 2 Hz downsampled query) | Requires subsampling / interpolation |
| **Dependency & Platform Risk** | Zero | Zero (standard PyTorch CPU) | High (legacy dependencies, CUDA) |
| **Suitability for Project Schedule** | **Immediate** | **High** | **Low** |

---

## 3. Strategic Recommendations

1. **Immediate Production Track (Part 2 & 3)**:
   - Implement the **Pure Kinematic Constant Velocity Predictor** as the baseline benchmark.
   - Do **not** incorporate exit targets, goal attraction, or obstacle geometry into this baseline.
2. **Advanced Machine Learning Track (Part 4 & 5)**:
   - Implement the **Lightweight PyTorch Sequence Model (GRU)** trained on pure trajectory history.
   - Separate any future goal-conditioned or interaction-aware predictor into an independent experimental model that explicitly declares environment dependencies.
3. **Latency Governance**:
   - Do **not** assume an arbitrary 100 ms budget for the simulation loop. Person A must confirm the actual real-time constraints. If the simulation demands $\le 10\text{ ms}$ total step time, neural prediction should either run at a lower frequency (e.g. 1 Hz) or offline.
