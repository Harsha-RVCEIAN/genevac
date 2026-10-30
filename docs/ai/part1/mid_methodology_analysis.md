# Motion Indeterminacy Diffusion (MID) Methodology Analysis

## 1. Executive Overview

This document presents a rigorous, verified analysis of the **Motion Indeterminacy Diffusion (MID)** methodology as formulated in the CVPR 2022 paper (*"Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"*, Gu et al.) and implemented in the official reference repository (`https://github.com/Gutianpei/MID.git`, commit `3b61674`).

The purpose of this analysis is to distill the core mathematical, algorithmic, and architectural principles of MID so that **our team can implement our own clean, independent MID-based trajectory prediction system natively within GenEvac-Sentinel**, rather than relying on or embedding the legacy reference codebase.

---

## 2. Core Thesis & Theoretical Foundations

### 2.1 The Trajectory Multi-Modality Problem
Pedestrian trajectory forecasting in shared spaces is fundamentally stochastic and multi-modal. A pedestrian approaching a doorway, corner, or intersection may choose among several distinct paths (e.g. left corridor, right corridor, or stopping).

Conventional stochastic forecasting methods model this indeterminacy using a single latent random variable $z \sim \mathcal{N}(0, I)$ within:
1. **Generative Adversarial Networks (GANs)**: S-GAN, Social-BiGAT. Often suffer from mode collapse and training instability.
2. **Conditional Variational Autoencoders (CVAEs)**: Trajectron++, Social-VAE. Often suffer from posterior collapse where the decoder ignores $z$ and relies solely on deterministic history, or generates blur/averaged paths.

### 2.2 The Motion Indeterminacy Diffusion Concept
MID reformulates trajectory prediction from a single-step latent mapping into a **continuous reverse diffusion process** over a parameterized Markov chain:
- **Forward Process (Indeterminacy Injection)**: Starts from the true future trajectory $y_0$ and progressively destroys structure by adding Gaussian noise over $T$ diffusion timesteps until $y_T$ becomes an isotropic Gaussian distribution $\mathcal{N}(0, I)$ covering trajectory representation space (not physical environment space).

  > **Note**: The Gaussian prior exists in trajectory representation space, NOT physical space. MID does not inherently model walls, obstacles, exits, or collision constraints. Physical validity depends on training data quality, conditioning, and explicitly implemented post-processing.
- **Reverse Process (Indeterminacy Discarding)**: A neural network progressively denoises $y_T \to y_{T-1} \dots \to y_0$, conditioned on an embedding of past motion and social interactions. At each reverse step, indeterminacy is discarded, resolving the trajectory into one of the plausible physical modes.

```text
Forward Process (Diffusion / Noise Injection):
y_0 (True Future) ----> y_1 ----> ... ----> y_t ----> ... ----> y_T ~ N(0, I)

Reverse Process (Denoising / Trajectory Generation):
y_T ~ N(0, I) --------> y_{T-1} -> ... -> y_t ----> ... ----> y_0 (Predicted Trajectory)
                            ^               ^
                            |               |
                 [Conditioning Context: Past Trajectory + Social Graph]
```

---

## 3. Mathematical Formulation

### 3.1 Representation of Trajectories
Let $X = (p_{-H+1}, \dots, p_0)$ denote the observed historical trajectory of a target pedestrian over $H$ observation steps, where $p_t = (x_t, y_t) \in \mathbb{R}^2$.
Let $Y = (p_1, \dots, p_N)$ denote the future trajectory over prediction horizon $N$ (in ETH/UCY benchmarks, $H = 8$ and $N = 12$).

In MID, the diffusion operates directly on the future sequence tensor $y \in \mathbb{R}^{B \times N \times 2}$, where $B$ is batch size (number of pedestrians), $N$ is the number of future timesteps, and $2$ represents Cartesian spatial coordinates $(x, y)$ or velocity derivatives $(v_x, v_y)$.

#### 3.1.1 Representation Note

- **Original MID reference implementation**: The reference implementation diffuses **velocity/displacement sequences** (verified from `models/autoencoder.py`), not raw global positions.
- **GenEvac adaptation parameterization**: We use cumulative local displacement: $Y_k = p_{t+k} - p_0$, where $p_0$ is the last observed position. Tensor shape: $(B, H, 2)$. Global reconstruction: $\hat{p}_{t+k} = p_0 + \hat{Y}_k$.
- **Important**: This is our **proposed GenEvac parameterization**, not necessarily identical to the original MID parameterization. These two representations differ and must not be mixed between training and inference.

### 3.2 Forward Diffusion Process
Given a variance schedule $\beta_1, \beta_2, \dots, \beta_T$, the forward conditional transition probability at step $t \in \{1, \dots, T\}$ is defined as:

$$q(y_t \mid y_{t-1}) = \mathcal{N}(y_t; \sqrt{1 - \beta_t} y_{t-1}, \beta_t I)$$

By defining $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, the forward marginal distribution at any arbitrary diffusion step $t$ can be sampled in closed form:

$$q(y_t \mid y_0) = \mathcal{N}(y_t; \sqrt{\bar{\alpha}_t} y_0, (1 - \bar{\alpha}_t) I)$$

Equivalently, using the reparameterization trick:

$$y_t = \sqrt{\bar{\alpha}_t} y_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 3.3 Reverse Denoising Process
The reverse transition $p_\theta(y_{t-1} \mid y_t, c)$ is conditioned on the context embedding $c$:

$$p_\theta(y_{t-1} \mid y_t, c) = \mathcal{N}(y_{t-1}; \mu_\theta(y_t, t, c), \sigma_t^2 I)$$

Following Ho et al. (DDPM), the mean $\mu_\theta(y_t, t, c)$ is parameterized through a noise prediction network $\epsilon_\theta(y_t, \beta_t, c)$:

$$\mu_\theta(y_t, t, c) = \frac{1}{\sqrt{\alpha_t}} \left( y_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(y_t, \beta_t, c) \right)$$

The variance $\sigma_t^2$ is bounded between $\beta_t$ (inflexible) and $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ (flexible):

$$\sigma_t = \gamma \sqrt{\beta_t} + (1 - \gamma) \sqrt{\tilde{\beta}_t}, \quad \gamma \in [0, 1]$$

### 3.4 Training Objective
The training objective is the simplified variational lower bound on the negative log-likelihood (Mean Squared Error on noise prediction):

$$\mathcal{L}(\theta) = \mathbb{E}_{y_0, c, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} y_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \beta_t, c) \right\|_2^2 \right]$$

During each training iteration:
1. A ground-truth trajectory batch $y_0$ and conditioning context $c$ are sampled.
2. A random diffusion timestep $t \sim \text{Uniform}(\{1, \dots, T\})$ is sampled for each batch element.
3. Standard Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ is added to $y_0$ according to $t$.
4. The neural network predicts $\hat{\epsilon} = \epsilon_\theta(y_t, \beta_t, c)$.
5. Backpropagation minimizes the MSE between $\hat{\epsilon}$ and $\epsilon$.

---

## 4. Reverse Sampling Strategies

### 4.1 Standard DDPM Sampling (Stochastic Markov Chain)
In full DDPM mode, sampling begins with pure noise $y_T \sim \mathcal{N}(0, I)$ and performs $T$ iterative steps:

$$y_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( y_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(y_t, \beta_t, c) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I) \text{ for } t > 1$$

- **Pros**: Complete coverage of the multi-modal distribution.
- **Cons**: Requires $T=100$ sequential neural evaluations per batch ($\approx 1.5 - 3.0\text{ s}$ latency on CPU).

### 4.2 Fast DDIM Sampling (Deterministic Non-Markovian Stride)
In DDIM (Song et al., adopted in MID's April 2023 update), the reverse process is accelerated by taking strides $S > 1$ (e.g. stride = 20 or 50, reducing 100 steps to 5 or 2 steps):

1. Estimate the clean future trajectory $y_0^{(t)}$ from the noisy sample $y_t$:
   $$\hat{y}_0^{(t)} = \frac{y_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(y_t, \beta_t, c)}{\sqrt{\bar{\alpha}_t}}$$
2. Step directly to $t - \text{stride}$:
   $$y_{t - \text{stride}} = \sqrt{\bar{\alpha}_{t - \text{stride}}} \hat{y}_0^{(t)} + \sqrt{1 - \bar{\alpha}_{t - \text{stride}}} \epsilon_\theta(y_t, \beta_t, c)$$

- **Pros**: Enables trajectory generation in only **2 to 5 forward passes**, providing theoretically up to a **$50\times$ speedup (untested on GenEvac hardware)** with minimal metric degradation.
- **Suitability for GenEvac**: Essential for practical integration. Target latency is **20–40 ms (untested target; actual latency depends on batch size and hardware)**.

  > **Note**: These latency figures are estimates derived from literature. Actual benchmark measurements on the target machine (Intel Arc, CPU-only) have not yet been taken. Latency must be measured for the complete inference pipeline including preprocessing, encoding, denoising, and coordinate reconstruction.

---

## 5. Reference Source-Code Verification & Mapping

The original codebase (`research/MID/`) structures these mechanisms across several modules:

> **IMPORTANT**: The source-code mapping below reflects static code inspection of the reference repository. It does NOT represent successful execution of inference, checkpoint loading, or benchmark evaluation. See `docs/ai/part1/decision_record.md` for the exact execution status of each investigation stage.

| Mathematical / Algorithmic Concept | Original Source File & Class/Function | Verified Code Mechanism |
|---|---|---|
| **Beta & Alpha Variance Schedules** | `models/diffusion.py:9-59`<br>`class VarianceSchedule` | Precomputes $\beta_t, \alpha_t, \bar{\alpha}_t, \sigma_t$ for linear ($\beta_1=10^{-4}, \beta_T=0.05$) or cosine modes. Stores them in buffers. |
| **Forward Noise Injection & Loss** | `models/diffusion.py:67-84`<br>`DiffusionTraj.get_loss()` | Computes $c_0 = \sqrt{\bar{\alpha}_t}$, $c_1 = \sqrt{1 - \bar{\alpha}_t}$, injects `e_rand`, calls `net(c0*x_0 + c1*e_rand, beta, context)`, returns `F.mse_loss`. |
| **Reverse Sampling Loop** | `models/diffusion.py:86-127`<br>`DiffusionTraj.sample()` | Loops $t$ from $T$ down to 1 with stride. Supports branching for `ddpm` and `ddim` formulations. |
| **Conditioning Feature Injection** | `models/common.py:58-72`<br>`class ConcatSquashLinear` | Implements feature-wise gating and modulation: `ret = self._layer(x) * sigmoid(self._hyper_gate(ctx)) + self._hyper_bias(ctx)`. |
| **Diffusion Temporal Backbone** | `models/diffusion.py:172-215`<br>`class TransformerConcatLinear` | Embeds diffusion time $[\beta, \sin(\beta), \cos(\beta)]$ + context $c$, applies `ConcatSquashLinear`, positional encoding, and standard PyTorch `TransformerEncoderLayer`. |
| **End-to-End Orchestrator** | `models/autoencoder.py:10-50`<br>`class AutoEncoder` | Coordinates encoder `obtain_encoded_tensors()`, calls diffusion loss and sample, integrates velocity into position via dynamics. |

---

## 6. Distinguishing Core Methodology from Implementation Artifacts

To successfully build our own GenEvac MID predictor, we must separate the fundamental scientific methodology from accidental implementation choices made in the 2022 research code:

### Core Methodology (To Adopt and Preserve)
1. **Diffusion-Based Formulation**: Formulating trajectory prediction as progressive denoising from $\mathcal{N}(0, I)$ rather than one-shot CVAE/GAN regression.
2. **Context-Conditioned Reverse Process**: Conditioning the denoiser network on historical movement and spatial neighbors via feature modulation (`ConcatSquashLinear` or FiLM).
3. **Temporal Attention over Future Steps**: Using a Transformer encoder to model dependencies across future timesteps $1 \dots N$.
4. **DDIM Fast Sampling**: Utilizing non-Markovian sampling strides to achieve 2-to-5 step generation.
5. **Multi-Modal Candidate Generation**: Sampling $K$ distinct trajectories ($K=20$) from different Gaussian initializations $y_T^{(k)} \sim \mathcal{N}(0, I)$ to represent motion uncertainty.

### Legacy Implementation Artifacts (To Discard or Replace)
1. **Monolithic Trajectron++ Scene Graphs**: The reference repository depends on an enormous, brittle Trajectron++ graph structure serialized via `dill`. This introduces massive complexity, legacy Python 3.7 C-extensions, and runtime overhead.
   - *GenEvac Decision*: **Replace** with a clean, native PyTorch tensor representation of past agent trajectories and spatial distance matrices.
2. **Hardcoded CUDA Allocations**: The reference code calls `.cuda()` unconditionally across 12 files.
   - *GenEvac Decision*: **Replace** with dynamic device dispatch (`device = torch.device(...)`) fully supporting CPU and MPS/CUDA.
3. **Complex Dynamic Vehicle Nodes**: Trajectron++ contains vehicle unicycle and single-integrator dynamics models unnecessary for pedestrian evacuations.
   - *GenEvac Decision*: **Omit**. Pedestrians operate under 2D kinematic point-mass / displacement dynamics.
4. **Coupled Raw File Ingestion**: The reference script parses ETH text files directly inside `process_data.py`.
   - *GenEvac Decision*: **Replace** with a clean adapter consuming GenEvac's standardized `TrajectorySnapshot` data contract.

