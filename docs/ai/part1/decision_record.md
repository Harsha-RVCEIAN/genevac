# Phase I: Part 1 Decision Record & Governance (Revised)

## 1. Context & Scope

- **Project**: GenEvac-Sentinel
- **Role**: Person B (AI, Trajectory Prediction & Evaluation)
- **Phase**: Part 1 (AI Research, MID Methodology Analysis, and Prediction System Planning)
- **Date**: 2026-10-30 (Updated)

---

## 2. Core Clarification: Three Separate Engineering Questions

Previous feasibility analyses conflated execution of the legacy research repository with our actual project goal. We now explicitly separate the three distinct questions:

| Question | Assessment | Impact on GenEvac-Sentinel |
|---|---|---|
| **Question 1: Can we understand the original MID methodology?** | **YES (Verified)** | The mathematical formulation, variance schedule ($\beta_1 \dots \beta_T$), forward diffusion process, reverse denoising equations, FiLM conditioning (`ConcatSquashLinear`), Transformer temporal backbone, and DDIM fast sampling are completely understood, verified against the source code, and documented in `docs/ai/part1/mid_methodology_analysis.md`. |
| **Question 2: Can we execute the original reference implementation in an isolated environment?** | **IN PROGRESS (Workstream B / Blocked on Checkpoints)** | The original repository lacks official pretrained checkpoints, contains script bugs (`main.py:29`, `main.py:46`), and hardcodes CUDA. This investigation is conducted in an isolated research environment (`research/mid_reference/.venv`) strictly for reference validation. |
| **Question 3: Can we independently implement a GenEvac-compatible model based on the MID methodology?** | **YES (Feasible & Sound)** | Our final deliverable is **our own native MID predictor** built in `src/genevac/prediction/mid/`, operating on Part 2 `TrajectorySnapshot` inputs and top-left Cartesian coordinates. **Execution failure of Question 2 does NOT block Question 3.** |

---

## 3. Verified Methodology & Architectural Decisions

### 3.1 What We Are Adopting
1. **Diffusion Paradigm**: Forward Gaussian noise addition $q(y_t | y_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} y_0, (1 - \bar{\alpha}_t) I)$ and parameterized reverse denoising $p_\theta(y_{t-1} | y_t, c)$.
2. **Conditioning Mechanism**: Contextual feature modulation (`ConcatSquashLinear` / FiLM) combining past trajectory history with diffusion time embeddings $[\beta_t, \sin(\beta_t), \cos(\beta_t)]$.
3. **Temporal Attention**: Multi-head self-attention across future timesteps to enforce trajectory smoothness and continuity.
4. **DDIM Fast Sampling**: 2-to-5 step inference to reduce denoiser evaluations; CPU latency target is 20–40 ms (untested; benchmark required).
5. **Multi-Modal Generation**: Sampling $K=20$ candidates to capture path bifurcations around evacuation obstacles.

### 3.2 What We Are Discarding or Replacing
1. **Trajectron++ Monolithic Graph**: Discarded. Replaced with clean PyTorch Spatial-Distance Multi-Head Attention.
2. **Hardcoded CUDA Allocations**: Discarded. Replaced with dynamic device dispatch (`torch.device`).
3. **Legacy Dependencies (`dill`, `easydict`, `tensorboardX`)**: Discarded from production `genevac`.
4. **Coupled Raw File Pickling**: Discarded. Replaced with native PyTorch datasets parsing simulation trajectory logs.

---

## 4.5 Reference Implementation Execution Status

The following table distinguishes what was actually achieved during the Workstream B investigation:

| Stage | Status |
|---|---|
| Source-code inspection (architecture, math mapping) | COMPLETE |
| Import and configuration validation (isolated venv) | COMPLETE |
| Synthetic component testing (denoiser forward pass with random tensors) | COMPLETE — Stages A–D passed |
| Full DDIM reverse-sampling validation (end-to-end) | NOT DEMONSTRATED |
| Pretrained checkpoint loading and inference | BLOCKED — No official checkpoints distributed |
| Benchmark evaluation (ADE/FDE on ETH/UCY) | NOT PERFORMED |

Stages A–D passing confirms the reference code can be imported and individual components can process synthetic tensors. It does NOT confirm that the model generates valid trajectories with trained weights.

---

## 4. Formal Decision: **PARTS 1, 2, AND 3 AUDITED & VERIFIED — READY FOR PART 4**

1. **Original Gutianpei/MID Reference**: **REFERENCE ONLY (Quarantined in `research/MID/`)**.
   - Not adopted as project code. Investigated in `research/mid_reference/.venv` for methodology validation.
2. **Native GenEvac Trajectory Prediction System**: **APPROVED TO PROCEED**.
   - Part 2 data contract, deep immutability, timestamp-gap handling, and category isolation are fully implemented and verified.
   - Part 3 (Pure Kinematic Constant Velocity baseline) is fully implemented (`ConstantVelocityPredictor` in `src/genevac/prediction/baseline.py`) with 20 passing unit/edge tests (70 total tests across suite).
   - Part 4 & 5 (Native GenEvac MID dataset & diffusion architecture) will follow the established roadmap.

---

## 5. Next Steps Roadmap

1. **Part 4 — Simulation Training Data Pipeline**: Slicing GenEvac simulation scenario runs into $(X_{\text{obs}}, Y_{\text{future}})$ pairs with boolean validity masks and zero-leakage splits.
2. **Part 5 — Native GenEvac MID Diffusion Architecture**: Implement `VarianceSchedule`, `TransformerDenoiser`, and `DDIMSampler` natively under `src/genevac/prediction/mid/`.
3. **Part 6–8 — Training, Snapshot Runtime Integration & Evaluation**: Train on simulation scenarios and evaluate head-to-head against the CV baseline.
