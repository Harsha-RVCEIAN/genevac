# GenEvac-Sentinel Part 1 & Part 2 Audit Correction Report

**Project**: GenEvac-Sentinel  
**Role**: Person B (AI & Trajectory Prediction)  
**Date**: 2026-10-30  
**Branch**: `harsha`  
**Execution Environment**: Windows, Python 3.11.9 (`.venv`), PyTorch 2.4.1+cpu  

---

## 1. Executive Summary

This report documents the audit and corrections applied across the **Part 1 AI Research / Methodology documents**, the **Part 2 Trajectory Data Contract and History Buffer**, and the newly implemented **Part 3 Timestamp-Aware Constant Velocity (CV) Baseline Predictor**.

All non-negotiable project boundaries were strictly observed:
- **Zero changes** to Person A's simulation engine, pedestrian lifecycle, environment, config, or execution loop.
- **Zero changes** to unrelated project modules.
- **Complete backward compatibility** preserved for existing interfaces.
- **All 70 automated tests pass** without failures or skips (up from 36 tests).

---

## 2. Issues Discovered & Audit Summary

| Issue # | Component | Discovered Condition | Corrective Action |
|---|---|---|---|
| **2.1** | `mid_methodology_analysis.md` | Claimed forward Gaussian prior $y_T$ "covers the walkable domain". | Corrected to state that prior exists in trajectory representation space, not physical space. Clarified that diffusion does not inherently enforce obstacle/wall boundaries without conditioning or explicit constraints. |
| **2.2** | `mid_methodology_analysis.md`, `genevac_mid_adaptation_design.md` | Ambiguous future-target definition (mixing positions, velocity derivatives, and normalized coordinates). | Standardized target parameterization as cumulative local displacement $Y_k = p_{t+k} - p_0$ with tensor shape $(B, H, 2)$ and global reconstruction $\hat{p}_{t+k} = p_0 + \hat{Y}_k$. Documented contrast with original MID velocity diffusion. |
| **2.3** | `genevac_mid_adaptation_design.md` | Overclaimed that lightweight attention "achieves the same relational modeling in <2ms" as Trajectron++. | Re-framed as a lightweight spatial-attention adaptation inspired by MID's interaction objective. Specified neighbor radius ($R=5.0\text{ m}$), latest position Euclidean distance, relative features, padding masks, zero-neighbor handling, and permutation invariance. |
| **2.4** | `genevac_mid_adaptation_design.md`, `implementation_roadmap.md` | Contradictory short-history padding descriptions (linear back-padding vs constant velocity extrapolation vs stationary). | Standardized on zero-padding with an explicit boolean `history_valid_mask`. Velocities are computed only from real observations; padded steps are excluded. |
| **2.5** | `mid_methodology_analysis.md`, `genevac_mid_adaptation_design.md`, `decision_record.md` | Unverified performance figures (20–40 ms CPU latency, 50× speedup, <2 ms social attention) presented as facts. | Relabeled all performance numbers as theoretical targets and estimates requiring host benchmarking across full pipelines (history preparation through coordinate reconstruction). |
| **2.6** | `mid_methodology_analysis.md`, `decision_record.md` | Investigation status conflated static source inspection and synthetic forward passes with full reference inference. | Added explicit 6-stage execution status matrix clearly showing stages A–D passed (synthetic forward passes) while checkpoint inference is blocked and benchmark evaluations were not performed. |
| **3.1** | `data_contract.py`, `tests/test_data_contract.py` | Need for explicit verification that skipped simulation steps (e.g., steps 10, 11, 13) are accepted when timestamps match $\Delta t \times \text{step\_diff}$. | Added unit tests verifying skipped-step acceptance in both `TrajectoryValidator.validate_observation_sequence` and `TrajectoryHistoryBuffer.record`. |
| **3.2** | `__init__.py`, `tests/test_data_contract.py` | `TrajectorySnapshot.trajectories` dictionary was mutable; external callers could modify snapshot entries. | Wrapped `trajectories` dictionary in `types.MappingProxyType` upon snapshot construction. Added tests asserting `TypeError` on assignment and isolating snapshots from subsequent buffer modifications. |
| **3.3** | `__init__.py`, `tests/test_data_contract.py` | `TrajectoryHistoryBuffer.record()` accepted observations categorized as `TrajectoryCategory.PREDICTED`. | Added explicit validation check rejecting `PREDICTED` observations across both object and field-based recording paths, raising `TrajectoryValidationError` and leaving the cache unmodified. |
| **3.4** | `docs/data_contract.md`, `genevac_mid_adaptation_design.md` | Contradictory mention of scene-mean centering ($x - \bar{x}$) in Section 14 of `data_contract.md`. | Replaced Section 14 with strict agent-centric transformation policy ($p_{\text{local}} = p_{\text{global}} - p_0$). Documented why scene-mean centering is prohibited in GenEvac. |
| **4.1** | `baseline.py`, `tests/test_baseline.py` | Old `LinearTrajectoryPredictor` only supported displacement per observation without timestamp awareness. | Implemented `ConstantVelocityPredictor` consuming `TrajectorySnapshot` and producing `TrajectoryPrediction`, calculating $v = \Delta p / \Delta t$ and future timestamps $t_{\text{future}} = t_{\text{source}} + k \times \Delta t$. Added 20 tests. |
| **5.1** | `tests/test_data_contract.py` | Output contract lacked dedicated end-to-end field and timestamp invariant tests. | Added `TestPredictionOutputContract` asserting all 7 output guarantees (fields, strictly greater timestamps, monotonic timestamps, finite coordinates, PREDICTED category, identity preservation, length match). |

---

## 3. Files Inspected

- `src/genevac/simulation/__init__.py` (verified read-only boundaries)
- `src/genevac/pedestrians/__init__.py` (verified `Pedestrian` constructor and fields)
- `src/genevac/environment/__init__.py` (verified geometry structures)
- `src/genevac/config.py` (verified configuration isolation)
- `src/genevac/main.py` (verified application entry point)
- `src/genevac/prediction/data_contract.py`
- `src/genevac/prediction/__init__.py`
- `tests/test_prediction.py`
- `tests/test_data_contract.py`
- `docs/data_contract.md`
- `docs/ai/part1/mid_methodology_analysis.md`
- `docs/ai/part1/genevac_mid_adaptation_design.md`
- `docs/ai/part1/implementation_roadmap.md`
- `docs/ai/part1/decision_record.md`

---

## 4. Files Changed & Created

### Documentation Files Changed:
1. [`docs/ai/part1/mid_methodology_analysis.md`](file:///c:/Users/harsh/projects/genevac/docs/ai/part1/mid_methodology_analysis.md)
   - Corrected diffusion prior explanation (Issues 2.1).
   - Added Subsection 3.1.1 clarifying GenEvac cumulative local displacement $Y_k = p_{t+k} - p_0$ vs original MID velocity sequence diffusion (Issue 2.2).
   - Reframed DDIM 2–5 steps as theoretical speedup with untested latency targets (Issue 2.5).
   - Clarified that source code mapping is static inspection, not end-to-end execution (Issue 2.6).

2. [`docs/ai/part1/genevac_mid_adaptation_design.md`](file:///c:/Users/harsh/projects/genevac/docs/ai/part1/genevac_mid_adaptation_design.md)
   - Cleaned up Section 2 adaptation table formatting.
   - Fixed Row 4 (zero-padding with validity mask), Row 6 (social attention framing), Row 10 (untested latency note).
   - Added Section 2.1 (Coordinate Transformation Note) and Section 2.2 (Social Interaction Conditioning Specification).
   - Standardized Section 5.3 edge case handling on zero-padding + validity masks.

3. [`docs/ai/part1/decision_record.md`](file:///c:/Users/harsh/projects/genevac/docs/ai/part1/decision_record.md)
   - Added Section 4.5 outlining the 6-stage investigation status matrix.
   - Updated Section 3.1 DDIM fast sampling entry to note untested status.
   - Updated Section 4 and Section 5 marking Parts 1, 2, and 3 audited and approved to proceed to Part 4.

4. [`docs/ai/part1/implementation_roadmap.md`](file:///c:/Users/harsh/projects/genevac/docs/ai/part1/implementation_roadmap.md)
   - Updated Stage 1 status to substantially complete with audit corrections.
   - Updated Stage 2 status to complete with audit corrections noted.
   - Updated Stage 3 status to COMPLETE (implemented and verified).
   - Enhanced Stage 8 metrics list with collision rates, off-walkable rates, density breakdowns, and p50/p95 latency.

5. [`docs/data_contract.md`](file:///c:/Users/harsh/projects/genevac/docs/data_contract.md)
   - Replaced Section 14 with the explicit agent-centric coordinate transformation policy ($p_{\text{local}} = p_{\text{global}} - p_0$). Prohibited scene-mean centering in GenEvac.

### Source Files Changed / Created:
6. [`src/genevac/prediction/baseline.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/prediction/baseline.py) *(NEW)*
   - Implemented `ConstantVelocityPredictor` with timestamp-aware velocity computation:
     $$v_x = \frac{x_2 - x_1}{t_2 - t_1}, \quad v_y = \frac{y_2 - y_1}{t_2 - t_1}$$
     $$p_{\text{future}} = p_2 + v \times (t_{\text{future}} - t_2)$$
   - Future timestamps derived strictly from `snapshot.current_timestamp + k * interval`.
   - Comprehensive validation rejecting $< 2$ observations, non-positive horizons, and non-finite numbers.
   - Provided `predict_batch()` returning individual predictions and error objects.

7. [`src/genevac/prediction/__init__.py`](file:///c:/Users/harsh/projects/genevac/src/genevac/prediction/__init__.py)
   - Wrapped `snapshot.trajectories` in `MappingProxyType` to ensure deep immutability.
   - Enforced rejection of `TrajectoryCategory.PREDICTED` observations in `record()`.
   - Re-exported `ConstantVelocityPredictor` in `__all__`.

### Test Files Changed / Created:
8. [`tests/test_data_contract.py`](file:///c:/Users/harsh/projects/genevac/tests/test_data_contract.py)
   - Added `test_buffer_accepts_skipped_step_observations`.
   - Added `test_reject_predicted_category_observation_object`.
   - Added `test_reject_predicted_category_field_recording`.
   - Added `test_snapshot_mapping_is_immutable`.
   - Added `test_snapshot_tuple_observations_are_immutable`.
   - Added `test_snapshot_isolates_from_buffer_mutation`.
   - Added `test_validate_observation_sequence_allows_skipped_steps`.
   - Added `TestPredictionOutputContract` with 7 validation tests.

9. [`tests/test_baseline.py`](file:///c:/Users/harsh/projects/genevac/tests/test_baseline.py) *(NEW)*
   - Added 20 comprehensive unit and contract tests for `ConstantVelocityPredictor`.

---

## 5. Verification Commands & Test Results

### Command 1: Linter Check on Person B Code
```powershell
.\.venv\Scripts\ruff.exe check src/genevac/prediction/ tests/test_prediction.py tests/test_data_contract.py tests/test_baseline.py
```
**Result**: `All checks passed!` (0 errors, 0 warnings).

### Command 2: Part 2 Data Contract Tests
```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\pytest.exe tests/test_data_contract.py -v
```
**Result**: **40 passed** in 0.14s.

### Command 3: Part 3 Baseline Predictor Tests
```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\pytest.exe tests/test_baseline.py -v
```
**Result**: **20 passed** in 0.10s.

### Command 4: Complete Test Suite Run
```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\pytest.exe tests -v
```
**Result**: **70 passed in 0.35s** (100% pass rate).

Test suite breakdown:
- `tests/test_baseline.py`: 20 passed
- `tests/test_data_contract.py`: 40 passed
- `tests/test_prediction.py`: 3 passed
- `tests/test_simulation.py`: 1 passed (Person A)
- `tests/test_pedestrians.py`: 2 passed (Person A)
- `tests/test_metrics.py`: 1 passed
- `tests/test_heatmaps.py`: 1 passed
- `tests/test_export.py`: 1 passed
- `tests/test_experiments.py`: 1 passed

---

## 6. Numerical Validation of the Baseline Predictor

As required by the audit prompt, we verified the documented example:
- **Observation 1**: Position = $(0.0, 0.0)$, Timestamp = $0.0\text{ s}$
- **Observation 2**: Position = $(2.0, 0.0)$, Timestamp = $0.2\text{ s}$
- **Velocity**: $v_x = (2.0 - 0.0) / (0.2 - 0.0) = 10.0\text{ m/s}$, $v_y = 0.0\text{ m/s}$
- **Source timestamp**: $0.2\text{ s}$, **Future step 1** ($k=1, \Delta t=0.1\text{ s}$): $t_{\text{future}} = 0.3\text{ s}$
- **Old (displacement-per-step) output**: $(4.0, 0.0)$ ❌
- **New (timestamp-aware) output**: $2.0 + 10.0 \times (0.3 - 0.2) = (3.0, 0.0)$ ✅ (Verified by `test_displacement_vs_timestamp_aware_numerical_example`).

---

## 7. Confirmation of Ownership Boundaries

- **Person A Simulation Files**: `src/genevac/simulation/`, `src/genevac/pedestrians/`, `src/genevac/environment/`, `src/genevac/config.py`, and `src/genevac/main.py` were **NOT modified**.
- **Pedestrian State Safety**: Verified by `test_no_mutation_of_simulation_owned_pedestrian_state` that prediction never mutates `Pedestrian.position`, `velocity`, or `status`.
- **Authoritative History Buffer Safety**: Verified by `test_reject_predicted_category_observation_object` and `test_reject_predicted_category_field_recording` that prediction results cannot enter the history buffer.

---

## 8. Remaining Limitations & Next Steps

1. **Host Latency Benchmarking**: Latency targets (e.g. 20–40 ms) remain estimates until measured on the host CPU during Part 8.
2. **Pretrained Weights**: The reference MID repository does not distribute pretrained weights; training must be conducted natively in Stage 6 using simulation archives.
3. **Next Step**: The project is verified and ready to proceed to **Stage 4: Independent Training Data Pipeline & Preprocessing** (`src/genevac/prediction/mid/preprocessing.py` and `dataset.py`).

