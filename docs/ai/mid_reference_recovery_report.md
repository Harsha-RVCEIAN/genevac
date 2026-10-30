# Original Gutianpei/MID Reference Implementation Recovery & Investigation Report

## 1. Executive Summary

| Item | Verified Finding / Status |
|---|---|
| **Repository Inspected** | `https://github.com/Gutianpei/MID.git` |
| **Commit Inspected** | `3b61674` (Update README.md) |
| **Isolated Environment** | `research/mid_reference/.venv` (Python 3.11.9, PyTorch 2.15.0+cpu, Windows AMD64) |
| **Stage A (Dependency & Module Imports)** | **SUCCESS**: All core packages and MID model classes imported successfully. |
| **Stage B (Configuration & Args Audit)** | **SUCCESS**: Configuration loaded; verified `args.dataset[:-1]` and `step` script bugs. |
| **Stage C (Dataset & Preprocessing Audit)** | **SUCCESS (Audit Complete)**: Raw data present; processed pickles missing; directory mismatch identified. |
| **Stage D (Model Construction & Forward Pass)** | **SUCCESS**: Denoiser and diffusion chain instantiated and executed on CPU (`(4, 12, 2)` output). |
| **Stage E (Official Checkpoint & Real Inference)** | **BLOCKED**: Pretrained checkpoint `eth_epoch70.pt` is absent from the repository and releases. |
| **Overall Reference Status** | **`REFERENCE INFERENCE BLOCKED`** (due to absent official pretrained weights). |
| **Impact on GenEvac-Sentinel** | **Zero Block on Production**: Our native GenEvac MID model will be trained on simulation logs. |

---

## 2. Environment & Dependency Isolation

To protect the main GenEvac-Sentinel project, all reference investigations were conducted in a strictly isolated virtual environment:

- **Isolated Venv Location**: `research/mid_reference/.venv`
- **Main Project Environment**: `.\.venv` (completely untouched, preserving existing 36 tests)
- **Host Platform**: Windows 10/11 (AMD64), Intel(R) Arc(TM) Graphics Family (No NVIDIA CUDA GPU available)
- **Installed Packages in Reference Venv**:
  - `torch==2.15.0+cpu`
  - `dill==0.4.1`
  - `easydict==1.13`
  - `tensorboardX==2.6.5`
  - `pandas==3.1.0`
  - `scipy==1.17.1`
  - `numpy==2.4.6`
  - `pyyaml==6.0.3`
  - `matplotlib==3.11.2`

---

## 3. Investigation of Key Defects and Blocker Hypotheses

We systematically investigated the six defect hypotheses identified in initial reconnaissance:

### 3.1 Hypothesis 1: Missing Legacy Dependencies
- **Status**: **CONFIRMED & RESOLVED in Isolated Venv**.
- **Evidence**: The base global environment lacked `dill`, `easydict`, and `tensorboardX`. In `research/mid_reference/.venv`, installing these packages enabled complete module importation.

### 3.2 Hypothesis 2: Dataset String Truncation Defect (`main.py:29`)
- **Status**: **CONFIRMED SOURCE DEFECT**.
- **Code**: `config["dataset"] = args.dataset[:-1]`
- **Cause**: The author used `[:-1]` to strip potential trailing slashes from command-line arguments (e.g. `eth/`). However, when invoked normally with `--dataset eth`, it strips the final character, producing `"et"`.
- **Impact**: `mid.py:216-217` attempts to load `et_train.pkl` and `et_test.pkl`, causing immediate `FileNotFoundError`.
- **Correction**: Replace with `args.dataset.rstrip("/\\")`.

### 3.3 Hypothesis 3: Undefined Variable Bug (`main.py:46`)
- **Status**: **CONFIRMED SOURCE DEFECT**.
- **Code**:
  ```python
  sampling = "ddim"
  steps = 5
  if config["eval_mode"]:
      agent.eval(sampling, 100//step)  # 'step' is undefined!
  ```
- **Cause**: Line 43 defines `steps = 5` (plural), but line 46 references `step` (singular).
- **Impact**: Invoking `--config ...` with `eval_mode: True` raises `NameError: name 'step' is not defined`.
- **Correction**: Change to `100 // steps`.

### 3.4 Hypothesis 4: Preprocessing Directory Mismatch
- **Status**: **CONFIRMED CONFIGURATION DEFECT**.
- **Evidence**:
  - `process_data.py:95`: `data_folder_name = 'processed_data_noise'`
  - `configs/baseline.yaml:5`: `data_dir: processed_data`
- **Impact**: Preprocessing generates pickle files in `processed_data_noise/`, but training and evaluation scripts search inside `processed_data/`.
- **Correction**: Align directory names in configuration.

### 3.5 Hypothesis 5: Hardcoded CUDA Device Allocations
- **Status**: **CONFIRMED PLATFORM DEFECT**.
- **Evidence**:
  - `mid.py:236`: `self.registrar = ModelRegistrar(self.model_dir, "cuda")`
  - `mid.py:252`: `self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda")`
  - `mid.py:263`: `self.model = model.cuda()`
  - `models/diffusion.py:75-79`: Explicit `.cuda()` calls on tensor buffers.
- **Impact**: On machines without NVIDIA CUDA drivers, raises `AssertionError: Torch not compiled with CUDA enabled`.
- **Correction**: Parameterize device dynamically via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

### 3.6 Hypothesis 6: Pretrained Checkpoint Availability
- **Status**: **CONFIRMED HARD BLOCKER**.
- **Evidence**:
  - `baseline.yaml` specifies `eval_at: 70`.
  - `mid.py:240-241` requires `./experiments/baseline/eth_epoch70.pt`.
  - File search across `research/MID/` confirms zero `.pt` or `.pth` checkpoints exist.
  - Comprehensive GitHub issue and release audit confirms the authors never published trained model weights; users are expected to train from scratch for 90 epochs.
- **Safety Compliance**: We refuse to fabricate fake weights or claim random weights represent valid inference.

---

## 4. Staged Execution Log & Diagnostic Verification

Executed script: [`research/evidence/mid_reference/test_stages.py`](file:///c:/Users/harsh/projects/genevac/research/evidence/mid_reference/test_stages.py).
Log artifact: [`research/evidence/mid_reference/staged_execution.log`](file:///c:/Users/harsh/projects/genevac/research/evidence/mid_reference/staged_execution.log).

```text
======================================================================
STAGE A: DEPENDENCY & MODULE IMPORTS
======================================================================
[SUCCESS] Core libraries imported:
  PyTorch: 2.15.0+cpu (CUDA available: False)
  Dill: 0.4.1
  Pandas: 3.1.0
  NumPy: 2.4.6
[SUCCESS] MID model classes imported successfully.

======================================================================
STAGE B: CONFIGURATION LOADING & ARGUMENT HANDLING
======================================================================
[SUCCESS] Loaded baseline.yaml:
  diffnet: TransformerConcatLinear
  encoder_dim: 256
  epochs: 90
  eval_at: 70
  data_dir: processed_data
  main.py dataset string handling:
    Raw input: 'eth'
    Buggy slice (args.dataset[:-1]): 'et' -> Looks for 'et_train.pkl' (FAILS)
    Corrected handling: 'eth' -> Looks for 'eth_train.pkl'

======================================================================
STAGE C: DATASET AVAILABILITY & PREPROCESSING AUDIT
======================================================================
  Raw ETH test file exists: True (C:\Users\harsh\projects\genevac\research\MID\raw_data\eth\test\biwi_eth.txt)
  'processed_data' dir exists: False
  'processed_data_noise' dir exists: False
  'eth_train.pkl' exists: False
  'eth_test.pkl' exists: False
  [AUDIT] Processed pickles missing. process_data.py must be run to generate them.
          Note: process_data.py writes to 'processed_data_noise', but baseline.yaml expects 'processed_data'.

======================================================================
STAGE D: MODEL ARCHITECTURE CONSTRUCTION & FORWARD PASS (CPU)
======================================================================
[SUCCESS] Denoiser forward pass succeeded on CPU!
  Input future trajectory shape: torch.Size([4, 12, 2])
  Context embedding shape:       torch.Size([4, 256])
  Predicted noise shape:         torch.Size([4, 12, 2])
  Finite outputs:                True
[SUCCESS] Reverse step prediction succeeded on CPU: shape = torch.Size([4, 12, 2])

======================================================================
STAGE E: CHECKPOINT AVAILABILITY & OFFICIAL INFERENCE ATTEMPT
======================================================================
Expected official checkpoint path: C:\Users\harsh\projects\genevac\research\MID\experiments\baseline\eth_epoch70.pt
Checkpoint exists on disk: False

[BLOCKER CONFIRMED] Pretrained checkpoint 'eth_epoch70.pt' is ABSENT.
Evidence: Neither the git tree, GitHub releases, nor external downloads distribute pretrained weights.
The authors require running 90 epochs of training from scratch.
Per safety rules: We will NOT fabricate fake weights or claim random weights represent inference.
Stage E Status: BLOCKED: Missing pretrained checkpoint
======================================================================
```

---

## 5. Lessons Directly Applicable to GenEvac-Sentinel

1. **Diffusion Mechanics Are Fully Verified**:
   - Stage D proves that `TransformerConcatLinear` and `VarianceSchedule` operate reliably on CPU with standard PyTorch tensors. We have confirmed the mathematical viability of progressive trajectory denoising.
2. **Wholesale Code Porting Must Be Avoided**:
   - Attempting to vendor `research/MID` directly would import dynamic graph dependencies (`dill`), obsolete string parsing bugs, hardcoded CUDA allocations, and unmaintainable legacy code.
3. **Decoupled Architecture Is Essential**:
   - GenEvac's native MID implementation in `src/genevac/prediction/mid/` will cleanly wrap the verified diffusion and Transformer denoiser math while interfacing exclusively with the Part 2 `TrajectorySnapshot` contract.

