"""
Staged Verification Script for Original MID Reference Implementation
Runs Stage A through Stage E in the isolated reference environment.
"""

import os
import sys
from pathlib import Path

# Add research/MID to sys.path
mid_root = Path(__file__).resolve().parent.parent.parent / "MID"
sys.path.insert(0, str(mid_root))

print("=" * 70)
print("STAGE A: DEPENDENCY & MODULE IMPORTS")
print("=" * 70)
try:
    import torch
    import dill
    import easydict
    import tensorboardX
    import yaml
    import numpy as np
    import pandas as pd
    print(f"[SUCCESS] Core libraries imported:")
    print(f"  PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print(f"  Dill: {dill.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print(f"  NumPy: {np.__version__}")

    # Import MID modules
    from models.diffusion import DiffusionTraj, VarianceSchedule, TransformerConcatLinear
    from models.common import ConcatSquashLinear
    from utils.model_registrar import ModelRegistrar
    print(f"[SUCCESS] MID model classes imported successfully.")
    stage_a_ok = True
except Exception as e:
    print(f"[FAILURE] Stage A failed: {e}")
    stage_a_ok = False


print("\n" + "=" * 70)
print("STAGE B: CONFIGURATION LOADING & ARGUMENT HANDLING")
print("=" * 70)
try:
    config_path = mid_root / "configs" / "baseline.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[SUCCESS] Loaded {config_path.name}:")
    print(f"  diffnet: {cfg.get('diffnet')}")
    print(f"  encoder_dim: {cfg.get('encoder_dim')}")
    print(f"  epochs: {cfg.get('epochs')}")
    print(f"  eval_at: {cfg.get('eval_at')}")
    print(f"  data_dir: {cfg.get('data_dir')}")

    # Test main.py dataset string handling defect
    raw_dataset_arg = "eth"
    buggy_dataset = raw_dataset_arg[:-1]
    fixed_dataset = raw_dataset_arg.rstrip("/\\")
    print(f"  main.py dataset string handling:")
    print(f"    Raw input: '{raw_dataset_arg}'")
    print(f"    Buggy slice (args.dataset[:-1]): '{buggy_dataset}' -> Looks for '{buggy_dataset}_train.pkl' (FAILS)")
    print(f"    Corrected handling: '{fixed_dataset}' -> Looks for '{fixed_dataset}_train.pkl'")
    stage_b_ok = True
except Exception as e:
    print(f"[FAILURE] Stage B failed: {e}")
    stage_b_ok = False


print("\n" + "=" * 70)
print("STAGE C: DATASET AVAILABILITY & PREPROCESSING AUDIT")
print("=" * 70)
try:
    raw_eth_test = mid_root / "raw_data" / "eth" / "test" / "biwi_eth.txt"
    processed_dir = mid_root / "processed_data"
    processed_noise_dir = mid_root / "processed_data_noise"
    eth_train_pkl = processed_dir / "eth_train.pkl"
    eth_test_pkl = processed_dir / "eth_test.pkl"

    print(f"  Raw ETH test file exists: {raw_eth_test.exists()} ({raw_eth_test})")
    print(f"  'processed_data' dir exists: {processed_dir.exists()}")
    print(f"  'processed_data_noise' dir exists: {processed_noise_dir.exists()}")
    print(f"  'eth_train.pkl' exists: {eth_train_pkl.exists()}")
    print(f"  'eth_test.pkl' exists: {eth_test_pkl.exists()}")
    if not eth_train_pkl.exists():
        print(f"  [AUDIT] Processed pickles missing. process_data.py must be run to generate them.")
        print(f"          Note: process_data.py writes to 'processed_data_noise', but baseline.yaml expects 'processed_data'.")
    stage_c_ok = raw_eth_test.exists()
except Exception as e:
    print(f"[FAILURE] Stage C failed: {e}")
    stage_c_ok = False


print("\n" + "=" * 70)
print("STAGE D: MODEL ARCHITECTURE CONSTRUCTION & FORWARD PASS (CPU)")
print("=" * 70)
try:
    device = torch.device("cpu")
    context_dim = 256
    tf_layer = 3
    num_future_points = 12

    # Instantiate denoiser
    denoiser = TransformerConcatLinear(point_dim=2, context_dim=context_dim, tf_layer=tf_layer, residual=False).to(device)
    var_sched = VarianceSchedule(num_steps=100, beta_T=5e-2, mode='linear').to(device)
    diff_model = DiffusionTraj(net=denoiser, var_sched=var_sched).to(device)

    # Test synthetic forward pass on CPU (avoiding hardcoded .cuda() in original DiffusionTraj.get_loss)
    batch_size = 4
    x_0 = torch.randn(batch_size, num_future_points, 2, device=device)
    context = torch.randn(batch_size, context_dim, device=device)
    beta = var_sched.betas[torch.randint(0, 100, (batch_size,))]

    # Direct denoiser forward pass
    out = denoiser(x_0, beta=beta, context=context)
    print(f"[SUCCESS] Denoiser forward pass succeeded on CPU!")
    print(f"  Input future trajectory shape: {x_0.shape}")
    print(f"  Context embedding shape:       {context.shape}")
    print(f"  Predicted noise shape:         {out.shape}")
    print(f"  Finite outputs:                {torch.isfinite(out).all().item()}")

    # Test DDIM sample step logic on CPU
    x_T = torch.randn(batch_size, num_future_points, 2, device=device)
    e_theta = denoiser(x_T, beta=var_sched.betas[[100]*batch_size], context=context)
    print(f"[SUCCESS] Reverse step prediction succeeded on CPU: shape = {e_theta.shape}")
    stage_d_ok = True
except Exception as e:
    print(f"[FAILURE] Stage D failed: {e}")
    import traceback
    traceback.print_exc()
    stage_d_ok = False


print("\n" + "=" * 70)
print("STAGE E: CHECKPOINT AVAILABILITY & OFFICIAL INFERENCE ATTEMPT")
print("=" * 70)
checkpoint_path = mid_root / "experiments" / "baseline" / "eth_epoch70.pt"
print(f"Expected official checkpoint path: {checkpoint_path}")
print(f"Checkpoint exists on disk: {checkpoint_path.exists()}")

if not checkpoint_path.exists():
    print(f"\n[BLOCKER CONFIRMED] Pretrained checkpoint 'eth_epoch70.pt' is ABSENT.")
    print("Evidence: Neither the git tree, GitHub releases, nor external downloads distribute pretrained weights.")
    print("The authors require running 90 epochs of training from scratch.")
    print("Per safety rules: We will NOT fabricate fake weights or claim random weights represent inference.")
    stage_e_status = "BLOCKED: Missing pretrained checkpoint"
else:
    stage_e_status = "Checkpoint present"

print(f"Stage E Status: {stage_e_status}")
print("=" * 70)

