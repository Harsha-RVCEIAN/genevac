
"""
Staged Verification Script for the Original MID Reference Implementation.

Stages:
A. Dependency and module imports
B. Configuration and actual source-code inspection
C. Raw and processed dataset availability
D. CPU model construction and denoiser forward pass
E. Local checkpoint availability

Important:
- Stage D does NOT verify complete DDIM reverse sampling.
- Stage E checks local checkpoint availability only.
- Missing data/checkpoints are reported as BLOCKED, not script failures.
- This script does not modify the original MID repository or GenEvac code.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

# ============================================================
# PATHS AND ENVIRONMENT DELEGATION
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
RESEARCH_ROOT = SCRIPT_PATH.parents[2]

MID_ROOT = RESEARCH_ROOT / "MID"

ISOLATED_PYTHON = (
    RESEARCH_ROOT
    / "mid_reference"
    / ".venv"
    / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
)

if not MID_ROOT.is_dir():
    print(f"[FATAL] MID repository not found: {MID_ROOT}")
    sys.exit(1)

if (
    ISOLATED_PYTHON.is_file()
    and Path(sys.executable).resolve() != ISOLATED_PYTHON.resolve()
):
    print(
        "[ENV DELEGATION] Forwarding execution to isolated reference venv: "
        f"{ISOLATED_PYTHON}\n"
    )
    result = subprocess.run(
        [str(ISOLATED_PYTHON), str(SCRIPT_PATH), *sys.argv[1:]],
        check=False,
    )
    sys.exit(result.returncode)

sys.path.insert(0, str(MID_ROOT))


# ============================================================
# RESULT MANAGEMENT
# ============================================================

results: dict[str, str] = {}


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def record(name: str, status: str) -> None:
    results[name] = status
    print(f"\n[STAGE RESULT] {name}: {status}")


def read_source(path: Path) -> str | None:
    if not path.is_file():
        return None

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return None


# ============================================================
# STAGE A: DEPENDENCIES AND MODULE IMPORTS
# ============================================================

section("STAGE A: DEPENDENCY AND MODULE IMPORT VERIFICATION")

stage_a_ok = False
torch_mod: Any = None
yaml_mod: Any = None
mid_modules: dict[str, Any] = {}

try:
    import dill
    import easydict
    import numpy as np
    import pandas as pd
    import tensorboardX
    import torch
    import yaml

    torch_mod = torch
    yaml_mod = yaml

    print("[SUCCESS] Core dependencies imported:")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  CUDA:         {torch.cuda.is_available()}")
    print(f"  Dill:         {getattr(dill, '__version__', 'available')}")
    print(f"  Pandas:       {pd.__version__}")
    print(f"  NumPy:        {np.__version__}")
    print(f"  easydict:     {easydict.__name__}")
    print(
        "  tensorboardX: "
        f"{getattr(tensorboardX, '__version__', 'available')}"
    )
    print(f"  PyYAML:       {yaml.__version__}")

    from models.common import ConcatSquashLinear
    from models.diffusion import (
        DiffusionTraj,
        TransformerConcatLinear,
        VarianceSchedule,
    )
    from utils.model_registrar import ModelRegistrar

    mid_modules = {
        "DiffusionTraj": DiffusionTraj,
        "VarianceSchedule": VarianceSchedule,
        "TransformerConcatLinear": TransformerConcatLinear,
        "ConcatSquashLinear": ConcatSquashLinear,
        "ModelRegistrar": ModelRegistrar,
    }

    print("\n[SUCCESS] MID model classes imported:")
    for name in mid_modules:
        print(f"  - {name}")

    stage_a_ok = True
    record("A - Dependencies and Imports", "PASSED")

except Exception as exc:  # noqa: BLE001
    print(f"[FAILURE] Stage A: {type(exc).__name__}: {exc}")
    record("A - Dependencies and Imports", "FAILED")


# ============================================================
# STAGE B: CONFIGURATION AND ACTUAL SOURCE INSPECTION
# ============================================================

section("STAGE B: CONFIGURATION AND SOURCE-CODE AUDIT")

stage_b_status = "FAILED"

try:
    if yaml_mod is None:
        import yaml as yaml_mod

    config_path = MID_ROOT / "configs" / "baseline.yaml"
    main_path = MID_ROOT / "main.py"
    process_data_path = MID_ROOT / "process_data.py"

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration missing: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml_mod.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("baseline.yaml must contain a YAML mapping.")

    print("[SUCCESS] baseline.yaml loaded:")
    for key in ("diffnet", "encoder_dim", "epochs", "eval_at", "data_dir"):
        print(f"  {key}: {config.get(key)}")

    main_source = read_source(main_path)
    process_source = read_source(process_data_path)

    if main_source is None:
        raise FileNotFoundError(f"main.py missing: {main_path}")

    # Parse the actual source file.
    # Detect expressions such as args.dataset[:-1].
    tree = ast.parse(main_source)
    slicing_defects: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue

        value = node.value
        slice_node = node.slice

        is_dataset_attribute = (
            isinstance(value, ast.Attribute)
            and value.attr == "dataset"
        )

        is_negative_one_slice = (
            isinstance(slice_node, ast.Slice)
            and slice_node.lower is None
            and slice_node.upper is not None
            and isinstance(slice_node.upper, ast.UnaryOp)
            and isinstance(slice_node.upper.op, ast.USub)
            and isinstance(slice_node.upper.operand, ast.Constant)
            and slice_node.upper.operand.value == 1
            and slice_node.step is None
        )

        if is_dataset_attribute and is_negative_one_slice:
            line = getattr(node, "lineno", 0)
            source_line = main_source.splitlines()[line - 1].strip()
            slicing_defects.append((line, source_line))

    if slicing_defects:
        print("\n[ISSUE FOUND] Dataset slicing expression exists in main.py:")
        for line, source_line in slicing_defects:
            print(f"  Line {line}: {source_line}")
        print("  For dataset='eth', slicing with [:-1] produces 'et'.")
        stage_b_status = "PARTIAL - Dataset argument defect detected"
    else:
        print(
            "\n[CHECK PASSED] No args.dataset[:-1]-style expression "
            "was found in main.py."
        )
        stage_b_status = "PASSED - Targeted source check"

    # Report data-directory references for manual comparison.
    print("\nDataset directory configuration:")
    configured_dir = config.get("data_dir")
    print(f"  baseline.yaml data_dir: {configured_dir}")

    if process_source is not None:
        print("\nPreprocessing directory references:")
        found_references = sorted(
            {
                term
                for term in ("processed_data", "processed_data_noise")
                if term in process_source
            }
        )

        if found_references:
            for term in found_references:
                print(f"  Found reference: {term}")
        else:
            print("  No known processed-data directory strings found.")
    else:
        print("  process_data.py is missing.")

    print(
        "\n[NOTE] This is a targeted static inspection. "
        "It does not execute the MID training entry point."
    )

except Exception as exc:  # noqa: BLE001
    print(f"[FAILURE] Stage B: {type(exc).__name__}: {exc}")
    record("B - Configuration and Source Audit", "FAILED")
else:
    record("B - Configuration and Source Audit", stage_b_status)


# ============================================================
# STAGE C: RAW AND PROCESSED DATA AVAILABILITY
# ============================================================

section("STAGE C: DATASET AVAILABILITY AND PREPROCESSING AUDIT")

stage_c_status = "FAILED"

try:
    if yaml_mod is None:
        raise RuntimeError("PyYAML unavailable; Stage A dependency failure.")

    config_path = MID_ROOT / "configs" / "baseline.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml_mod.safe_load(file)

    raw_root = MID_ROOT / "raw_data" / "eth"

    raw_train_dir = raw_root / "train"
    raw_test_dir = raw_root / "test"

    raw_train_files = (
        list(raw_train_dir.rglob("*.txt"))
        if raw_train_dir.is_dir()
        else []
    )

    raw_test_files = (
        list(raw_test_dir.rglob("*.txt"))
        if raw_test_dir.is_dir()
        else []
    )

    configured_dir_value = config.get("data_dir", "processed_data")
    configured_dir = Path(str(configured_dir_value))

    if not configured_dir.is_absolute():
        configured_dir = MID_ROOT / configured_dir

    alternate_dir = MID_ROOT / "processed_data_noise"

    def find_pickles(directory: Path, pattern: str) -> list[Path]:
        if not directory.is_dir():
            return []
        return list(directory.glob(pattern))

    train_pickles = find_pickles(configured_dir, "*train*.pkl")
    test_pickles = find_pickles(configured_dir, "*test*.pkl")

    alternate_train = find_pickles(alternate_dir, "*train*.pkl")
    alternate_test = find_pickles(alternate_dir, "*test*.pkl")

    print(f"Raw ETH training files: {len(raw_train_files)}")
    for path in raw_train_files:
        print(f"  - {path.relative_to(MID_ROOT)}")

    print(f"\nRaw ETH testing files: {len(raw_test_files)}")
    for path in raw_test_files:
        print(f"  - {path.relative_to(MID_ROOT)}")

    print(f"\nConfigured processed directory: {configured_dir}")
    print(f"  Training pickles: {len(train_pickles)}")
    print(f"  Testing pickles:  {len(test_pickles)}")

    print(f"\nAlternative directory: {alternate_dir}")
    print(f"  Training pickles: {len(alternate_train)}")
    print(f"  Testing pickles:  {len(alternate_test)}")

    raw_ready = bool(raw_train_files and raw_test_files)
    configured_processed_ready = bool(train_pickles and test_pickles)
    alternate_processed_ready = bool(alternate_train and alternate_test)

    print("\nReadiness:")
    print(f"  Raw train/test data ready: {raw_ready}")
    print(
        "  Configured processed train/test ready: "
        f"{configured_processed_ready}"
    )
    print(
        "  Alternative processed train/test ready: "
        f"{alternate_processed_ready}"
    )

    if configured_processed_ready:
        stage_c_status = "PASSED - Configured processed data found"
    elif alternate_processed_ready:
        stage_c_status = (
            "BLOCKED - Processed data exists in alternate directory, "
            "not configured data_dir"
        )
    elif raw_ready:
        stage_c_status = (
            "BLOCKED - Raw data exists, but processed train/test "
            "pickles are missing"
        )
    else:
        stage_c_status = (
            "BLOCKED - Raw train/test data is incomplete and "
            "processed data is unavailable"
        )

except Exception as exc:  # noqa: BLE001
    print(f"[FAILURE] Stage C: {type(exc).__name__}: {exc}")
    record("C - Dataset Availability", "FAILED")
else:
    record("C - Dataset Availability", stage_c_status)


# ============================================================
# STAGE D: CPU MODEL CONSTRUCTION AND FORWARD PASS
# ============================================================

section("STAGE D: CPU MODEL CONSTRUCTION AND DENOISER FORWARD PASS")

stage_d_status = "SKIPPED"

if not stage_a_ok or torch_mod is None:
    print("[SKIPPED] Requires successful Stage A and PyTorch.")
else:
    try:
        torch = torch_mod
        device = torch.device("cpu")

        batch_size = 4
        future_points = 12
        context_dim = 256
        transformer_layers = 3

        denoiser_cls = mid_modules["TransformerConcatLinear"]
        variance_cls = mid_modules["VarianceSchedule"]
        diffusion_cls = mid_modules["DiffusionTraj"]

        denoiser = denoiser_cls(
            point_dim=2,
            context_dim=context_dim,
            tf_layer=transformer_layers,
            residual=False,
        ).to(device)

        variance_schedule = variance_cls(
            num_steps=100,
            beta_T=5e-2,
            mode="linear",
        ).to(device)

        diffusion_model = diffusion_cls(
            net=denoiser,
            var_sched=variance_schedule,
        ).to(device)

        assert diffusion_model.net is denoiser
        assert diffusion_model.var_sched is variance_schedule

        x_0 = torch.randn(
            batch_size,
            future_points,
            2,
            device=device,
        )

        context = torch.randn(
            batch_size,
            context_dim,
            device=device,
        )

        beta_count = variance_schedule.betas.shape[0]

        if beta_count < 2:
            raise RuntimeError(
                f"Unexpected variance schedule length: {beta_count}"
            )

        timestep_indices = torch.randint(
            low=1,
            high=beta_count,
            size=(batch_size,),
        )

        beta = variance_schedule.betas[timestep_indices]

        denoiser.eval()

        with torch.no_grad():
            predicted_noise = denoiser(
                x_0,
                beta=beta,
                context=context,
            )

        if predicted_noise.shape != x_0.shape:
            raise AssertionError(
                f"Unexpected output shape: {predicted_noise.shape}"
            )

        if not torch.isfinite(predicted_noise).all().item():
            raise AssertionError("Denoiser output contains NaN or Inf.")

        print("[SUCCESS] Model construction completed.")
        print(f"  Device:                 {device}")
        print(f"  Input trajectory:       {tuple(x_0.shape)}")
        print(f"  Context:                {tuple(context.shape)}")
        print(f"  Variance schedule size: {beta_count}")
        print(f"  Predicted noise:        {tuple(predicted_noise.shape)}")
        print("  Output finite:          True")

        # A second denoiser evaluation at the final schedule beta.
        # This is NOT a DDIM reverse-sampling loop.
        x_t = torch.randn(
            batch_size,
            future_points,
            2,
            device=device,
        )

        final_beta = variance_schedule.betas[-1].expand(batch_size)

        with torch.no_grad():
            final_noise = denoiser(
                x_t,
                beta=final_beta,
                context=context,
            )

        if not torch.isfinite(final_noise).all().item():
            raise AssertionError("Final denoiser output is non-finite.")

        print(
            "[SUCCESS] Final-index denoiser evaluation: "
            f"{tuple(final_noise.shape)}"
        )

        print(
            "\n[BOUNDARY] Complete DDIM sampling, checkpoint loading, "
            "and real trajectory inference were NOT tested."
        )

        stage_d_status = (
            "PASSED - Synthetic CPU denoiser forward pass only"
        )

    except Exception as exc:  # noqa: BLE001
        print(f"[FAILURE] Stage D: {type(exc).__name__}: {exc}")
        record("D - CPU Denoiser Forward Pass", "FAILED")
    else:
        record("D - CPU Denoiser Forward Pass", stage_d_status)


# ============================================================
# STAGE E: LOCAL CHECKPOINT AVAILABILITY
# ============================================================

section("STAGE E: LOCAL CHECKPOINT AVAILABILITY")

checkpoint_path = (
    MID_ROOT
    / "experiments"
    / "baseline"
    / "eth_epoch70.pt"
)

print(f"Expected checkpoint: {checkpoint_path}")
print(f"Exists locally: {checkpoint_path.is_file()}")

if checkpoint_path.is_file():
    stage_e_status = (
        "CHECKPOINT FOUND - Inference not attempted by this diagnostic"
    )
    print("[FOUND] Expected local checkpoint exists.")
else:
    stage_e_status = "BLOCKED - Expected local checkpoint missing"
    print("[BLOCKED] Expected local checkpoint is absent.")

print(
    "[LIMITATION] This local check does not prove that weights "
    "are unavailable from GitHub releases or external sources."
)

record("E - Local Checkpoint Availability", stage_e_status)


# ============================================================
# FINAL SUMMARY
# ============================================================

section("EXECUTION SUMMARY")

for name, status in results.items():
    print(f"{name:43s}: {status}")

print("=" * 70)

print("\nSTATUS DEFINITIONS:")
print("  PASSED   - The specific check completed successfully.")
print("  PARTIAL  - A defect or incomplete condition was identified.")
print("  BLOCKED  - A required external input or artifact is missing.")
print("  FAILED   - The diagnostic itself encountered an error.")
print("  SKIPPED  - A prerequisite for the check was unavailable.")

print("\nINTERPRETATION:")
print("- Stage D is not proof of full MID inference.")
print("- Stage C requires processed train AND test data.")
print("- Stage E checks only the expected local checkpoint.")
print("- GenEvac implementation is independent of reference recovery.")

diagnostic_failures = [
    name
    for name, status in results.items()
    if status == "FAILED" or status == "SKIPPED"
]

if diagnostic_failures:
    print("\n[OVERALL] Diagnostic execution had failed or skipped stages.")
    sys.exit(1)

print("\n[OVERALL] Diagnostic script completed without execution errors.")
sys.exit(0)
