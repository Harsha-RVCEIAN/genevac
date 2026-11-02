# GenEvac-Sentinel: Environment Setup Guide

This guide outlines the reproducible steps to configure the development environment for GenEvac-Sentinel.

---

## 1. Prerequisites
- **Python:** Python 3.11.x (verified with Python 3.11.15 on macOS aarch64).
- **Operating System:** macOS / Linux / Windows.
- **Git:** Git installed.

---

## 2. Environment Setup

### Option A: Standard `venv` (Recommended)
Navigate to the repository directory (`genevac`):

```bash
cd genevac

# 1. Create a Python 3.11 virtual environment
python3.11 -m venv .venv

# 2. Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install runtime and development dependencies
pip install -r requirements-dev.txt

# Or install in editable mode:
pip install -e ".[dev]"
```

### Option B: Using `uv` (Fast alternative)
If `uv` is installed:

```bash
cd genevac
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

---

## 3. Verification Commands

To verify that the installation succeeded:

1. **Verify Python and Pygame:**
   ```bash
   python -c "import pygame; print('Pygame version:', pygame.__version__)"
   ```
   *Expected output: `Pygame version: 2.6.1` (or `>= 2.6.0`)*

2. **Run Pytest Smoke Test:**
   ```bash
   pytest -v
   ```
   *Expected output: `1 passed`*

---

## 4. Operational & Git Guidelines
- **No Direct Push:** Automated tools must never execute `git push`. All repository pushes are handled manually by the project owner following code review.
- **Strict Phase Scoping:** Development proceeds strictly one phase at a time (Week 1, Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5).
