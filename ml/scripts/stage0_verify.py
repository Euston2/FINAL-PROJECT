"""
Stage 0 — Verify project skeleton and environment.

Run this script after creating the venv and installing requirements.txt.
It checks that:
  1. All required packages import successfully.
  2. All project directories exist.
  3. All expected Parquet dataset files are found.
  4. The random seed is set and deterministic.

If everything passes, you're ready for Stage 1.
"""

import sys
import os
import subprocess

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from  ml/scripts/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import config FIRST — it sets MPLCONFIGDIR before matplotlib is loaded,
# which avoids the "Access is denied" error on some Windows setups.
from src.config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, SCRIPTS_DIR, SRC_DIR,
    RAW_DATA_DIR, RANDOM_SEED, DAY_FILES,
)


def check_package(pkg_name: str) -> tuple[bool, str]:
    """
    Check if a package can be imported by running a tiny subprocess.
    This avoids DLL conflicts between packages like onnxruntime and
    scapy that both load native libraries.
    """
    code = (
        f"import {pkg_name}; "
        f"print(getattr({pkg_name}, '__version__', 'unknown'))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()[:200]
    except subprocess.TimeoutExpired:
        return False, "import timed out after 60s"


def main():
    print("=" * 60)
    print("EDEMS ML  -  Stage 0: Environment Verification")
    print("=" * 60)

    all_ok = True

    # ------------------------------------------------------------------
    # Check 1: Required Python packages
    # ------------------------------------------------------------------
    required_packages = [
        "pandas", "numpy", "pyarrow", "sklearn",
        "skl2onnx", "onnxruntime", "scapy", "matplotlib", "tqdm",
    ]
    print("\n[1/4] Checking Python packages...\n")
    for pkg in required_packages:
        ok, info = check_package(pkg)
        if ok:
            print(f"  OK    {pkg:20s} {info}")
        else:
            print(f"  FAIL  {pkg:20s} {info}")
            all_ok = False

    if not all_ok:
        print("\n  Some packages are missing. Run:")
        print("     pip install -r requirements.txt")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Check 2: Project directory structure
    # ------------------------------------------------------------------
    print("\n[2/4] Checking directory structure...\n")
    for label, path in [
        ("data/",    DATA_DIR),
        ("models/",  MODELS_DIR),
        ("results/", RESULTS_DIR),
        ("scripts/", SCRIPTS_DIR),
        ("src/",     SRC_DIR),
        ("raw data", RAW_DATA_DIR),
    ]:
        exists = path.exists()
        mark = "OK  " if exists else "FAIL"
        print(f"  {mark}  {label:12s} {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  Some directories are missing.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Check 3: Dataset files
    # ------------------------------------------------------------------
    print("\n[3/4] Checking dataset files...\n")
    for day_tag, filename in sorted(DAY_FILES.items()):
        path = RAW_DATA_DIR / filename
        exists = path.exists()
        mark = "OK  " if exists else "FAIL"
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0
        print(f"  {mark}  {day_tag}  ({size_mb:6.1f} MB)  {filename}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  Some dataset files are missing.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Check 4: Random seed determinism
    # ------------------------------------------------------------------
    import numpy as np

    print(f"\n[4/4] Random seed = {RANDOM_SEED}")
    np.random.seed(RANDOM_SEED)
    sample = np.random.rand(5)
    print(f"  First 5 random values: {sample}")
    print("  (Should be identical every time you run this script.)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed. You are ready for Stage 1!")
    else:
        print("Some checks FAILED. Fix the issues above and re-run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
