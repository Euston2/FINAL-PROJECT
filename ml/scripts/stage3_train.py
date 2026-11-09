"""
Stage 3 — Train Isolation Forest and Calibrate Operational Detection Threshold.

This script executes the Stage 3 workflow:
  1. Loads strictly BENIGN flows from TRAIN_DAYS (15-02-2018 and 22-02-2018).
  2. Fits an IsolationForest model on normal baseline traffic.
  3. Loads held-out BENIGN flows from VAL_DAYS (23-02-2018).
  4. Calibrates the anomaly detection threshold to enforce a 1% False Positive Rate.
  5. Serializes the trained model and metadata into ml/models/.

To run:
    python scripts/stage3_train.py
"""

import sys
import os
import time
import json
import joblib
from sklearn.ensemble import IsolationForest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    TRAIN_DAYS, VAL_DAYS, SELECTED_FEATURES, IFOREST_PARAMS,
    TARGET_FPR, MODELS_DIR, RANDOM_SEED
)
from src.model import load_day_flows, calibrate_anomaly_threshold


def main():
    print("=" * 80)
    print("EDEMS ML — Stage 3: Day-Based Training & Threshold Calibration")
    print("=" * 80)
    print(f"Features:          {len(SELECTED_FEATURES)} approved reproducible columns")
    print(f"Training Days:     {TRAIN_DAYS} (Strictly Benign flows)")
    print(f"Validation Day:    {VAL_DAYS} (Held-out Benign for threshold)")
    print(f"Target False Alarm:{TARGET_FPR * 100:.1f}%\n")

    # -------------------------------------------------------------------------
    # 1. Load Training Data (Strictly Benign baseline)
    # -------------------------------------------------------------------------
    print("[1/4] Loading training flows from baseline days...")
    t0 = time.time()
    # Sample 250,000 flows per day for a massive, representative 500,000 flow baseline
    X_train, _ = load_day_flows(TRAIN_DAYS, benign_only=True, sample_per_day=250000, random_state=RANDOM_SEED)
    print(f"  Loaded {len(X_train):,d} benign training flows in {time.time()-t0:.2f}s "
          f"(Memory: {X_train.nbytes / (1024**2):.1f} MB)")

    # -------------------------------------------------------------------------
    # 2. Train Isolation Forest
    # -------------------------------------------------------------------------
    print("\n[2/4] Fitting Isolation Forest model...")
    print(f"  Hyperparameters: n_estimators={IFOREST_PARAMS['n_estimators']}, "
          f"max_samples={IFOREST_PARAMS['max_samples']}, random_state={RANDOM_SEED}")
    t0 = time.time()
    model = IsolationForest(**IFOREST_PARAMS)
    model.fit(X_train)
    fit_time = time.time() - t0
    print(f"  Model successfully trained in {fit_time:.2f}s")

    # -------------------------------------------------------------------------
    # 3. Load Held-Out Benign Validation Data & Calibrate Threshold
    # -------------------------------------------------------------------------
    print("\n[3/4] Calibrating decision threshold on held-out benign validation flows...")
    t0 = time.time()
    X_val, _ = load_day_flows(VAL_DAYS, benign_only=True, sample_per_day=100000, random_state=RANDOM_SEED)
    print(f"  Loaded {len(X_val):,d} unseen benign validation flows from {VAL_DAYS} in {time.time()-t0:.2f}s")

    threshold, empirical_fpr, stats = calibrate_anomaly_threshold(
        model, X_val, target_fpr=TARGET_FPR
    )

    print(f"\n  Benign Validation Score Distribution:")
    print(f"    - Mean Score:  {stats['mean']:+.4f} (higher = more normal)")
    print(f"    - Std Dev:     {stats['std']:.4f}")
    print(f"    - Min Score:   {stats['min']:+.4f}")
    print(f"    - Max Score:   {stats['max']:+.4f}")
    print(f"\n  Calibrated Threshold (tau): {threshold:+.6f}")
    print(f"  Target FPR:                 {TARGET_FPR * 100:.2f}%")
    print(f"  Empirical Validation FPR:   {empirical_fpr * 100:.2f}%")
    print(f"  Decision Rule:              Score < {threshold:.6f} ==> ANOMALY / ALERT")

    # -------------------------------------------------------------------------
    # 4. Save Model Artifacts
    # -------------------------------------------------------------------------
    print("\n[4/4] Saving model artifacts to ml/models/...")
    model_path = MODELS_DIR / "edems_iforest.joblib"
    meta_path = MODELS_DIR / "threshold_meta.json"

    joblib.dump(model, model_path)
    print(f"  Saved trained model:  {model_path}")

    metadata = {
        "model_type": "IsolationForest",
        "selected_features": SELECTED_FEATURES,
        "n_features": len(SELECTED_FEATURES),
        "train_days": TRAIN_DAYS,
        "val_days": VAL_DAYS,
        "target_fpr": TARGET_FPR,
        "empirical_val_fpr": empirical_fpr,
        "threshold": threshold,
        "score_stats": stats,
        "iforest_params": {
            "n_estimators": IFOREST_PARAMS["n_estimators"],
            "max_samples": IFOREST_PARAMS["max_samples"],
            "random_state": RANDOM_SEED,
        },
        "score_convention": "scikit-learn score_samples (lower is more anomalous; anomaly if score < threshold)"
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved model metadata: {meta_path}")

    print("\n" + "=" * 80)
    print("STAGE 3 TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
