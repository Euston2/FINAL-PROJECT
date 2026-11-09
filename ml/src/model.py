"""
EDEMS ML Component — Model Training & Anomaly Threshold Module.

This module provides helper functions to:
  1. Load day-partitioned traffic flows from ml/data/
  2. Train scikit-learn's Isolation Forest on strictly benign baseline traffic
  3. Calibrate the operational anomaly detection threshold against empirical
     held-out benign validation traffic to enforce a 1% false alarm budget.
"""

from typing import List, Tuple, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from src.config import (
    DATA_DIR, MODELS_DIR, SELECTED_FEATURES, LABEL_COL, BENIGN_LABEL,
    RANDOM_SEED
)


def load_day_flows(
    day_tags: List[str],
    benign_only: bool = True,
    sample_per_day: Optional[int] = None,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load selected feature columns and labels for specified day-tags from ml/data/.

    Parameters
    ----------
    day_tags : list of str
        Tags representing the days to load (e.g. ['15-02-2018', '22-02-2018']).
    benign_only : bool
        If True, filters exclusively for Benign traffic (used for training/validation).
    sample_per_day : int, optional
        If specified, uniformly sub-samples at most N rows per day for memory efficiency.
    random_state : int
        Seed for deterministic sampling.

    Returns
    -------
    X : np.ndarray, shape (N, 24)
        Float32 array of the 24 selected reproducible features.
    labels : np.ndarray or None
        Array of ground-truth label strings (or None if benign_only).
    """
    frames = []
    cols_to_load = SELECTED_FEATURES + [LABEL_COL]

    for day_tag in day_tags:
        file_path = DATA_DIR / f"{day_tag}_cleaned.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Cleaned data file not found: {file_path}")

        df = pd.read_parquet(file_path, columns=cols_to_load)

        if benign_only:
            df = df[df[LABEL_COL] == BENIGN_LABEL]

        if sample_per_day is not None and len(df) > sample_per_day:
            df = df.sample(n=sample_per_day, random_state=random_state)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    X = combined[SELECTED_FEATURES].to_numpy(dtype=np.float32)

    labels = combined[LABEL_COL].to_numpy() if not benign_only else None
    return X, labels


def calibrate_anomaly_threshold(
    model: IsolationForest,
    X_val_benign: np.ndarray,
    target_fpr: float = 0.01
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calibrate the decision threshold on held-out benign validation traffic.

    In scikit-learn IsolationForest:
      - score_samples(X) computes negative average path length.
      - Lower scores represent more isolated / abnormal samples.
      - Normal samples have higher (less negative) scores.

    To enforce an operational False Positive Rate (FPR) of target_fpr (e.g. 1%),
    we set the threshold tau at the (target_fpr * 100)-th percentile of benign scores.
    Any flow with score < tau is flagged as an anomaly.

    Returns
    -------
    threshold : float
        The calculated score threshold.
    empirical_fpr : float
        Actual false positive rate achieved on the validation set.
    score_stats : dict
        Mean, std, min, and max of the benign validation score distribution.
    """
    # Compute continuous anomaly scores for benign validation flows
    val_scores = model.score_samples(X_val_benign)

    # Threshold is the alpha-percentile (e.g. 1st percentile for 1% FPR)
    percentile = target_fpr * 100.0
    threshold = float(np.percentile(val_scores, percentile))

    # Calculate empirical false alarm rate: proportion of benign flows scored below threshold
    false_positives = np.sum(val_scores < threshold)
    empirical_fpr = float(false_positives / len(val_scores))

    score_stats = {
        "mean": float(np.mean(val_scores)),
        "std": float(np.std(val_scores)),
        "min": float(np.min(val_scores)),
        "max": float(np.max(val_scores)),
        "1st_percentile": threshold,
        "5th_percentile": float(np.percentile(val_scores, 5.0)),
        "10th_percentile": float(np.percentile(val_scores, 10.0)),
    }

    return threshold, empirical_fpr, score_stats
