"""
Stage C — Realistic Evaluation at Operational Attack Prevalence.

Problem: The Stage 4 test set contains 78.5% attacks, which is absurdly
unrealistic. In a real enterprise network, attack traffic is a tiny
fraction of total volume (typically 0.1%–5%). The inflated attack
prevalence causes:
  - Artificially high PR-AUC (baseline = prevalence = 78.5%).
  - Misleading precision (even random flagging hits attacks frequently).
  - Confidence intervals that are far too narrow.

This script:
  1. Re-evaluates the existing IsolationForest at 1% and 5% attack
     prevalence by subsampling attacks relative to a fixed benign pool.
  2. Repeats each prevalence setting across 10 random seeds to compute
     mean and standard deviation of all metrics.
  3. Computes per-class ROC-AUC (which is prevalence-independent) for
     every attack class against a fixed benign test pool.
  4. Reports precision, PR-AUC, PR-AUC baseline, and recall at 1% and
     5% FPR for each prevalence setting.

Outputs saved to:
  ml/results/diagnostics/stage_c_realistic_eval.csv
  ml/results/diagnostics/stage_c_perclass_auc.csv
  ml/results/diagnostics/stage_c_summary.json

To run:
    python scripts/stageC_realistic_eval.py
"""

import sys
import os
import json
import time
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR,
    SELECTED_FEATURES, LABEL_COL, BENIGN_LABEL, RANDOM_SEED
)


def evaluate_at_prevalence(
    model, X_benign, X_attacks, labels_attacks, threshold,
    target_prevalence, n_seeds=10, base_seed=42
):
    """
    Subsample attacks to achieve a target prevalence, repeat across seeds.

    Parameters
    ----------
    model : fitted IsolationForest
    X_benign : np.ndarray — benign test flows (fixed across seeds)
    X_attacks : np.ndarray — all attack flows available for subsampling
    labels_attacks : np.ndarray — string labels for each attack flow
    threshold : float — calibrated anomaly threshold (from Stage 3)
    target_prevalence : float — e.g. 0.01 for 1% attacks
    n_seeds : int — number of random repetitions
    base_seed : int

    Returns
    -------
    seed_results : list of dicts — per-seed metric snapshots
    """
    n_benign = len(X_benign)
    # How many attack flows needed to achieve target prevalence:
    # prevalence = n_attack / (n_benign + n_attack)
    # => n_attack = prevalence * n_benign / (1 - prevalence)
    n_attack_target = int(target_prevalence * n_benign / (1.0 - target_prevalence))

    if n_attack_target > len(X_attacks):
        n_attack_target = len(X_attacks)

    seed_results = []
    for seed_offset in range(n_seeds):
        rng = np.random.RandomState(base_seed + seed_offset)
        idx = rng.choice(len(X_attacks), size=n_attack_target, replace=False)

        X_atk_sample = X_attacks[idx]
        X_eval = np.vstack([X_benign, X_atk_sample])
        y_true = np.concatenate([np.zeros(n_benign), np.ones(n_attack_target)])

        # Anomaly scores: higher = more anomalous (negate score_samples)
        anomaly_scores = -model.score_samples(X_eval)

        # Threshold-based predictions (using calibrated threshold from Stage 3)
        # Original convention: score_samples < threshold => anomaly
        # Equivalent: -score_samples > -threshold => anomaly
        neg_threshold = -threshold
        y_pred = (anomaly_scores >= neg_threshold).astype(int)

        # Metrics
        actual_prevalence = float(y_true.mean())
        baseline_pr_auc = actual_prevalence  # random classifier PR-AUC = prevalence

        try:
            roc_auc = float(roc_auc_score(y_true, anomaly_scores))
        except ValueError:
            roc_auc = np.nan

        try:
            pr_auc = float(average_precision_score(y_true, anomaly_scores))
        except ValueError:
            pr_auc = np.nan

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # FPR
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Recall at specific FPR thresholds (1% and 5%)
        benign_scores = anomaly_scores[:n_benign]
        attack_scores = anomaly_scores[n_benign:]

        thresh_1pct_fpr = float(np.percentile(benign_scores, 99.0))
        thresh_5pct_fpr = float(np.percentile(benign_scores, 95.0))

        recall_at_1pct_fpr = float(np.mean(attack_scores >= thresh_1pct_fpr) * 100)
        recall_at_5pct_fpr = float(np.mean(attack_scores >= thresh_5pct_fpr) * 100)

        seed_results.append({
            "seed": base_seed + seed_offset,
            "n_benign": n_benign,
            "n_attack": n_attack_target,
            "actual_prevalence": round(actual_prevalence, 4),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "baseline_pr_auc": round(baseline_pr_auc, 4),
            "pr_auc_above_baseline": round(pr_auc - baseline_pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "recall_at_1pct_fpr": recall_at_1pct_fpr,
            "recall_at_5pct_fpr": recall_at_5pct_fpr,
        })

    return seed_results


def run_stage_c():
    print("=" * 90)
    print("EDEMS ML — Stage C: Realistic Evaluation at Operational Attack Prevalence")
    print("=" * 90)

    diag_dir = RESULTS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Load Model & Data
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading trained IsolationForest and test data...")
    model = joblib.load(MODELS_DIR / "edems_iforest.joblib")
    with open(MODELS_DIR / "threshold_meta.json") as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    print(f"  Calibrated threshold (tau): {threshold:+.6f}")

    # Load all cleaned data
    cols = SELECTED_FEATURES + [LABEL_COL]
    all_frames = []
    for p in sorted(glob.glob(str(DATA_DIR / "*_cleaned.parquet"))):
        all_frames.append(pd.read_parquet(p, columns=cols))
    df_all = pd.concat(all_frames, ignore_index=True)

    df_benign = df_all[df_all[LABEL_COL] == BENIGN_LABEL]
    df_attacks = df_all[df_all[LABEL_COL] != BENIGN_LABEL]

    # Fixed benign test pool: 100,000 representative flows
    if len(df_benign) > 100000:
        df_benign = df_benign.sample(n=100000, random_state=RANDOM_SEED)

    X_benign = df_benign[SELECTED_FEATURES].to_numpy(dtype=np.float32)
    X_attacks = df_attacks[SELECTED_FEATURES].to_numpy(dtype=np.float32)
    labels_attacks = df_attacks[LABEL_COL].astype(str).to_numpy()

    print(f"  Benign test pool: {len(X_benign):,d} flows")
    print(f"  Attack pool: {len(X_attacks):,d} flows ({len(np.unique(labels_attacks))} classes)")

    # -------------------------------------------------------------------------
    # 2. Evaluate at 1% and 5% Attack Prevalence (10 seeds each)
    # -------------------------------------------------------------------------
    prevalence_settings = [0.01, 0.05]
    all_results = []

    for prev in prevalence_settings:
        print(f"\n[2/4] Evaluating at {prev*100:.0f}% attack prevalence (10 seeds)...")
        seed_results = evaluate_at_prevalence(
            model, X_benign, X_attacks, labels_attacks,
            threshold, target_prevalence=prev, n_seeds=10, base_seed=RANDOM_SEED
        )
        for r in seed_results:
            r["target_prevalence"] = prev
        all_results.extend(seed_results)

        # Compute mean and std across seeds
        df_seeds = pd.DataFrame(seed_results)
        print(f"\n  Results at {prev*100:.0f}% prevalence (mean +/- std across 10 seeds):")
        for metric in ["roc_auc", "pr_auc", "baseline_pr_auc", "pr_auc_above_baseline",
                       "precision", "recall", "f1", "fpr",
                       "recall_at_1pct_fpr", "recall_at_5pct_fpr"]:
            vals = df_seeds[metric].dropna()
            if len(vals) > 0:
                m = vals.mean()
                s = vals.std()
                if metric in ["recall_at_1pct_fpr", "recall_at_5pct_fpr"]:
                    print(f"    {metric:25s}: {m:7.2f}% +/- {s:5.2f}%")
                else:
                    print(f"    {metric:25s}: {m:7.4f} +/- {s:5.4f}")

    results_df = pd.DataFrame(all_results)
    results_csv = diag_dir / "stage_c_realistic_eval.csv"
    results_df.to_csv(results_csv, index=False)

    # -------------------------------------------------------------------------
    # 3. Per-Class ROC-AUC (Prevalence-Independent)
    # -------------------------------------------------------------------------
    print("\n[3/4] Computing per-class ROC-AUC (prevalence-independent)...")

    # Pre-score benign pool
    benign_anomaly_scores = -model.score_samples(X_benign)
    benign_thresh_1pct = float(np.percentile(benign_anomaly_scores, 99.0))
    benign_thresh_5pct = float(np.percentile(benign_anomaly_scores, 95.0))

    perclass_records = []
    unique_attacks = sorted(np.unique(labels_attacks))

    for atk_lbl in unique_attacks:
        mask = (labels_attacks == atk_lbl)
        X_atk = X_attacks[mask]
        n_atk = len(X_atk)
        unreliable = n_atk < 100

        atk_anomaly_scores = -model.score_samples(X_atk)

        y_eval = np.concatenate([np.zeros(len(X_benign)), np.ones(n_atk)])
        scores_eval = np.concatenate([benign_anomaly_scores, atk_anomaly_scores])

        try:
            auc = float(roc_auc_score(y_eval, scores_eval))
        except ValueError:
            auc = np.nan

        rec_1pct = float(np.mean(atk_anomaly_scores >= benign_thresh_1pct) * 100)
        rec_5pct = float(np.mean(atk_anomaly_scores >= benign_thresh_5pct) * 100)

        perclass_records.append({
            "attack_class": atk_lbl,
            "sample_count": n_atk,
            "is_unreliable": unreliable,
            "roc_auc": round(auc, 4),
            "recall_at_1pct_fpr": round(rec_1pct, 2),
            "recall_at_5pct_fpr": round(rec_5pct, 2),
        })

    perclass_df = pd.DataFrame(perclass_records).sort_values("roc_auc", ascending=False)
    perclass_csv = diag_dir / "stage_c_perclass_auc.csv"
    perclass_df.to_csv(perclass_csv, index=False)

    print(f"\n{'Attack Class':28s} | {'Count':8s} | {'ROC-AUC':8s} | {'Rec@1%FPR':10s} | {'Rec@5%FPR':10s} | Reliability")
    print("-" * 90)
    for _, r in perclass_df.iterrows():
        rel = "UNRELIABLE (N<100)" if r["is_unreliable"] else "OK"
        cnt = f"{r['sample_count']:,d}"
        if r["is_unreliable"]:
            cnt += "*"
        print(f"{r['attack_class']:28s} | {cnt:8s} | {r['roc_auc']:8.4f} | {r['recall_at_1pct_fpr']:9.2f}% | {r['recall_at_5pct_fpr']:9.2f}% | {rel}")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # 4. Summary JSON
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating summary...")

    summary = {}
    for prev in prevalence_settings:
        df_p = results_df[results_df["target_prevalence"] == prev]
        key = f"prevalence_{int(prev*100)}pct"
        summary[key] = {}
        for metric in ["roc_auc", "pr_auc", "baseline_pr_auc", "pr_auc_above_baseline",
                       "precision", "recall", "f1", "fpr",
                       "recall_at_1pct_fpr", "recall_at_5pct_fpr"]:
            vals = df_p[metric].dropna()
            summary[key][metric] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
            }

    summary["perclass_roc_auc"] = {
        r["attack_class"]: r["roc_auc"]
        for _, r in perclass_df.iterrows()
    }

    summary_path = diag_dir / "stage_c_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nExecution Time: {elapsed:.1f} seconds")
    print(f"Saved: {results_csv.name}, {perclass_csv.name}, {summary_path.name}")
    print("=" * 90)


if __name__ == "__main__":
    run_stage_c()
