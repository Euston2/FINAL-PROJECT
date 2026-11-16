"""
Stage E — Unsupervised Model Comparison on 60-Second Windowed Features.

Objective:
  Compare unsupervised anomaly detection algorithms on the 22 window-level
  aggregate features developed in Stage D to determine the most effective,
  robust, and deployable architecture for the live EDEMS capture system.

Algorithms Compared:
  1. Isolation Forest (default: max_samples='auto', max_features=1.0)
  2. Isolation Forest (max_samples=256 — smaller subsamples limit masking)
  3. Isolation Forest (max_samples=all / 926 — full sample depth)
  4. Isolation Forest (max_features=0.8 — random subspace bagging)
  5. Local Outlier Factor (LOF with novelty=True, k=20 neighbors)
  6. PCA Reconstruction Error (k=8 components, linear subspace modeling)
  7. Autoencoder (MLP 22 -> 14 -> 8 -> 14 -> 22, non-linear bottleneck)

Methodology:
  - Trained strictly on benign windows from TRAIN_DAYS.
  - Threshold calibrated for 1% False Positive Rate on VAL_DAYS.
  - Evaluated on held-out TEST_DAYS at 5% attack prevalence (10 random seeds).
  - Reports: Training time, ROC-AUC, PR-AUC, Precision, Recall, Recall@1%FPR,
    and full per-class detection matrices.

Outputs saved to:
  ml/results/diagnostics/stage_e_models_summary.csv
  ml/results/diagnostics/stage_e_perclass_matrix.csv
  ml/results/diagnostics/stage_e_comparison_plot.png
  ml/results/diagnostics/stage_e_summary.json

To run:
    python scripts/stageE_model_comparison.py
"""

import sys
import os
import json
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    DATA_DIR, RESULTS_DIR, TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    LABEL_COL, BENIGN_LABEL, RANDOM_SEED, TARGET_FPR
)

# Import feature names and extractor from Stage D
from scripts.stageD_windowed_features import (
    WINDOW_FEATURE_NAMES, extract_day_windows, WINDOWS_PER_DAY
)


class ModelWrapper:
    """
    Standardizes fitting and anomaly score computation across heterogeneous
    unsupervised anomaly detection algorithms. Higher score = more anomalous.
    """
    def __init__(self, name, model_type, model_obj, needs_scaling=False):
        self.name = name
        self.model_type = model_type
        self.model = model_obj
        self.needs_scaling = needs_scaling
        self.scaler = StandardScaler() if needs_scaling else None
        self.train_time = 0.0

    def fit(self, X_train):
        t0 = time.time()
        if self.needs_scaling:
            X_tr = self.scaler.fit_transform(X_train)
        else:
            X_tr = X_train

        if self.model_type == "autoencoder":
            # Autoencoder targets its own scaled input
            self.model.fit(X_tr, X_tr)
        elif self.model_type == "pca":
            self.model.fit(X_tr)
        else:
            self.model.fit(X_tr)

        self.train_time = time.time() - t0
        return self

    def score(self, X):
        """Returns anomaly scores: higher = more anomalous."""
        if self.needs_scaling:
            X_eval = self.scaler.transform(X)
        else:
            X_eval = X

        if self.model_type == "iforest":
            # score_samples: more negative = more anomalous -> negate so higher = anomaly
            return -self.model.score_samples(X_eval)
        elif self.model_type == "lof":
            return -self.model.score_samples(X_eval)
        elif self.model_type == "pca":
            # Reconstruction error = Mean Squared Error between X and inverse_transform
            reconstructed = self.model.inverse_transform(self.model.transform(X_eval))
            return np.mean((X_eval - reconstructed) ** 2, axis=1)
        elif self.model_type == "autoencoder":
            reconstructed = self.model.predict(X_eval)
            return np.mean((X_eval - reconstructed) ** 2, axis=1)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


def build_candidate_models(n_train_samples):
    """Instantiates all candidate unsupervised models for comparison."""
    models = [
        ModelWrapper(
            name="IForest (auto, feat=1.0)",
            model_type="iforest",
            model_obj=IsolationForest(
                n_estimators=100, max_samples="auto", max_features=1.0,
                random_state=RANDOM_SEED, n_jobs=-1
            ),
            needs_scaling=False
        ),
        ModelWrapper(
            name="IForest (max_samples=256)",
            model_type="iforest",
            model_obj=IsolationForest(
                n_estimators=100, max_samples=256, max_features=1.0,
                random_state=RANDOM_SEED, n_jobs=-1
            ),
            needs_scaling=False
        ),
        ModelWrapper(
            name="IForest (max_samples=all)",
            model_type="iforest",
            model_obj=IsolationForest(
                n_estimators=100, max_samples=min(2048, n_train_samples), max_features=1.0,
                random_state=RANDOM_SEED, n_jobs=-1
            ),
            needs_scaling=False
        ),
        ModelWrapper(
            name="IForest (max_features=0.8)",
            model_type="iforest",
            model_obj=IsolationForest(
                n_estimators=100, max_samples="auto", max_features=0.8,
                random_state=RANDOM_SEED, n_jobs=-1
            ),
            needs_scaling=False
        ),
        ModelWrapper(
            name="Local Outlier Factor (k=20)",
            model_type="lof",
            model_obj=LocalOutlierFactor(
                n_neighbors=20, novelty=True, n_jobs=-1
            ),
            needs_scaling=True
        ),
        ModelWrapper(
            name="PCA Reconstruction (k=8)",
            model_type="pca",
            model_obj=PCA(
                n_components=8, random_state=RANDOM_SEED
            ),
            needs_scaling=True
        ),
        ModelWrapper(
            name="Autoencoder MLP (14-8-14)",
            model_type="autoencoder",
            model_obj=MLPRegressor(
                hidden_layer_sizes=(14, 8, 14),
                activation="relu",
                max_iter=250,
                random_state=RANDOM_SEED,
                early_stopping=True,
                n_iter_no_change=15
            ),
            needs_scaling=True
        ),
    ]
    return models


def load_or_extract_windows(diag_dir):
    """Loads cached window dataset if available; otherwise extracts and caches."""
    cache_path = diag_dir / "stage_e_windows_cache.parquet"
    if cache_path.exists():
        print("  Loading pre-extracted windows from cache...")
        df_all = pd.read_parquet(cache_path)
    else:
        print("  Extracting 60-second windows across all 10 days...")
        all_dfs = []
        for d in (TRAIN_DAYS + VAL_DAYS + TEST_DAYS):
            w_df = extract_day_windows(d)
            all_dfs.append(w_df)
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_parquet(cache_path, index=False)
        print(f"  Cached {len(df_all)} windows to {cache_path.name}")
    return df_all


def run_stage_e():
    print("=" * 95)
    print("EDEMS ML — Stage E: Unsupervised Model Comparison on 60-Second Windows")
    print("=" * 95)

    diag_dir = RESULTS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[1/4] Preparing windowed datasets (Train / Val / Test)...")
    df_all = load_or_extract_windows(diag_dir)

    # Train pool: purely benign windows from TRAIN_DAYS
    train_mask = df_all["day"].isin(TRAIN_DAYS) & (df_all["is_attack"] == 0)
    df_train = df_all[train_mask]
    X_train = df_train[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)

    # Validation pool: purely benign windows from VAL_DAYS for 1% FPR calibration
    val_mask = df_all["day"].isin(VAL_DAYS) & (df_all["is_attack"] == 0)
    df_val = df_all[val_mask]
    X_val = df_val[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)

    # Test pool: held-out windows from TEST_DAYS (plus VAL benign to stabilize ratio)
    test_mask = df_all["day"].isin(TEST_DAYS)
    df_test = df_all[test_mask]

    df_test_benign = pd.concat([df_test[df_test["is_attack"] == 0], df_val], ignore_index=True)
    df_test_attack = df_test[df_test["is_attack"] == 1]

    X_test_benign = df_test_benign[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)
    X_test_attack = df_test_attack[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)
    labels_test_attack = df_test_attack["primary_attack"].to_numpy()

    print(f"  Training Windows (Benign)  : {len(X_train)}")
    print(f"  Validation Windows (Benign): {len(X_val)}")
    print(f"  Test Windows (Benign)      : {len(X_test_benign)}")
    print(f"  Test Windows (Attack)      : {len(X_test_attack)}")

    # -------------------------------------------------------------------------
    # 2. Train and Evaluate Models
    # -------------------------------------------------------------------------
    print("\n[2/4] Training and evaluating candidate anomaly detection models...")
    models = build_candidate_models(len(X_train))

    summary_records = []
    perclass_matrix = {}

    target_prev = 0.05  # 5% attack prevalence
    n_seeds = 10
    n_benign = len(X_test_benign)
    n_attack_target = int(target_prev * n_benign / (1.0 - target_prev))

    unique_attacks = sorted(np.unique(labels_test_attack))

    for m in models:
        # Fit model
        m.fit(X_train)

        # Calibrate 1% FPR threshold on validation benign windows
        val_scores = m.score(X_val)
        # Threshold: 99th percentile of validation anomaly scores (top 1% flagged)
        tau = float(np.percentile(val_scores, 99.0))

        # Evaluate at 5% prevalence across 10 random seeds
        seed_aucs, seed_praucs, seed_precs, seed_recs, seed_f1s, seed_rec1s = [], [], [], [], [], []
        for s in range(n_seeds):
            rng = np.random.RandomState(RANDOM_SEED + s)
            sub_idx = rng.choice(len(X_test_attack), size=n_attack_target, replace=False)
            X_eval = np.vstack([X_test_benign, X_test_attack[sub_idx]])
            y_eval = np.concatenate([np.zeros(n_benign), np.ones(n_attack_target)])

            scores_eval = m.score(X_eval)
            y_pred = (scores_eval >= tau).astype(int)

            try:
                seed_aucs.append(float(roc_auc_score(y_eval, scores_eval)))
            except ValueError:
                seed_aucs.append(np.nan)

            try:
                seed_praucs.append(float(average_precision_score(y_eval, scores_eval)))
            except ValueError:
                seed_praucs.append(np.nan)

            seed_precs.append(float(precision_score(y_eval, y_pred, zero_division=0)))
            seed_recs.append(float(recall_score(y_eval, y_pred, zero_division=0)))
            seed_f1s.append(float(f1_score(y_eval, y_pred, zero_division=0)))

            # Recall at 1% FPR against test benign
            ben_scores = scores_eval[:n_benign]
            atk_scores = scores_eval[n_benign:]
            thresh_1 = float(np.percentile(ben_scores, 99.0))
            seed_rec1s.append(float(np.mean(atk_scores >= thresh_1) * 100))

        # Per-class ROC-AUC and Recall@1%FPR against full test benign pool
        benign_test_scores = m.score(X_test_benign)
        thresh_1_full = float(np.percentile(benign_test_scores, 99.0))

        perclass_auc_dict = {}
        perclass_rec_dict = {}

        for atk_name in unique_attacks:
            mask = (labels_test_attack == atk_name)
            X_atk_sub = X_test_attack[mask]
            atk_sub_scores = m.score(X_atk_sub)

            y_pc = np.concatenate([np.zeros(len(X_test_benign)), np.ones(len(X_atk_sub))])
            sc_pc = np.concatenate([benign_test_scores, atk_sub_scores])

            try:
                auc_pc = float(roc_auc_score(y_pc, sc_pc))
            except ValueError:
                auc_pc = np.nan

            rec_pc = float(np.mean(atk_sub_scores >= thresh_1_full) * 100)

            perclass_auc_dict[atk_name] = round(auc_pc, 4)
            perclass_rec_dict[atk_name] = round(rec_pc, 1)

        perclass_matrix[m.name] = {
            "auc": perclass_auc_dict,
            "rec_at_1pct": perclass_rec_dict
        }

        summary_records.append({
            "model": m.name,
            "train_time_sec": round(m.train_time, 3),
            "roc_auc_mean": round(float(np.mean(seed_aucs)), 4),
            "roc_auc_std": round(float(np.std(seed_aucs)), 4),
            "pr_auc_mean": round(float(np.mean(seed_praucs)), 4),
            "pr_auc_std": round(float(np.std(seed_praucs)), 4),
            "precision_mean": round(float(np.mean(seed_precs)), 4),
            "recall_mean": round(float(np.mean(seed_recs)), 4),
            "f1_mean": round(float(np.mean(seed_f1s)), 4),
            "recall_at_1pct_mean": round(float(np.mean(seed_rec1s)), 2),
            "recall_at_1pct_std": round(float(np.std(seed_rec1s)), 2),
            "threshold_tau": round(tau, 6),
        })

        print(f"  {m.name:28s} | Fit: {m.train_time:5.3f}s | AUC: {np.mean(seed_aucs):.4f} | PR-AUC: {np.mean(seed_praucs):.4f} | Rec@1%: {np.mean(seed_rec1s):5.1f}%")

    df_summary = pd.DataFrame(summary_records).sort_values("roc_auc_mean", ascending=False)
    sum_csv = diag_dir / "stage_e_models_summary.csv"
    df_summary.to_csv(sum_csv, index=False)

    # -------------------------------------------------------------------------
    # 3. Master Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("STAGE E MASTER COMPARISON TABLE (At 5% Operational Prevalence)")
    print("=" * 110)
    print(f"{'Model Architecture':28s} | {'Train Time':10s} | {'ROC-AUC':17s} | {'PR-AUC':17s} | {'Recall@1% FPR':16s} | {'F1 Score':8s}")
    print("-" * 110)
    for _, r in df_summary.iterrows():
        t_str = f"{r['train_time_sec']:.3f}s"
        auc_str = f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}"
        pr_str = f"{r['pr_auc_mean']:.4f} +/- {r['pr_auc_std']:.4f}"
        rec_str = f"{r['recall_at_1pct_mean']:.1f}% +/- {r['recall_at_1pct_std']:.1f}%"
        f1_str = f"{r['f1_mean']:.4f}"
        print(f"{r['model']:28s} | {t_str:10s} | {auc_str:17s} | {pr_str:17s} | {rec_str:16s} | {f1_str:8s}")
    print("-" * 110)

    # -------------------------------------------------------------------------
    # 4. Per-Class Breakdown Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PER-CLASS ROC-AUC BREAKDOWN BY MODEL ARCHITECTURE")
    print("=" * 115)
    header = f"{'Attack Class':24s} | " + " | ".join([f"{m.name[:10]:10s}" for m in models])
    print(header)
    print("-" * 115)
    matrix_rows = []
    for atk_name in unique_attacks:
        row_str = f"{atk_name:24s} | "
        row_data = {"attack_class": atk_name}
        for m in models:
            auc_val = perclass_matrix[m.name]["auc"].get(atk_name, np.nan)
            row_data[f"{m.name}_auc"] = auc_val
            row_data[f"{m.name}_rec1"] = perclass_matrix[m.name]["rec_at_1pct"].get(atk_name, np.nan)
            row_str += f"{auc_val:10.4f} | "
        print(row_str)
        matrix_rows.append(row_data)
    print("-" * 115)

    df_matrix = pd.DataFrame(matrix_rows)
    mat_csv = diag_dir / "stage_e_perclass_matrix.csv"
    df_matrix.to_csv(mat_csv, index=False)

    # -------------------------------------------------------------------------
    # 5. Comparative Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_summary))
    bars1 = ax.bar(x - 0.2, df_summary["roc_auc_mean"], width=0.4, label="ROC-AUC", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + 0.2, df_summary["pr_auc_mean"], width=0.4, label="PR-AUC (5% Prev)", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Metric Value (0 to 1)", fontsize=11)
    ax.set_title("EDEMS Stage E: Anomaly Detection Algorithm Benchmark on 60s Windows", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    # Annotate bars with values
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = diag_dir / "stage_e_comparison_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved benchmark comparison chart: {plot_path.name}")

    # -------------------------------------------------------------------------
    # 6. Summary JSON
    # -------------------------------------------------------------------------
    summary_data = {
        "execution_time_seconds": round(time.time() - t_start, 2),
        "target_prevalence": target_prev,
        "n_seeds": n_seeds,
        "models": df_summary.to_dict(orient="records"),
        "per_class_matrix": perclass_matrix,
    }
    sum_json = diag_dir / "stage_e_summary.json"
    with open(sum_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Complete summary saved to: {sum_json.name}")
    print("=" * 95)


if __name__ == "__main__":
    run_stage_e()
