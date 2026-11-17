"""
Stage E — Unsupervised Flow-Level Model Comparison (24 PCAP-Reproducible Features).

Objective:
  Compare unsupervised anomaly detection architectures on the single-flow
  representation (24 reproducible features) using the exact Stage B2 evaluation
  harness (50k train / 50k test benign pool, all 14 attack classes evaluated,
  threshold chosen from benign test scores at 1% FPR).

Configurations Evaluated (11 models total):
  1.  IsolationForest (max_samples=256, max_features=0.5)
  2.  IsolationForest (max_samples=256, max_features=1.0)
  3.  IsolationForest (max_samples=2048, max_features=0.5)
  4.  IsolationForest (max_samples=2048, max_features=1.0)
  5.  IsolationForest (max_samples="auto", max_features=0.5)
  6.  IsolationForest (max_samples="auto", max_features=1.0)
  7.  LocalOutlierFactor (novelty=True, k=20) on benign training pool
  8.  PCA Reconstruction Error (explaining 90% variance)
  9.  PCA Reconstruction Error (explaining 95% variance)
  10. PCA Reconstruction Error (explaining 99% variance)
  11. Small Autoencoder MLP (24 -> 16 -> 8 -> 16 -> 24) with early stopping

Preprocessing:
  - log1p on 19 heavy-tailed features (durations, counts, bytes, rates, IATs).
  - StandardScaler fitted strictly on benign TRAIN.
  - Note: IsolationForest is invariant to monotonic log1p, but distance/subspace
    models (LOF, PCA, Autoencoder) strictly require scaling.

Outputs saved to:
  ml/results/stage_e_flow_models_summary.csv
  ml/results/stage_e_flow_perclass_matrix.csv
  ml/results/stage_e_flow_summary.json

To run:
    python scripts/stageE_flow_model_comparison.py
"""

import sys
import os
import json
import time
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    DATA_DIR, RESULTS_DIR, SELECTED_FEATURES, LABEL_COL, BENIGN_LABEL,
    RANDOM_SEED, TARGET_FPR
)

# 19 heavy-tailed physical metrics that span orders of magnitude
HEAVY_TAILED_COLS = [
    c for c in SELECTED_FEATURES
    if c not in ["Protocol", "SYN Flag Count", "RST Flag Count", "ACK Flag Count", "Down/Up Ratio"]
]


def apply_preprocessing(df_features, scaler=None, fit_scaler=False):
    """
    Applies log1p to heavy-tailed columns, then standardizes with StandardScaler.
    Fit is performed STRICTLY on the training set.
    """
    df_transformed = df_features.copy()
    for col in HEAVY_TAILED_COLS:
        df_transformed[col] = np.log1p(np.maximum(df_transformed[col].to_numpy(dtype=np.float32), 0.0))

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_transformed)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(df_transformed)
        return X_scaled, scaler


def run_stage_e():
    print("=" * 105)
    print("EDEMS ML — Stage E: Unsupervised Flow-Level Model Comparison (24 Features)")
    print("=" * 105)

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = RESULTS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Load Data (Exact Stage B2 Harness)
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading benign baseline (50k train / 50k test) and all attack classes...")
    benign_frames = []
    cols_to_load = SELECTED_FEATURES + [LABEL_COL]

    for p in sorted(glob.glob(str(DATA_DIR / "*_cleaned.parquet"))):
        df_p = pd.read_parquet(p, columns=cols_to_load)
        df_b = df_p[df_p[LABEL_COL] == BENIGN_LABEL]
        if len(df_b) > 10000:
            df_b = df_b.sample(n=10000, random_state=RANDOM_SEED)
        benign_frames.append(df_b[SELECTED_FEATURES])

    df_benign_all = pd.concat(benign_frames, ignore_index=True)
    b_tr, b_te = train_test_split(df_benign_all, test_size=0.5, random_state=RANDOM_SEED)

    print(f"  Benign pool: {len(b_tr):,d} train flows, {len(b_te):,d} test flows.")

    # Load attacks
    attack_frames = []
    for p in sorted(glob.glob(str(DATA_DIR / "*_cleaned.parquet"))):
        df_p = pd.read_parquet(p, columns=cols_to_load)
        df_a = df_p[df_p[LABEL_COL] != BENIGN_LABEL]
        if len(df_a) > 0:
            attack_frames.append(df_a)

    df_attacks_all = pd.concat(attack_frames, ignore_index=True)
    unique_attacks = sorted(df_attacks_all[LABEL_COL].unique().tolist())

    attack_test_dict = {}
    print(f"  Discovered {len(unique_attacks)} distinct attack classes across dataset:")
    for a in unique_attacks:
        df_cls = df_attacks_all[df_attacks_all[LABEL_COL] == a][SELECTED_FEATURES]
        # In Stage B2, 50% was held out for test
        if len(df_cls) >= 4:
            _, te = train_test_split(df_cls, test_size=0.5, random_state=RANDOM_SEED)
        else:
            te = df_cls
        cnt = len(df_cls)
        unreliable = cnt < 100
        unreliable_tag = " [UNRELIABLE: N < 100]" if unreliable else ""
        print(f"    - {a:28s}: {cnt:8,d} flows (test N={len(te):,d}){unreliable_tag}")
        attack_test_dict[a] = {
            "df_test": te,
            "total_count": cnt,
            "is_unreliable": unreliable
        }

    # -------------------------------------------------------------------------
    # 2. Preprocess Data
    # -------------------------------------------------------------------------
    print("\n[2/5] Applying preprocessing (log1p on heavy-tailed cols + StandardScaler on Benign Train)...")
    X_train_raw = b_tr.to_numpy(dtype=np.float32)
    X_test_raw = b_te.to_numpy(dtype=np.float32)

    X_train_scaled, scaler = apply_preprocessing(b_tr, fit_scaler=True)
    X_test_scaled, _ = apply_preprocessing(b_te, scaler=scaler, fit_scaler=False)

    attack_test_processed = {}
    for a in unique_attacks:
        df_a_te = attack_test_dict[a]["df_test"]
        X_atk_raw = df_a_te.to_numpy(dtype=np.float32)
        X_atk_scaled, _ = apply_preprocessing(df_a_te, scaler=scaler, fit_scaler=False)
        attack_test_processed[a] = {
            "raw": X_atk_raw,
            "scaled": X_atk_scaled,
            "count": attack_test_dict[a]["total_count"],
            "is_unreliable": attack_test_dict[a]["is_unreliable"]
        }

    # Pre-compute PCA variance components
    pca_probe = PCA(random_state=RANDOM_SEED).fit(X_train_scaled)
    cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
    k90 = int(np.argmax(cum_var >= 0.90) + 1)
    k95 = int(np.argmax(cum_var >= 0.95) + 1)
    k99 = int(np.argmax(cum_var >= 0.99) + 1)
    print(f"  PCA Variance Analysis: 90% var = {k90} comps, 95% var = {k95} comps, 99% var = {k99} comps")

    # -------------------------------------------------------------------------
    # 3. Define the 11 Candidate Configurations
    # -------------------------------------------------------------------------
    print("\n[3/5] Defining 11 candidate model configurations...")
    model_configs = [
        # Isolation Forest variants (use raw or scaled; IForest is monotonic-invariant)
        {
            "name": "IForest (samples=256, feat=0.5)",
            "family": "IsolationForest",
            "model": IsolationForest(n_estimators=100, max_samples=256, max_features=0.5, random_state=RANDOM_SEED, n_jobs=-1),
            "use_scaled": False
        },
        {
            "name": "IForest (samples=256, feat=1.0)",
            "family": "IsolationForest",
            "model": IsolationForest(n_estimators=100, max_samples=256, max_features=1.0, random_state=RANDOM_SEED, n_jobs=-1),
            "use_scaled": False
        },
        {
            "name": "IForest (samples=2048, feat=0.5)",
            "family": "IsolationForest",
            "model": IsolationForest(n_estimators=100, max_samples=2048, max_features=0.5, random_state=RANDOM_SEED, n_jobs=-1),
            "use_scaled": False
        },
        {
            "name": "IForest (samples=2048, feat=1.0)",
            "family": "IsolationForest",
            "model": IsolationForest(n_estimators=100, max_samples=2048, max_features=1.0, random_state=RANDOM_SEED, n_jobs=-1),
            "use_scaled": False
        },
        {
            "name": "IForest (samples=auto, feat=0.5)",
            "family": "IsolationForest",
            "model": IsolationForest(n_estimators=100, max_samples="auto", max_features=0.5, random_state=RANDOM_SEED, n_jobs=-1),
            "use_scaled": False
        },
        {
            "name": "IForest (samples=auto, feat=1.0)",
            "family": "IsolationForest",
            "model": IsolationForest(n_estimators=100, max_samples="auto", max_features=1.0, random_state=RANDOM_SEED, n_jobs=-1),
            "use_scaled": False
        },
        # Local Outlier Factor (novelty=True on benign train)
        {
            "name": "LocalOutlierFactor (k=20, novelty=True)",
            "family": "LOF",
            "model": LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
            "use_scaled": True
        },
        # PCA Reconstruction Error variants
        {
            "name": f"PCA Recon (90% var, k={k90})",
            "family": "PCA",
            "model": PCA(n_components=k90, random_state=RANDOM_SEED),
            "use_scaled": True
        },
        {
            "name": f"PCA Recon (95% var, k={k95})",
            "family": "PCA",
            "model": PCA(n_components=k95, random_state=RANDOM_SEED),
            "use_scaled": True
        },
        {
            "name": f"PCA Recon (99% var, k={k99})",
            "family": "PCA",
            "model": PCA(n_components=k99, random_state=RANDOM_SEED),
            "use_scaled": True
        },
        # Small Autoencoder MLP (24 -> 16 -> 8 -> 16 -> 24)
        {
            "name": "Autoencoder MLP (16-8-16)",
            "family": "Autoencoder",
            "model": MLPRegressor(
                hidden_layer_sizes=(16, 8, 16),
                activation="relu",
                max_iter=250,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=RANDOM_SEED
            ),
            "use_scaled": True
        },
    ]

    # -------------------------------------------------------------------------
    # 4. Train, Score & Evaluate Models
    # -------------------------------------------------------------------------
    print("\n[4/5] Training and benchmarking all 11 configurations on benign flows...")

    def score_data(cfg, data_mat):
        """Standardized scoring helper: higher score = more anomalous."""
        fam = cfg["family"]
        m = cfg["model"]
        if fam == "IsolationForest":
            return -m.score_samples(data_mat)
        elif fam == "LOF":
            return -m.score_samples(data_mat)
        elif fam == "PCA":
            recon = m.inverse_transform(m.transform(data_mat))
            return np.mean((data_mat - recon) ** 2, axis=1)
        elif fam == "Autoencoder":
            recon = m.predict(data_mat)
            return np.mean((data_mat - recon) ** 2, axis=1)
        else:
            raise ValueError(f"Unknown family: {fam}")

    summary_records = []
    perclass_matrix_records = []

    for cfg in model_configs:
        name = cfg["name"]
        fam = cfg["family"]
        use_scaled = cfg["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train_raw
        X_te = X_test_scaled if use_scaled else X_test_raw

        # 1. Training Time
        t0 = time.time()
        if fam == "Autoencoder":
            cfg["model"].fit(X_tr, X_tr)
        else:
            cfg["model"].fit(X_tr)
        t_train = time.time() - t0

        # 2. Scoring Time per 1,000 flows
        sample_1k = X_te[:1000]
        t0 = time.time()
        _ = score_data(cfg, sample_1k)
        t_score_1k_ms = (time.time() - t0) * 1000.0

        # 3. Benign Validation Scores & 1% FPR Threshold
        # We score the held-out benign test pool (50k flows)
        # For LOF, evaluate on 20,000 to keep run time responsive
        if fam == "LOF":
            eval_benign = X_te[:20000]
        else:
            eval_benign = X_te

        benign_scores = score_data(cfg, eval_benign)
        thresh_1pct = float(np.percentile(benign_scores, 99.0))

        # 4. Per-Class Evaluation
        class_aucs = {}
        class_rec1s = {}

        for a in unique_attacks:
            atk_mat = attack_test_processed[a]["scaled"] if use_scaled else attack_test_processed[a]["raw"]
            # Subsample very large attack classes to 25k for fast responsive scoring on distance models
            if len(atk_mat) > 25000:
                rng = np.random.RandomState(RANDOM_SEED)
                sub_idx = rng.choice(len(atk_mat), size=25000, replace=False)
                atk_eval = atk_mat[sub_idx]
            else:
                atk_eval = atk_mat

            atk_scores = score_data(cfg, atk_eval)

            y_eval = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(atk_scores))])
            scores_eval = np.concatenate([benign_scores, atk_scores])

            try:
                auc_val = float(roc_auc_score(y_eval, scores_eval))
            except ValueError:
                auc_val = np.nan

            rec_val = float(np.mean(atk_scores >= thresh_1pct) * 100.0)

            class_aucs[a] = auc_val
            class_rec1s[a] = rec_val

            perclass_matrix_records.append({
                "model": name,
                "attack_class": a,
                "is_unreliable": attack_test_processed[a]["is_unreliable"],
                "roc_auc": round(auc_val, 4),
                "recall_at_1pct_fpr": round(rec_val, 2),
                "inverted": bool(auc_val < 0.50) if not np.isnan(auc_val) else False
            })

        # Reliable classes only (N >= 100)
        reliable_aucs = [
            class_aucs[a] for a in unique_attacks
            if not attack_test_processed[a]["is_unreliable"] and not np.isnan(class_aucs[a])
        ]
        mean_reliable_auc = float(np.mean(reliable_aucs)) if len(reliable_aucs) > 0 else np.nan

        # Reliable classes recall @ 1% FPR
        reliable_recs = [
            class_rec1s[a] for a in unique_attacks
            if not attack_test_processed[a]["is_unreliable"]
        ]
        mean_reliable_rec1 = float(np.mean(reliable_recs)) if len(reliable_recs) > 0 else np.nan

        # Count of inverted classes (AUC < 0.5)
        num_inverted = sum(1 for a in unique_attacks if class_aucs[a] < 0.50)

        summary_records.append({
            "model": name,
            "family": fam,
            "train_time_sec": round(t_train, 3),
            "score_time_1k_ms": round(t_score_1k_ms, 2),
            "mean_reliable_auc": round(mean_reliable_auc, 4),
            "mean_reliable_rec1": round(mean_reliable_rec1, 2),
            "num_inverted_classes": num_inverted,
            "hulk_auc": round(class_aucs.get("DoS attacks-Hulk", np.nan), 4),
            "hoic_auc": round(class_aucs.get("DDOS attack-HOIC", np.nan), 4),
            "infil_auc": round(class_aucs.get("Infilteration", np.nan), 4),
            "loic_udp_auc": round(class_aucs.get("DDOS attack-LOIC-UDP", np.nan), 4),
            "bot_auc": round(class_aucs.get("Bot", np.nan), 4),
            "ssh_auc": round(class_aucs.get("SSH-Bruteforce", np.nan), 4),
        })

        print(f"  {name:36s} | Train: {t_train:6.2f}s | Score 1k: {t_score_1k_ms:5.1f}ms | Mean Rel AUC: {mean_reliable_auc:.4f} | Inverted: {num_inverted}")

    # -------------------------------------------------------------------------
    # 5. Output Summary & Master Tables
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating Summary Tables...")
    df_summary = pd.DataFrame(summary_records).sort_values("mean_reliable_auc", ascending=False)
    sum_csv = results_dir / "stage_e_flow_models_summary.csv"
    df_summary.to_csv(sum_csv, index=False)
    df_summary.to_csv(diag_dir / "stage_e_flow_models_summary.csv", index=False)

    df_perclass = pd.DataFrame(perclass_matrix_records)
    mat_csv = results_dir / "stage_e_flow_perclass_matrix.csv"
    df_perclass.to_csv(mat_csv, index=False)
    df_perclass.to_csv(diag_dir / "stage_e_flow_perclass_matrix.csv", index=False)

    print("\n" + "=" * 120)
    print("STAGE E MASTER FLOW-LEVEL MODEL COMPARISON TABLE (11 Configurations)")
    print("=" * 120)
    print(f"{'Model Configuration':36s} | {'Train (s)':9s} | {'1k Lat (ms)':11s} | {'Mean Rel AUC':12s} | {'Mean Rel Rec@1%':15s} | {'AUC < 0.5':9s}")
    print("-" * 120)
    for _, r in df_summary.iterrows():
        print(f"{r['model']:36s} | {r['train_time_sec']:9.2f} | {r['score_time_1k_ms']:11.1f} | {r['mean_reliable_auc']:12.4f} | {r['mean_reliable_rec1']:14.1f}% | {r['num_inverted_classes']:9d}")
    print("-" * 120)

    # Save complete JSON
    summary_json = {
        "execution_time_seconds": round(time.time() - t_start, 2),
        "models": df_summary.to_dict(orient="records"),
        "best_model_by_mean_auc": df_summary.iloc[0]["model"],
    }
    json_path = results_dir / "stage_e_flow_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    with open(diag_dir / "stage_e_flow_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nSaved summary artifacts to:")
    print(f"  - {sum_csv}")
    print(f"  - {mat_csv}")
    print(f"  - {json_path}")
    print(f"Execution finished in {time.time()-t_start:.1f}s")
    print("=" * 105)


if __name__ == "__main__":
    run_stage_e()
