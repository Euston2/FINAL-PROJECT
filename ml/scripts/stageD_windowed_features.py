"""
Stage D — Windowed Aggregate Features & Anomaly Detection.

Problem Addressed:
  Stages B2 and C revealed that single-flow features cannot detect volumetric
  attacks (DoS Hulk, DDoS HOIC, DDoS LOIC-HTTP, Botnet, SSH-Bruteforce) because
  individual flows use legitimate TCP/HTTP structures and form dense clusters
  in feature space, causing IsolationForest to score them as "more normal"
  than benign baseline traffic (e.g., Hulk ROC-AUC was 0.2282).

Architectural Solution:
  In the EDEMS architecture, live traffic is captured into rotating 60-second
  PCAP files by edems_capture.exe. This stage lifts feature extraction from
  individual flows to the 60-second capture window level:
    - 480 non-overlapping windows per 8-hour capture day.
    - Each window aggregates all flows occurring in that 60-second period.
    - 22 aggregate features capture burst volume, rate, protocol mix, flag
      asymmetry (SYN-only, zero-backward packets), and parameter variances.
    - Trained strictly on benign windows from TRAIN_DAYS.
    - Evaluated using Stage C methodology (operational 1% and 5% prevalence,
      10 random seeds, per-class ROC-AUC, recall at 1% FPR).
    - Direct side-by-side comparison with the Stage C flow-level baseline.

Outputs saved to:
  ml/results/diagnostics/stage_d_windowed_eval.csv
  ml/results/diagnostics/stage_d_perclass_comparison.csv
  ml/results/diagnostics/stage_d_comparison_plot.png
  ml/results/diagnostics/stage_d_summary.json

To run:
    python scripts/stageD_windowed_features.py
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    DATA_DIR, RESULTS_DIR, TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    LABEL_COL, BENIGN_LABEL, RANDOM_SEED, TARGET_FPR, IFOREST_PARAMS
)

# Number of 60-second capture windows in an 8-hour daily capture (8 * 3600 / 60)
WINDOWS_PER_DAY = 480

WINDOW_FEATURE_NAMES = [
    "win_flow_count",
    "win_flows_per_sec",
    "win_total_bytes",
    "win_bytes_per_sec",
    "win_total_packets",
    "win_packets_per_sec",
    "win_pct_zero_bwd_pkts",
    "win_pct_syn_only",
    "win_pct_rst",
    "win_pct_tcp",
    "win_pct_udp",
    "win_duration_mean",
    "win_duration_std",
    "win_fwd_len_mean",
    "win_fwd_len_std",
    "win_fwd_len_max",
    "win_bwd_len_mean",
    "win_bwd_len_max",
    "win_iat_mean",
    "win_iat_std",
    "win_iat_max",
    "win_down_up_ratio_mean",
]


def extract_day_windows(day_tag, n_windows=WINDOWS_PER_DAY):
    """
    Extracts 60-second aggregate window features from a cleaned day parquet file.
    """
    path = DATA_DIR / f"{day_tag}_cleaned.parquet"
    df = pd.read_parquet(path)

    # Divide day into non-overlapping sequential chunks
    chunks = np.array_split(np.arange(len(df)), n_windows)
    records = []

    for win_idx, idxs in enumerate(chunks):
        sub = df.iloc[idxs]
        n_flows = len(sub)
        if n_flows == 0:
            continue

        is_atk = (sub[LABEL_COL] != BENIGN_LABEL)
        atk_count = int(is_atk.sum())

        if atk_count > 0:
            atk_labels, counts = np.unique(sub.loc[is_atk, LABEL_COL], return_counts=True)
            primary_class = str(atk_labels[np.argmax(counts)])
        else:
            primary_class = BENIGN_LABEL

        tot_bytes = float((sub["Fwd Packets Length Total"] + sub["Bwd Packets Length Total"]).sum())
        tot_pkts = float((sub["Total Fwd Packets"] + sub["Total Backward Packets"]).sum())

        records.append({
            "day": day_tag,
            "window_idx": win_idx,
            "win_flow_count": n_flows,
            "win_flows_per_sec": n_flows / 60.0,
            "win_total_bytes": tot_bytes,
            "win_bytes_per_sec": tot_bytes / 60.0,
            "win_total_packets": tot_pkts,
            "win_packets_per_sec": tot_pkts / 60.0,
            "win_pct_zero_bwd_pkts": float((sub["Total Backward Packets"] == 0).mean()),
            "win_pct_syn_only": float(((sub["SYN Flag Count"] > 0) & (sub["ACK Flag Count"] == 0)).mean()),
            "win_pct_rst": float((sub["RST Flag Count"] > 0).mean()),
            "win_pct_tcp": float((sub["Protocol"] == 6).mean()),
            "win_pct_udp": float((sub["Protocol"] == 17).mean()),
            "win_duration_mean": float(sub["Flow Duration"].mean()),
            "win_duration_std": float(sub["Flow Duration"].std()) if n_flows > 1 else 0.0,
            "win_fwd_len_mean": float(sub["Fwd Packet Length Mean"].mean()),
            "win_fwd_len_std": float(sub["Fwd Packet Length Std"].mean()),
            "win_fwd_len_max": float(sub["Fwd Packet Length Max"].max()),
            "win_bwd_len_mean": float(sub["Bwd Packet Length Mean"].mean()),
            "win_bwd_len_max": float(sub["Bwd Packet Length Max"].max()),
            "win_iat_mean": float(sub["Flow IAT Mean"].mean()),
            "win_iat_std": float(sub["Flow IAT Std"].mean()),
            "win_iat_max": float(sub["Flow IAT Max"].max()),
            "win_down_up_ratio_mean": float(sub["Down/Up Ratio"].mean()),
            "is_attack": int(atk_count > 0),
            "atk_flow_count": atk_count,
            "atk_ratio": atk_count / n_flows,
            "primary_attack": primary_class,
        })

    return pd.DataFrame(records)


def evaluate_windows_at_prevalence(
    model, X_benign, X_attacks, threshold,
    target_prevalence, n_seeds=10, base_seed=42
):
    """
    Subsamples attack windows to operational prevalence and computes metrics.
    """
    n_benign = len(X_benign)
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

        anomaly_scores = -model.score_samples(X_eval)
        neg_threshold = -threshold
        y_pred = (anomaly_scores >= neg_threshold).astype(int)

        actual_prev = float(y_true.mean())
        base_pr = actual_prev

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

        benign_scores = anomaly_scores[:n_benign]
        atk_scores = anomaly_scores[n_benign:]
        thresh_1pct = float(np.percentile(benign_scores, 99.0))
        rec_1pct = float(np.mean(atk_scores >= thresh_1pct) * 100) if len(atk_scores) > 0 else 0.0

        seed_results.append({
            "target_prevalence": target_prevalence,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "baseline_pr_auc": base_pr,
            "pr_auc_above_baseline": pr_auc - base_pr if not np.isnan(pr_auc) else np.nan,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "recall_at_1pct_fpr": rec_1pct,
        })

    return seed_results


def run_stage_d():
    print("=" * 90)
    print("EDEMS ML — Stage D: Windowed Aggregate Features & Model Evaluation")
    print("=" * 90)

    diag_dir = RESULTS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Extract Windows Across All Days
    # -------------------------------------------------------------------------
    print("\n[1/5] Extracting 60-second window aggregates across all days...")
    all_day_dfs = {}
    for day in (TRAIN_DAYS + VAL_DAYS + TEST_DAYS):
        t0 = time.time()
        w_df = extract_day_windows(day)
        all_day_dfs[day] = w_df
        n_atk = (w_df["is_attack"] == 1).sum()
        n_ben = (w_df["is_attack"] == 0).sum()
        print(f"  Day {day:12s}: {len(w_df):3d} windows ({n_ben:3d} pure benign, {n_atk:3d} attack) in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # 2. Train IsolationForest on Benign Windows
    # -------------------------------------------------------------------------
    print("\n[2/5] Training IsolationForest on purely benign training windows...")
    train_frames = [all_day_dfs[d][all_day_dfs[d]["is_attack"] == 0] for d in TRAIN_DAYS]
    df_train_benign = pd.concat(train_frames, ignore_index=True)
    X_train = df_train_benign[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)

    print(f"  Training pool: {len(X_train)} benign windows from {TRAIN_DAYS}")
    model = IsolationForest(
        n_estimators=IFOREST_PARAMS["n_estimators"],
        max_samples="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train)

    # Calibrate threshold on held-out validation benign windows
    val_frames = [all_day_dfs[d][all_day_dfs[d]["is_attack"] == 0] for d in VAL_DAYS]
    df_val_benign = pd.concat(val_frames, ignore_index=True)
    X_val = df_val_benign[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)

    val_scores = model.score_samples(X_val)
    # Threshold for 1% false alarm rate: 1st percentile of score_samples
    threshold = float(np.percentile(val_scores, TARGET_FPR * 100))
    print(f"  Validation pool: {len(X_val)} benign windows from {VAL_DAYS}")
    print(f"  Calibrated anomaly threshold (tau at 1% FPR): {threshold:+.6f}")

    # -------------------------------------------------------------------------
    # 3. Assemble Test Pool
    # -------------------------------------------------------------------------
    print("\n[3/5] Assembling held-out test windows from 7 TEST_DAYS...")
    test_frames = [all_day_dfs[d] for d in TEST_DAYS]
    df_test_all = pd.concat(test_frames, ignore_index=True)

    df_test_benign = df_test_all[df_test_all["is_attack"] == 0]
    df_test_attack = df_test_all[df_test_all["is_attack"] == 1]

    # Augment test benign with validation benign to ensure stable denominator
    df_test_benign_pool = pd.concat([df_test_benign, df_val_benign], ignore_index=True)

    X_test_benign = df_test_benign_pool[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)
    X_test_attack = df_test_attack[WINDOW_FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)
    labels_test_attack = df_test_attack["primary_attack"].to_numpy()

    print(f"  Held-out Benign Windows: {len(X_test_benign)}")
    print(f"  Held-out Attack Windows: {len(X_test_attack)} across {len(np.unique(labels_test_attack))} primary classes")

    # -------------------------------------------------------------------------
    # 4. Operational Prevalence Evaluation (1% and 5%)
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating at operational attack prevalence (1% and 5%, 10 seeds)...")
    eval_records = []
    for prev in [0.01, 0.05]:
        seed_res = evaluate_windows_at_prevalence(
            model, X_test_benign, X_test_attack, threshold,
            target_prevalence=prev, n_seeds=10, base_seed=RANDOM_SEED
        )
        eval_records.extend(seed_res)
        df_p = pd.DataFrame(seed_res)
        print(f"\n  Results at {prev*100:.0f}% prevalence (mean +/- std across 10 seeds):")
        for m in ["roc_auc", "pr_auc", "baseline_pr_auc", "pr_auc_above_baseline",
                  "precision", "recall", "f1", "recall_at_1pct_fpr"]:
            v = df_p[m].dropna()
            if m == "recall_at_1pct_fpr":
                print(f"    {m:25s}: {v.mean():7.2f}% +/- {v.std():5.2f}%")
            else:
                print(f"    {m:25s}: {v.mean():7.4f} +/- {v.std():5.4f}")

    eval_df = pd.DataFrame(eval_records)
    eval_csv = diag_dir / "stage_d_windowed_eval.csv"
    eval_df.to_csv(eval_csv, index=False)

    # -------------------------------------------------------------------------
    # 5. Per-Class ROC-AUC & Direct Comparison with Flow-Level (Stage C)
    # -------------------------------------------------------------------------
    print("\n[5/5] Computing per-class metrics and comparing Flow vs Window...")

    benign_scores = -model.score_samples(X_test_benign)
    benign_thresh_1pct = float(np.percentile(benign_scores, 99.0))

    # Load Stage C flow-level per-class results for direct comparison
    stage_c_path = diag_dir / "stage_c_perclass_auc.csv"
    stage_c_lookup = {}
    if stage_c_path.exists():
        df_c = pd.read_csv(stage_c_path)
        for _, row in df_c.iterrows():
            stage_c_lookup[row["attack_class"]] = {
                "flow_auc": row["roc_auc"],
                "flow_rec1": row["recall_at_1pct_fpr"],
            }

    perclass_records = []
    unique_atks = sorted(np.unique(labels_test_attack))

    for atk_name in unique_atks:
        mask = (labels_test_attack == atk_name)
        X_atk_sub = X_test_attack[mask]
        n_windows = len(X_atk_sub)
        unreliable = n_windows < 5

        atk_scores = -model.score_samples(X_atk_sub)
        y_eval = np.concatenate([np.zeros(len(X_test_benign)), np.ones(n_windows)])
        scores_eval = np.concatenate([benign_scores, atk_scores])

        try:
            win_auc = float(roc_auc_score(y_eval, scores_eval))
        except ValueError:
            win_auc = np.nan

        win_rec1 = float(np.mean(atk_scores >= benign_thresh_1pct) * 100)

        c_info = stage_c_lookup.get(atk_name, {"flow_auc": np.nan, "flow_rec1": np.nan})

        delta_auc = win_auc - c_info["flow_auc"] if not np.isnan(win_auc) and not np.isnan(c_info["flow_auc"]) else np.nan

        perclass_records.append({
            "attack_class": atk_name,
            "window_count": n_windows,
            "flow_level_auc": c_info["flow_auc"],
            "window_level_auc": round(win_auc, 4),
            "delta_auc": round(delta_auc, 4) if not np.isnan(delta_auc) else np.nan,
            "flow_rec_at_1pct": c_info["flow_rec1"],
            "win_rec_at_1pct": round(win_rec1, 2),
            "is_unreliable": unreliable,
        })

    comp_df = pd.DataFrame(perclass_records).sort_values("window_level_auc", ascending=False)
    comp_csv = diag_dir / "stage_d_perclass_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    print("\n" + "=" * 115)
    print("STAGE D MASTER COMPARATIVE TABLE: FLOW-LEVEL (Stage C) vs. WINDOW-LEVEL (Stage D)")
    print("=" * 115)
    print(f"{'Attack Class':26s} | {'Windows':7s} | {'Flow AUC':8s} | {'Win AUC':8s} | {'Delta AUC':9s} | {'Flow Rec@1%':11s} | {'Win Rec@1%':10s} | Status")
    print("-" * 115)
    for _, r in comp_df.iterrows():
        rel = "UNRELIABLE (N<5)" if r["is_unreliable"] else "OK"
        w_cnt = f"{r['window_count']}"
        if r["is_unreliable"]:
            w_cnt += "*"
        f_auc = f"{r['flow_level_auc']:.4f}" if not np.isnan(r['flow_level_auc']) else "N/A"
        w_auc = f"{r['window_level_auc']:.4f}" if not np.isnan(r['window_level_auc']) else "N/A"
        d_auc = f"{r['delta_auc']:+.4f}" if not np.isnan(r['delta_auc']) else "N/A"
        f_rec = f"{r['flow_rec_at_1pct']:.1f}%" if not np.isnan(r['flow_rec_at_1pct']) else "N/A"
        w_rec = f"{r['win_rec_at_1pct']:.1f}%" if not np.isnan(r['win_rec_at_1pct']) else "N/A"
        print(f"{r['attack_class']:26s} | {w_cnt:7s} | {f_auc:8s} | {w_auc:8s} | {d_auc:9s} | {f_rec:11s} | {w_rec:10s} | {rel}")
    print("-" * 115)

    # -------------------------------------------------------------------------
    # 6. Comparative Bar Plot
    # -------------------------------------------------------------------------
    plot_df = comp_df[~comp_df["is_unreliable"]].dropna(subset=["flow_level_auc", "window_level_auc"])
    if len(plot_df) > 0:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(plot_df))
        w = 0.35
        plt.bar(x - w/2, plot_df["flow_level_auc"], width=w, label="Flow-Level (Stage C)", color="#e74c3c", alpha=0.85)
        plt.bar(x + w/2, plot_df["window_level_auc"], width=w, label="Window-Level (Stage D)", color="#2ecc71", alpha=0.85)
        plt.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random Guess (0.50)")
        plt.xticks(x, plot_df["attack_class"], rotation=30, ha="right", fontsize=9)
        plt.ylabel("ROC-AUC Score", fontsize=11)
        plt.title("EDEMS Diagnostics: Single-Flow vs. 60-Second Windowed ROC-AUC", fontsize=13, fontweight="bold")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plot_path = diag_dir / "stage_d_comparison_plot.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved comparative chart to: {plot_path.name}")

    # -------------------------------------------------------------------------
    # 7. Summary JSON
    # -------------------------------------------------------------------------
    summary = {
        "execution_time_seconds": round(time.time() - t_start, 2),
        "windows_per_day": WINDOWS_PER_DAY,
        "n_train_benign_windows": len(X_train),
        "n_val_benign_windows": len(X_val),
        "n_test_benign_windows": len(X_test_benign),
        "n_test_attack_windows": len(X_test_attack),
        "calibrated_threshold": threshold,
        "prevalence_metrics": {},
        "per_class": comp_df.to_dict(orient="records"),
    }
    for prev in [0.01, 0.05]:
        df_p = eval_df[eval_df["target_prevalence"] == prev]
        key = f"prevalence_{int(prev*100)}pct"
        summary["prevalence_metrics"][key] = {
            m: {"mean": round(float(df_p[m].mean()), 4), "std": round(float(df_p[m].std()), 4)}
            for m in ["roc_auc", "pr_auc", "baseline_pr_auc", "pr_auc_above_baseline",
                      "precision", "recall", "f1", "recall_at_1pct_fpr"]
        }

    sum_path = diag_dir / "stage_d_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExecution Complete in {time.time()-t_start:.1f}s")
    print(f"Generated files in {diag_dir}:")
    print(f"  - stage_d_windowed_eval.csv")
    print(f"  - stage_d_perclass_comparison.csv")
    print(f"  - stage_d_comparison_plot.png")
    print(f"  - stage_d_summary.json")
    print("=" * 90)


if __name__ == "__main__":
    run_stage_d()
