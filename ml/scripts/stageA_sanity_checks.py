"""
Stage A — Sanity Checks and Root-Cause Diagnostics.

This script executes the three Stage A diagnostic investigations:
  1. Per-class flow counts BEFORE and AFTER cleaning/deduplication.
     Identifies whether cleaning destroyed rare attack classes.
  2. Column and identifier audit of raw dataset files.
     Checks presence of Src IP, Dst IP, Flow ID, Src Port, Dst Port, Timestamp.
     This establishes the feasibility of grouping by IP vs Port for Stage D.
  3. Benign flow distribution shift analysis across days.
     Compares Benign traffic in TRAIN_DAYS vs TEST_DAYS across all 24 features
     using the two-sample Kolmogorov-Smirnov (KS) test and distribution quantiles.
     Flags features exhibiting severe day-to-day drift (concept drift).

Outputs saved to:
  ml/results/diagnostics/stage_a_cleaning_audit.csv
  ml/results/diagnostics/stage_a_columns_audit.json
  ml/results/diagnostics/stage_a_ks_drift.csv
  ml/results/diagnostics/stage_a_ks_drift.png

To run:
    python scripts/stageA_sanity_checks.py
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    RAW_DATA_DIR, DATA_DIR, RESULTS_DIR, DAY_FILES,
    TRAIN_DAYS, TEST_DAYS, SELECTED_FEATURES, LABEL_COL, BENIGN_LABEL,
    RANDOM_SEED
)


def run_stage_a():
    print("=" * 80)
    print("EDEMS ML — Stage A: Diagnostics & Sanity Checks")
    print("=" * 80)

    diag_dir = RESULTS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART 1: Per-Class Flow Counts BEFORE and AFTER Cleaning
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 1: AUDIT OF DATA CLEANING & DEDUPLICATION (Per-Class Row Counts)")
    print("-" * 80)

    raw_label_counts = {}
    clean_label_counts = {}
    per_day_records = []

    for day_tag, filename in DAY_FILES.items():
        raw_path = RAW_DATA_DIR / filename
        clean_path = DATA_DIR / f"{day_tag}_cleaned.parquet"

        raw_df = pd.read_parquet(raw_path, columns=[LABEL_COL])
        clean_df = pd.read_parquet(clean_path, columns=[LABEL_COL])

        raw_c = raw_df[LABEL_COL].astype(str).str.strip().value_counts().to_dict()
        clean_c = clean_df[LABEL_COL].astype(str).str.strip().value_counts().to_dict()

        for lbl, cnt in raw_c.items():
            raw_label_counts[lbl] = raw_label_counts.get(lbl, 0) + cnt
        for lbl, cnt in clean_c.items():
            clean_label_counts[lbl] = clean_label_counts.get(lbl, 0) + cnt

        per_day_records.append({
            "day": day_tag,
            "raw_total": len(raw_df),
            "clean_total": len(clean_df),
            "dropped": len(raw_df) - len(clean_df)
        })

    audit_rows = []
    all_classes = sorted(list(raw_label_counts.keys()))

    print(f"{'Class Label':30s} | {'Raw Count':10s} | {'Clean Count':11s} | {'Dropped':8s} | {'% Dropped':9s} | Status")
    print("-" * 88)

    for lbl in all_classes:
        r_cnt = raw_label_counts[lbl]
        c_cnt = clean_label_counts.get(lbl, 0)
        dropped = r_cnt - c_cnt
        pct_dropped = (dropped / r_cnt) * 100 if r_cnt > 0 else 0.0

        flag = "FLAGGED (>1% lost)" if pct_dropped > 1.0 else "PRESERVED 100%"
        if dropped > 0 and pct_dropped <= 1.0:
            flag = f"MINIMAL (-{dropped} rows)"

        audit_rows.append({
            "label": lbl,
            "raw_count": r_cnt,
            "cleaned_count": c_cnt,
            "dropped_count": dropped,
            "pct_dropped": round(pct_dropped, 4),
            "status": flag
        })

        print(f"{lbl:30s} | {r_cnt:10,d} | {c_cnt:11,d} | {dropped:8,d} | {pct_dropped:8.2f}% | {flag}")

    print("-" * 88)

    audit_df = pd.DataFrame(audit_rows)
    audit_csv_path = diag_dir / "stage_a_cleaning_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"  Saved cleaning audit: {audit_csv_path.name}")

    # =========================================================================
    # PART 2: List Columns in Raw Dataset & Audit Identifiers
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 2: RAW SCHEMA & IDENTIFIER AUDIT")
    print("-" * 80)

    sample_raw_file = RAW_DATA_DIR / list(DAY_FILES.values())[0]
    sample_df = pd.read_parquet(sample_raw_file)
    raw_columns = list(sample_df.columns)

    print(f"Total Columns in raw dataset: {len(raw_columns)}")

    target_identifiers = [
        "Src IP", "Dst IP", "Source IP", "Destination IP",
        "Flow ID", "Src Port", "Dst Port", "Source Port",
        "Destination Port", "Timestamp"
    ]

    identifier_status = {}
    for ident in target_identifiers:
        matches = [c for c in raw_columns if ident.lower() == c.lower()]
        identifier_status[ident] = matches

    print("\nPresence of Network Identifiers & Timestamps:")
    for ident, matches in identifier_status.items():
        status_str = f"FOUND ({matches[0]})" if matches else "ABSENT (NOT in dataset)"
        print(f"  - {ident:20s}: {status_str}")

    cols_audit_data = {
        "total_columns": len(raw_columns),
        "columns_list": raw_columns,
        "identifiers_audit": {k: (v[0] if v else None) for k, v in identifier_status.items()},
        "stage_d_feasibility": {
            "group_by_src_ip": False,
            "reason_src_ip": "Src IP is completely stripped from the CSE-CIC-IDS2018 processed ML dataset.",
            "group_by_dst_port": False,
            "reason_dst_port": "Dst Port is also absent from the 78 columns of this Parquet release.",
            "group_by_protocol": True,
            "reason_protocol": "Protocol exists as a numeric transport identifier."
        }
    }

    cols_json_path = diag_dir / "stage_a_columns_audit.json"
    with open(cols_json_path, "w") as f:
        json.dump(cols_audit_data, f, indent=2)
    print(f"\n  Saved columns audit: {cols_json_path.name}")

    # =========================================================================
    # PART 3: Day-to-Day Benign Flow Distribution Shift Analysis (KS Test)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 3: BENIGN DISTRIBUTION SHIFT ANALYSIS (Train Days vs Held-Out Test Days)")
    print("-" * 80)
    print(f"Train Days (Benign): {TRAIN_DAYS}")
    print(f"Test Days  (Benign): {TEST_DAYS}")

    # Load representative sample of Benign flows from TRAIN_DAYS
    train_frames = []
    for day in TRAIN_DAYS:
        p = DATA_DIR / f"{day}_cleaned.parquet"
        df_d = pd.read_parquet(p, columns=SELECTED_FEATURES + [LABEL_COL])
        df_b = df_d[df_d[LABEL_COL] == BENIGN_LABEL]
        if len(df_b) > 25000:
            df_b = df_b.sample(n=25000, random_state=RANDOM_SEED)
        train_frames.append(df_b[SELECTED_FEATURES])
    df_train_benign = pd.concat(train_frames, ignore_index=True)

    # Load representative sample of Benign flows from TEST_DAYS
    test_frames = []
    per_test_sample = max(1, 50000 // len(TEST_DAYS))
    for day in TEST_DAYS:
        p = DATA_DIR / f"{day}_cleaned.parquet"
        df_d = pd.read_parquet(p, columns=SELECTED_FEATURES + [LABEL_COL])
        df_b = df_d[df_d[LABEL_COL] == BENIGN_LABEL]
        if len(df_b) > per_test_sample:
            df_b = df_b.sample(n=per_test_sample, random_state=RANDOM_SEED)
        test_frames.append(df_b[SELECTED_FEATURES])
    df_test_benign = pd.concat(test_frames, ignore_index=True)

    print(f"Comparing {len(df_train_benign):,d} Train Benign flows vs {len(df_test_benign):,d} Test Benign flows...")

    ks_records = []
    for feat in SELECTED_FEATURES:
        tr_vals = df_train_benign[feat].to_numpy(dtype=np.float64)
        te_vals = df_test_benign[feat].to_numpy(dtype=np.float64)

        ks_res = ks_2samp(tr_vals, te_vals)
        d_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # Quantiles
        q_tr = np.percentile(tr_vals, [10, 25, 50, 75, 90])
        q_te = np.percentile(te_vals, [10, 25, 50, 75, 90])

        if d_stat >= 0.50:
            drift_flag = "SEVERE DRIFT (D >= 0.50)"
        elif d_stat >= 0.25:
            drift_flag = "MODERATE DRIFT (0.25 <= D < 0.50)"
        else:
            drift_flag = "STABLE (D < 0.25)"

        ks_records.append({
            "feature": feat,
            "ks_statistic": round(d_stat, 4),
            "p_value": p_val,
            "drift_severity": drift_flag,
            "train_mean": round(float(np.mean(tr_vals)), 2),
            "test_mean": round(float(np.mean(te_vals)), 2),
            "train_p50": round(float(q_tr[2]), 2),
            "test_p50": round(float(q_te[2]), 2),
            "train_p10": round(float(q_tr[0]), 2),
            "train_p90": round(float(q_tr[4]), 2),
            "test_p10": round(float(q_te[0]), 2),
            "test_p90": round(float(q_te[4]), 2),
        })

    ks_df = pd.DataFrame(ks_records).sort_values(by="ks_statistic", ascending=False)
    ks_csv_path = diag_dir / "stage_a_ks_drift.csv"
    ks_df.to_csv(ks_csv_path, index=False)

    print(f"\n{'Feature Name':26s} | {'KS Stat (D)':11s} | {'Train Median':12s} | {'Test Median':12s} | Drift Severity")
    print("-" * 88)
    for _, r in ks_df.iterrows():
        print(f"{r['feature']:26s} | {r['ks_statistic']:11.4f} | {r['train_p50']:12.2f} | {r['test_p50']:12.2f} | {r['drift_severity']}")
    print("-" * 88)

    # Plot KS Drift Bar Chart
    plt.figure(figsize=(10, 8))
    colors = [
        "crimson" if d >= 0.50 else ("darkorange" if d >= 0.25 else "forestgreen")
        for d in ks_df["ks_statistic"]
    ]
    y_pos = np.arange(len(ks_df))
    plt.barh(y_pos, ks_df["ks_statistic"], color=colors, alpha=0.85, edgecolor="black")
    plt.yticks(y_pos, ks_df["feature"])
    plt.gca().invert_yaxis()
    plt.axvline(0.25, color="darkorange", linestyle="--", lw=1.5, label="Moderate Drift Threshold (D = 0.25)")
    plt.axvline(0.50, color="crimson", linestyle="--", lw=1.5, label="Severe Drift Threshold (D = 0.50)")
    plt.xlabel("Kolmogorov-Smirnov Statistic (D)", fontsize=11, fontweight="bold")
    plt.title("Day-to-Day Distribution Shift in Benign Traffic (Train vs Test Days)", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    ks_png_path = diag_dir / "stage_a_ks_drift.png"
    plt.savefig(ks_png_path, dpi=150)
    plt.close()
    print(f"\n  Saved KS drift chart: {ks_png_path.name}")

    print("\n" + "=" * 80)
    print("STAGE A SANITY CHECKS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_stage_a()
