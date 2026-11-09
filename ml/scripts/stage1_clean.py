"""
Stage 1 — Load and Clean Traffic Flow Datasets.

This script processes each day's dataset file, applies rigorous and defensible
cleaning rules, saves the sanitized files into ml/data/ as Parquet, and outputs
detailed audit statistics (before/after row counts, dropped reasons, and label
distributions).

To run:
    python scripts/stage1_clean.py
"""

import sys
import os
import time
import pandas as pd
from tqdm import tqdm

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    RAW_DATA_DIR, DATA_DIR, DAY_FILES, LABEL_COL
)
from src.cleaning import clean_dataframe


def main():
    print("=" * 80)
    print("EDEMS ML — Stage 1: Load and Clean Dataset Files")
    print("=" * 80)
    print(f"Source Directory: {RAW_DATA_DIR}")
    print(f"Target Directory: {DATA_DIR}\n")

    overall_stats = []
    overall_labels = {}

    start_total_time = time.time()

    for day_tag, filename in DAY_FILES.items():
        file_path = RAW_DATA_DIR / filename
        if not file_path.exists():
            print(f"[-] WARNING: File {filename} not found, skipping...")
            continue

        print(f"\nProcessing [{day_tag}] {filename}...")
        t0 = time.time()

        # Load raw data
        raw_df = pd.read_parquet(file_path)
        initial_len = len(raw_df)

        # Inspect pre-cleaning labels
        if LABEL_COL in raw_df.columns:
            raw_labels = raw_df[LABEL_COL].astype(str).str.strip().value_counts().to_dict()
        else:
            raw_labels = {"Unknown": initial_len}

        # Apply cleaning
        cleaned_df, stats = clean_dataframe(raw_df)
        elapsed = time.time() - t0

        # Output cleaned Parquet file to data/
        out_filename = f"{day_tag}_cleaned.parquet"
        out_path = DATA_DIR / out_filename
        cleaned_df.to_parquet(out_path, index=False, engine="pyarrow")

        # Record statistics
        stats["day_tag"] = day_tag
        stats["filename"] = filename
        stats["elapsed_s"] = round(elapsed, 2)
        overall_stats.append(stats)

        # Record post-cleaning label counts
        post_labels = cleaned_df[LABEL_COL].value_counts().to_dict()
        for lbl, count in post_labels.items():
            overall_labels[lbl] = overall_labels.get(lbl, 0) + count

        # Print per-file report
        print(f"  Rows: {initial_len:,} -> {stats['final_rows']:,} "
              f"(-{stats['total_dropped']:,} | {stats['retention_rate_pct']}% retained) "
              f"in {elapsed:.1f}s")
        print(f"  Dropped: [Headers: {stats.get('header_rows_dropped', 0)}] "
              f"[Inf: {stats.get('inf_rows_dropped', 0)}] "
              f"[NaN: {stats.get('nan_rows_dropped', 0)}] "
              f"[Negatives: {stats.get('negative_phys_rows_dropped', 0)}] "
              f"[Duplicates: {stats.get('duplicate_rows_dropped', 0)}]")
        print("  Label Distribution:")
        for lbl, count in post_labels.items():
            pct = (count / stats['final_rows']) * 100 if stats['final_rows'] > 0 else 0
            print(f"    - {lbl:25s}: {count:8,d} ({pct:6.2f}%)")

    # -------------------------------------------------------------------------
    # Overall Dataset Summary Table
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - start_total_time
    total_init = sum(s["initial_rows"] for s in overall_stats)
    total_final = sum(s["final_rows"] for s in overall_stats)
    total_dropped = sum(s["total_dropped"] for s in overall_stats)
    overall_retention = (total_final / total_init) * 100 if total_init > 0 else 0.0

    print("\n" + "=" * 80)
    print("STAGE 1 CLEANING SUMMARY ACROSS ALL DAYS")
    print("=" * 80)
    print(f"Total Initial Rows: {total_init:,}")
    print(f"Total Dropped Rows: {total_dropped:,}")
    print(f"Total Cleaned Rows: {total_final:,}")
    print(f"Overall Retention:  {overall_retention:.2f}%")
    print(f"Total Time Taken:   {total_elapsed:.1f} seconds\n")

    print("Aggregate Label Distribution:")
    for lbl, count in sorted(overall_labels.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_final) * 100 if total_final > 0 else 0
        print(f"  {lbl:30s}: {count:9,d}  ({pct:5.2f}%)")

    print("\nCleaned parquet files successfully saved in ml/data/")
    print("=" * 80)


if __name__ == "__main__":
    main()
