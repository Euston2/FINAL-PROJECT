"""
EDEMS ML Component — Data Cleaning Module.

This module encapsulates all cleaning operations for CIC-IDS2018 traffic flow data.
In network intrusion detection research, raw output from tools like CICFlowMeter
is known to suffer from specific artifacts:
  1. Header row duplication (when files are concatenated or restarted).
  2. Division-by-zero causing NaN / Infinity in rate calculations (e.g., flow
     duration of 0 microseconds with Flow Bytes/s = Bytes / 0).
  3. Negative timing values (Flow Duration, Flow IAT) caused by integer overflow
     or clock synchronization drift across captures.
  4. Exact duplicate flow records caused by packet capture replay or probe loops.
  5. Inconsistent datatypes across days (e.g., strings vs. numeric).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply defensible data cleaning transformations to a raw flow DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from CICFlowMeter output (CSV or Parquet).

    Returns
    -------
    cleaned_df : pd.DataFrame
        Sanitized DataFrame ready for feature selection and modeling.
    stats : dict
        Detailed metrics tracking rows dropped at each cleaning step for auditability.
    """
    initial_rows = len(df)
    stats: Dict[str, Any] = {"initial_rows": initial_rows}

    # -------------------------------------------------------------------------
    # Step 1: Strip whitespace from column names and string values
    # Rationale: Different versions of CICFlowMeter add leading/trailing spaces
    # (e.g., ' Label' vs 'Label', 'Flow Bytes/s ' vs 'Flow Bytes/s').
    # -------------------------------------------------------------------------
    df.columns = df.columns.str.strip()

    # -------------------------------------------------------------------------
    # Step 2: Remove repeated header rows
    # Rationale: If the capture process re-appended headers mid-file, column
    # names appear inside data cells (e.g., Label == 'Label').
    # -------------------------------------------------------------------------
    if "Label" in df.columns:
        header_mask = df["Label"].astype(str).str.strip().str.lower() == "label"
        n_headers = int(header_mask.sum())
        if n_headers > 0:
            df = df[~header_mask]
        stats["header_rows_dropped"] = n_headers
        # Standardize label strings (strip whitespace)
        df["Label"] = df["Label"].astype(str).str.strip()

    # -------------------------------------------------------------------------
    # Step 3: Handle Infinity and NaN in numeric columns
    # Rationale: If Flow Duration is 0, calculating rate metrics:
    #   Flow Bytes/s = (Total Bytes) / (Flow Duration)
    # results in IEEE 754 Infinity or NaN (0/0). Infinite rates cannot be
    # fed into machine learning algorithms without numerical instability.
    # We replace infinite values with NaN and drop corrupted records.
    # -------------------------------------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check if any columns that SHOULD be numeric are object type (e.g. string numbers)
    for col in df.columns:
        if col != "Label" and col not in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            num_cols.append(col)

    inf_mask = np.isinf(df[num_cols]).any(axis=1)
    n_inf = int(inf_mask.sum())
    stats["inf_rows_dropped"] = n_inf
    df = df[~inf_mask]

    nan_mask = df[num_cols].isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    stats["nan_rows_dropped"] = n_nan
    df = df[~nan_mask]

    # -------------------------------------------------------------------------
    # Step 4: Handle negative values in physical metrics
    # Rationale: Metrics such as Flow Duration, packet counts, byte counts,
    # and inter-arrival times (IAT) are physical quantities and cannot be
    # negative. Negative values in CICFlowMeter arise from 32-bit signed
    # integer overflows in microsecond counters or out-of-order packet timestamps.
    # Note: Window sizes (Init Fwd Win Bytes) can be -1 by design in CICFlowMeter
    # to signify "No SYN packet observed". We do NOT drop rows for -1 window sizes,
    # but strictly drop rows with negative duration, rates, or IAT.
    # -------------------------------------------------------------------------
    physically_non_negative_cols = [
        col for col in [
            "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
            "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Mean", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Mean", "Bwd IAT Max", "Bwd IAT Min",
            "Total Fwd Packets", "Total Backward Packets",
            "Fwd Packets Length Total", "Bwd Packets Length Total",
        ] if col in df.columns
    ]

    neg_mask = (df[physically_non_negative_cols] < 0).any(axis=1)
    n_neg = int(neg_mask.sum())
    stats["negative_phys_rows_dropped"] = n_neg
    df = df[~neg_mask]

    # -------------------------------------------------------------------------
    # Step 5: Remove duplicate flow records
    # Rationale: Exact duplicates can skew density estimation in unsupervised
    # algorithms (e.g. Isolation Forest) and inflate benign validation scores.
    # -------------------------------------------------------------------------
    n_before_dedup = len(df)
    df = df.drop_duplicates()
    n_duplicates = n_before_dedup - len(df)
    stats["duplicate_rows_dropped"] = n_duplicates

    # -------------------------------------------------------------------------
    # Step 6: Standardize data types for downstream efficiency
    # Convert numeric columns to float32 (saving 50% RAM compared to float64)
    # and label to category.
    # -------------------------------------------------------------------------
    for col in num_cols:
        df[col] = df[col].astype(np.float32)
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype("category")

    final_rows = len(df)
    stats["final_rows"] = final_rows
    stats["total_dropped"] = initial_rows - final_rows
    stats["retention_rate_pct"] = round((final_rows / initial_rows) * 100, 3) if initial_rows > 0 else 0.0

    return df, stats
