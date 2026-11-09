"""
EDEMS ML Component — Shared configuration and constants.

This module centralises settings that every script in the project imports,
so we never have inconsistent paths, random seeds, or feature lists
scattered across files.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. MATPLOTLIB CACHE FIX (must happen before matplotlib is imported)
# ---------------------------------------------------------------------------
# On some Windows setups the default matplotlib config directory
# (AppData\Local\matplotlib) is not writable.  We redirect it to a folder
# inside our project so matplotlib can cache fonts without errors.
_MPL_CACHE = Path(__file__).resolve().parent.parent / ".mpl_cache"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

# ---------------------------------------------------------------------------
# 1. RANDOM SEED
# ---------------------------------------------------------------------------
# Every algorithm that involves randomness (train/test split, Isolation
# Forest, etc.) must use this seed so that results are fully reproducible.
# In your viva, you can explain: "I fixed a global seed so that every run
# produces identical output, which is critical for reproducibility in
# scientific work."
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 2. PROJECT PATHS
# ---------------------------------------------------------------------------
# We derive all paths from the location of THIS file, so the project stays
# portable if you ever move the folder.
_ML_ROOT = Path(__file__).resolve().parent.parent   # …/ml/

DATA_DIR    = _ML_ROOT / "data"        # cleaned / intermediate data
MODELS_DIR  = _ML_ROOT / "models"      # trained model artefacts
RESULTS_DIR = _ML_ROOT / "results"     # evaluation plots and metrics
SCRIPTS_DIR = _ML_ROOT / "scripts"     # runnable stage scripts
SRC_DIR     = _ML_ROOT / "src"         # reusable library modules

# Path to the raw Kaggle Parquet files (your dataset)
RAW_DATA_DIR = Path(
    r"C:\Users\eusto\OneDrive\Desktop\FINAL YEAR PROJECT"
    r"\build\Kaggle Dataset\archive"
)

# ---------------------------------------------------------------------------
# 3. DATASET FILE MAPPING
# ---------------------------------------------------------------------------
# Maps a short "day tag" to the Parquet filename.  The day tag is used
# throughout the pipeline for logging, splitting, and result labelling.
# The tag format is the date from the filename for easy reference.
DAY_FILES = {
    "14-02-2018": "BruteForce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet",      # noqa: E501
    "15-02-2018": "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet",             # noqa: E501
    "16-02-2018": "DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet",               # noqa: E501
    "20-02-2018": "DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet",             # noqa: E501
    "21-02-2018": "DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet",           # noqa: E501
    "22-02-2018": "Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet",             # noqa: E501
    "23-02-2018": "Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet",               # noqa: E501
    "28-02-2018": "Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",          # noqa: E501
    "01-03-2018": "Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet",           # noqa: E501
    "02-03-2018": "Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet",             # noqa: E501
}

# Attack types present on each day (for reference during evaluation)
DAY_ATTACKS = {
    "14-02-2018": ["FTP-BruteForce", "SSH-Bruteforce"],
    "15-02-2018": ["DoS-GoldenEye", "DoS-Slowloris"],
    "16-02-2018": ["DoS-SlowHTTPTest", "DoS-Hulk"],
    "20-02-2018": ["DDoS-LOIC-HTTP"],
    "21-02-2018": ["DDoS-LOIC-UDP", "DDOS-HOIC"],
    "22-02-2018": ["BruteForce-Web", "BruteForce-XSS", "SQL-Injection"],
    "23-02-2018": ["BruteForce-Web", "BruteForce-XSS", "SQL-Injection"],
    "28-02-2018": ["Infiltration"],
    "01-03-2018": ["Infiltration"],
    "02-03-2018": ["Bot"],
}

# ---------------------------------------------------------------------------
# 4. LABEL COLUMN
# ---------------------------------------------------------------------------
# The dataset's target column.  "Benign" means normal traffic; everything
# else is an attack label.  We will use labels ONLY for evaluation, never
# for training (since we're training an unsupervised anomaly detector).
LABEL_COL = "Label"
BENIGN_LABEL = "Benign"

# ---------------------------------------------------------------------------
# 5. APPROVED REPRODUCIBLE FEATURES (Stage 2)
# ---------------------------------------------------------------------------
# 24 features computable identically from raw PCAPs using Scapy.
# Order is strictly maintained across training, ONNX export, and inference.
SELECTED_FEATURES = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "Down/Up Ratio",
]

# ---------------------------------------------------------------------------
# 6. DAY-BASED SPLIT STRATEGY (Stage 3 & 4)
# ---------------------------------------------------------------------------
# We split BY DAY (not randomly across rows) to prevent temporal session
# leakage. Normal network flows from one session share TCP sequence numbers,
# connection durations, and IP behavior; random row splitting causes severe
# evaluation overoptimism.
#
# - TRAIN_DAYS: Benign flows only, baseline behavior modeling.
# - VAL_DAYS: Held-out Benign flows for empirical false-alarm calibration.
# - TEST_DAYS: Complete days containing diverse real-world attacks.
TRAIN_DAYS = ["15-02-2018", "22-02-2018"]
VAL_DAYS   = ["23-02-2018"]
TEST_DAYS  = [
    "14-02-2018",  # FTP / SSH Brute Force
    "16-02-2018",  # DoS (Hulk, SlowHTTPTest)
    "20-02-2018",  # DDoS (LOIC-HTTP)
    "21-02-2018",  # DDoS (LOIC-UDP, HOIC)
    "28-02-2018",  # Infiltration 1 (Insider threat proxy)
    "01-03-2018",  # Infiltration 2 (Insider threat proxy)
    "02-03-2018",  # Botnet Ares (C2 / Exfiltration proxy)
]

# Target False Alarm Rate for threshold selection (1% = 0.01)
TARGET_FPR = 0.01

# IsolationForest hyperparameters
IFOREST_PARAMS = {
    "n_estimators": 100,
    "max_samples": 1024,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
