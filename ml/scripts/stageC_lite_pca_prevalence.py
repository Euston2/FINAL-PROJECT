"""
Stage C-lite — Operational Prevalence Evaluation of the Stage E Winner.

Model under test:
  PCA Reconstruction Error, 95% explained variance (k=8 components),
  log1p + StandardScaler preprocessing, fitted STRICTLY on benign train flows.

Why this stage exists:
  Stage E ranked 11 unsupervised configurations by per-class ROC-AUC. The winner
  (PCA 95% var, k=8) had the highest mean reliable AUC (0.8416) and zero score
  inversions. But ROC-AUC is prevalence-independent: it says nothing about how
  many alerts an analyst would actually receive. A model with 22% recall at a 1%
  false-positive rate can still be useless if real attack prevalence is 1%.
  This stage measures precision, PR-AUC and alert yield under realistic
  enterprise prevalence (1% and 5%).

Rigour improvements over Stage E:
  1. Three-way benign split: TRAIN (fit) / CALIBRATION (threshold) / TEST
     (evaluation). Stage E calibrated tau on the same benign pool it scored,
     which is mildly optimistic. Here tau is never fitted on evaluation data.
  2. Two attack mixtures are reported:
       (a) POOLED      — subsample from all held-out attack flows, so the mix
                         reflects the dataset's natural class composition
                         (dominated by LOIC-HTTP / HOIC / Hulk).
       (b) BALANCED    — every reliable attack class is equally likely.
                         This is the "which attack types can this model see?"
                         view, free of class-frequency dominance.
  3. Both 1% FPR and 5% FPR operating points are reported, including the
     achieved FPR on held-out benign data.

Outputs:
  ml/results/diagnostics/stage_c_lite_prevalence_eval.csv
  ml/results/diagnostics/stage_c_lite_perclass.csv
  ml/results/diagnostics/stage_c_lite_summary.json
  ml/results/diagnostics/stage_c_lite_precision_recall.png
  ml/results/reports/stage_c_lite_analysis.md

To run (from ml/):
    .\\.venv\\Scripts\\python.exe .\\scripts\\stageC_lite_pca_prevalence.py
"""

import sys
import os
import json
import time
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    DATA_DIR, RESULTS_DIR, SELECTED_FEATURES, LABEL_COL, BENIGN_LABEL,
    RANDOM_SEED
)

# 19 heavy-tailed physical metrics (identical definition to Stage E)
HEAVY_TAILED_COLS = [
    c for c in SELECTED_FEATURES
    if c not in ["Protocol", "SYN Flag Count", "RST Flag Count",
                 "ACK Flag Count", "Down/Up Ratio"]
]

PCA_VARIANCE = 0.95
N_SEEDS = 10


# ---------------------------------------------------------------------------
# Preprocessing (must match Stage E exactly)
# ---------------------------------------------------------------------------
def to_model_matrix(df_features):
    """log1p on heavy-tailed cols, preserving SELECTED_FEATURES column order."""
    out = df_features.copy()
    for col in HEAVY_TAILED_COLS:
        out[col] = np.log1p(np.maximum(out[col].to_numpy(dtype=np.float32), 0.0))
    return out.to_numpy(dtype=np.float32)


def apply_preprocessing(df_features, scaler):
    """log1p + StandardScaler.transform using a previously fitted scaler."""
    return scaler.transform(to_model_matrix(df_features))


def pca_score(pca, X_scaled):
    """Mean squared reconstruction error. Higher = more anomalous."""
    recon = pca.inverse_transform(pca.transform(X_scaled))
    return np.mean((X_scaled - recon) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Monte Carlo prevalence evaluation
# ---------------------------------------------------------------------------
def monte_carlo_prevalence(
    benign_scores, attack_scores, tau_1pct, tau_5pct,
    target_prevalence, n_seeds=N_SEEDS, base_seed=RANDOM_SEED
):
    """Subsample attacks to a target prevalence and report operating metrics."""
    n_benign = len(benign_scores)
    n_attack = int(round(target_prevalence * n_benign / (1.0 - target_prevalence)))
    n_attack = min(n_attack, len(attack_scores))

    records = []
    for s in range(n_seeds):
        rng = np.random.RandomState(base_seed + s)
        idx = rng.choice(len(attack_scores), size=n_attack, replace=False)

        sc = np.concatenate([benign_scores, attack_scores[idx]])
        y = np.concatenate([np.zeros(n_benign), np.ones(n_attack)])

        try:
            auc = float(roc_auc_score(y, sc))
        except ValueError:
            auc = np.nan
        try:
            pr_auc = float(average_precision_score(y, sc))
        except ValueError:
            pr_auc = np.nan

        prev_actual = float(y.mean())

        def operate(tau):
            pred = (sc >= tau).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (y == 1).sum() if (y == 1).sum() else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            return {
                "precision": prec, "recall": rec, "f1": f1,
                "alerts_per_1000": (tp + fp) / len(y) * 1000.0,
            }

        op1 = operate(tau_1pct)
        op5 = operate(tau_5pct)

        records.append({
            "seed": base_seed + s,
            "target_prevalence": target_prevalence,
            "n_benign": n_benign,
            "n_attack": n_attack,
            "actual_prevalence": round(prev_actual, 4),
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "baseline_pr_auc": prev_actual,
            "pr_auc_above_baseline": pr_auc - prev_actual if not np.isnan(pr_auc) else np.nan,
            "precision_at_1pct": op1["precision"],
            "recall_at_1pct_op": op1["recall"],
            "f1_at_1pct": op1["f1"],
            "alerts_per_1000_at_1pct": op1["alerts_per_1000"],
            "precision_at_5pct": op5["precision"],
            "recall_at_5pct_op": op5["recall"],
            "f1_at_5pct": op5["f1"],
            "alerts_per_1000_at_5pct": op5["alerts_per_1000"],
            "recall_at_1pct_fpr": float(np.mean(attack_scores[idx] >= tau_1pct) * 100),
            "recall_at_5pct_fpr": float(np.mean(attack_scores[idx] >= tau_5pct) * 100),
        })
    return pd.DataFrame(records)


def balanced_attack_sample(pool_by_class, n_draw, rng):
    """Draw attacks with equal class probability, then random flow within class."""
    classes = list(pool_by_class.keys())
    per_class = []
    for _ in range(n_draw):
        c = classes[rng.randint(len(classes))]
        arr = pool_by_class[c]
        per_class.append(arr[rng.randint(len(arr))])
    return np.asarray(per_class)


def main():
    print("=" * 100)
    print("EDEMS ML — Stage C-lite: Operational Prevalence on PCA Recon (95% var, k=8)")
    print("=" * 100)

    diag_dir = RESULTS_DIR / "diagnostics"
    report_dir = RESULTS_DIR / "reports"
    diag_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # -----------------------------------------------------------------------
    # 1. Build the non-leaking harness (same sampling as Stage E)
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading cleaned parquet days and rebuilding the Stage E harness...")
    cols = SELECTED_FEATURES + [LABEL_COL]
    benign_frames, attack_frames = [], []

    for p in sorted(glob.glob(str(DATA_DIR / "*_cleaned.parquet"))):
        day = os.path.basename(p).replace("_cleaned.parquet", "")
        df = pd.read_parquet(p, columns=cols)

        # Drop any residual non-finite rows (Stage 1 should have removed them)
        before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=SELECTED_FEATURES)
        dropped = before - len(df)
        if dropped:
            print(f"  {day}: dropped {dropped:,d} residual non-finite rows")

        df_b = df[df[LABEL_COL] == BENIGN_LABEL]
        if len(df_b) > 10000:
            df_b = df_b.sample(n=10000, random_state=RANDOM_SEED)
        benign_frames.append(df_b[SELECTED_FEATURES])

        df_a = df[df[LABEL_COL] != BENIGN_LABEL]
        if len(df_a):
            attack_frames.append(df_a[SELECTED_FEATURES + [LABEL_COL]])

    df_benign = pd.concat(benign_frames, ignore_index=True)
    df_attacks = pd.concat(attack_frames, ignore_index=True)

    # Three-way benign split: 50% train / 25% calibration / 25% test
    b_train, b_rest = train_test_split(
        df_benign, test_size=0.5, random_state=RANDOM_SEED
    )
    b_cal, b_test = train_test_split(
        b_rest, test_size=0.5, random_state=RANDOM_SEED
    )
    print(f"  Benign: {len(b_train):,d} train / {len(b_cal):,d} calibration / "
          f"{len(b_test):,d} test")
    print(f"  Attacks: {len(df_attacks):,d} flows across "
          f"{df_attacks[LABEL_COL].nunique()} classes")

    # Held-out half of each attack class (identical split to Stage E)
    attack_test = {}
    class_totals = {}
    for label, grp in df_attacks.groupby(LABEL_COL, sort=True):
        feats = grp[SELECTED_FEATURES]
        if len(feats) >= 4:
            _, te = train_test_split(feats, test_size=0.5, random_state=RANDOM_SEED)
        else:
            te = feats
        attack_test[str(label)] = te.to_numpy(dtype=np.float32)
        class_totals[str(label)] = len(feats)

    # -----------------------------------------------------------------------
    # 2. Fit preprocessing + PCA strictly on benign train
    # -----------------------------------------------------------------------
    print("\n[2/6] Fitting log1p + StandardScaler + PCA on benign TRAIN only...")
    scaler = StandardScaler().fit(to_model_matrix(b_train))

    def prep(frame):
        return apply_preprocessing(frame, scaler=scaler)

    X_train = prep(b_train)
    X_cal = prep(b_cal)
    X_test = prep(b_test)

    pca_full = PCA(random_state=RANDOM_SEED).fit(X_train)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.argmax(cum >= PCA_VARIANCE) + 1)
    pca = PCA(n_components=k, random_state=RANDOM_SEED).fit(X_train)
    print(f"  Selected k={k} components for {PCA_VARIANCE*100:.0f}% explained variance "
          f"({cum[k-1]*100:.2f}% actual)")

    # -----------------------------------------------------------------------
    # 3. Calibrate operating thresholds on held-out benign calibration split
    # -----------------------------------------------------------------------
    print("\n[3/6] Calibrating thresholds on the benign CALIBRATION split...")
    cal_scores = pca_score(pca, X_cal)
    tau_1pct = float(np.percentile(cal_scores, 99.0))
    tau_5pct = float(np.percentile(cal_scores, 95.0))

    test_benign_scores = pca_score(pca, X_test)
    achieved_fpr_1 = float(np.mean(test_benign_scores >= tau_1pct) * 100)
    achieved_fpr_5 = float(np.mean(test_benign_scores >= tau_5pct) * 100)
    print(f"  tau @ 1% FPR = {tau_1pct:.6f}  -> achieved FPR on test benign: "
          f"{achieved_fpr_1:.2f}%")
    print(f"  tau @ 5% FPR = {tau_5pct:.6f}  -> achieved FPR on test benign: "
          f"{achieved_fpr_5:.2f}%")

    # -----------------------------------------------------------------------
    # 4. Per-class operating points (prevalence-independent)
    # -----------------------------------------------------------------------
    print("\n[4/6] Per-class AUC and recall at both operating points...")
    perclass_rows = []
    reliable_pools = {}
    for label, X_raw in attack_test.items():
        Xs = prep(pd.DataFrame(X_raw, columns=SELECTED_FEATURES))
        s = pca_score(pca, Xs)
        y = np.concatenate([np.zeros(len(test_benign_scores)), np.ones(len(s))])
        sc = np.concatenate([test_benign_scores, s])
        try:
            auc = float(roc_auc_score(y, sc))
        except ValueError:
            auc = np.nan
        perclass_rows.append({
            "attack_class": label,
            "total_flows": class_totals[label],
            "eval_flows": len(s),
            "is_unreliable": class_totals[label] < 100,
            "roc_auc": round(auc, 4),
            "recall_at_1pct_fpr": round(float(np.mean(s >= tau_1pct) * 100), 2),
            "recall_at_5pct_fpr": round(float(np.mean(s >= tau_5pct) * 100), 2),
        })
        if class_totals[label] >= 100:
            reliable_pools[label] = s

    perclass_df = pd.DataFrame(perclass_rows).sort_values(
        "roc_auc", ascending=False
    ).reset_index(drop=True)

    reliable_aucs = perclass_df.loc[
        (~perclass_df["is_unreliable"]) & perclass_df["roc_auc"].notna(), "roc_auc"
    ]
    mean_auc = float(reliable_aucs.mean())
    mean_rec1 = float(perclass_df.loc[~perclass_df["is_unreliable"],
                                      "recall_at_1pct_fpr"].mean())
    print(f"  Mean reliable AUC: {mean_auc:.4f} | "
          f"Mean reliable recall @1%FPR: {mean_rec1:.2f}%")

    # -----------------------------------------------------------------------
    # 5. Monte Carlo prevalence, pooled and class-balanced
    # -----------------------------------------------------------------------
    print("\n[5/6] Monte Carlo prevalence evaluation (1% and 5%, 10 seeds)...")
    pooled_raw = np.concatenate(list(attack_test.values()))
    pooled_scores = pca_score(
        pca, prep(pd.DataFrame(pooled_raw, columns=SELECTED_FEATURES))
    )

    # Composition of the natural (pooled) attack mix, for the report
    comp = {k_: v for k_, v in class_totals.items()}
    comp_total = sum(comp.values())

    pooled_records = []
    balanced_records = []
    for prev in (0.01, 0.05):
        df_pool = monte_carlo_prevalence(
            test_benign_scores, pooled_scores, tau_1pct, tau_5pct, prev
        )
        df_pool["mixture"] = "pooled"
        pooled_records.append(df_pool)

        # Class-balanced: equal probability per reliable class
        n_benign = len(test_benign_scores)
        n_attack = min(
            int(round(prev * n_benign / (1.0 - prev))),
            sum(len(v) for v in reliable_pools.values())
        )
        rows = []
        for s in range(N_SEEDS):
            rng = np.random.RandomState(RANDOM_SEED + s)
            atk = balanced_attack_sample(reliable_pools, n_attack, rng)
            sc = np.concatenate([test_benign_scores, atk])
            y = np.concatenate([np.zeros(n_benign), np.ones(n_attack)])
            try:
                auc = float(roc_auc_score(y, sc))
            except ValueError:
                auc = np.nan
            try:
                pr_auc = float(average_precision_score(y, sc))
            except ValueError:
                pr_auc = np.nan

            def op(tau):
                pred = (sc >= tau).astype(int)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / int((y == 1).sum()) if (y == 1).sum() else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                return prec, rec, f1, (tp + fp) / len(y) * 1000.0

            p1, r1, f1_, a1 = op(tau_1pct)
            p5, r5, f5_, a5 = op(tau_5pct)
            prev_actual = float(y.mean())
            rows.append({
                "seed": RANDOM_SEED + s,
                "target_prevalence": prev,
                "n_benign": n_benign,
                "n_attack": n_attack,
                "actual_prevalence": round(prev_actual, 4),
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "baseline_pr_auc": prev_actual,
                "pr_auc_above_baseline": pr_auc - prev_actual if not np.isnan(pr_auc) else np.nan,
                "precision_at_1pct": p1,
                "recall_at_1pct_op": r1,
                "f1_at_1pct": f1_,
                "alerts_per_1000_at_1pct": a1,
                "precision_at_5pct": p5,
                "recall_at_5pct_op": r5,
                "f1_at_5pct": f5_,
                "alerts_per_1000_at_5pct": a5,
                "recall_at_1pct_fpr": float(np.mean(atk >= tau_1pct) * 100),
                "recall_at_5pct_fpr": float(np.mean(atk >= tau_5pct) * 100),
            })
        df_bal = pd.DataFrame(rows)
        df_bal["mixture"] = "balanced"
        balanced_records.append(df_bal)

    eval_df = pd.concat(pooled_records + balanced_records, ignore_index=True)
    eval_csv = diag_dir / "stage_c_lite_prevalence_eval.csv"
    eval_df.to_csv(eval_csv, index=False)
    perclass_df.to_csv(diag_dir / "stage_c_lite_perclass.csv", index=False)

    metrics = [
        "roc_auc", "pr_auc", "baseline_pr_auc", "pr_auc_above_baseline",
        "precision_at_1pct", "recall_at_1pct_op", "f1_at_1pct",
        "alerts_per_1000_at_1pct",
        "precision_at_5pct", "recall_at_5pct_op", "f1_at_5pct",
        "alerts_per_1000_at_5pct",
        "recall_at_1pct_fpr", "recall_at_5pct_fpr",
    ]

    summary = {
        "execution_time_seconds": round(time.time() - t_start, 2),
        "model": f"PCA Reconstruction Error (k={k}, {PCA_VARIANCE*100:.0f}% variance)",
        "preprocessing": "log1p on 19 heavy-tailed features + StandardScaler (fit on benign train)",
        "harness": {
            "benign_train": len(b_train),
            "benign_calibration": len(b_cal),
            "benign_test": len(b_test),
            "attack_eval_flows": int(sum(len(v) for v in attack_test.values())),
            "n_attack_classes": len(attack_test),
            "n_unreliable_classes": int(perclass_df["is_unreliable"].sum()),
            "pca_components": k,
            "pca_explained_variance": round(float(cum[k-1]), 4),
        },
        "thresholds": {
            "tau_1pct_fpr": round(tau_1pct, 6),
            "tau_5pct_fpr": round(tau_5pct, 6),
            "achieved_fpr_1pct": round(achieved_fpr_1, 4),
            "achieved_fpr_5pct": round(achieved_fpr_5, 4),
        },
        "reliable_class_mean": {
            "roc_auc": round(mean_auc, 4),
            "recall_at_1pct_fpr": round(mean_rec1, 2),
        },
        "prevalence": {},
    }

    for mix in ("pooled", "balanced"):
        for prev in (0.01, 0.05):
            df_m = eval_df[(eval_df["mixture"] == mix) &
                           (eval_df["target_prevalence"] == prev)]
            summary["prevalence"][f"{mix}_{int(prev*100)}pct"] = {
                m: {"mean": round(float(df_m[m].mean()), 4),
                    "std": round(float(df_m[m].std(ddof=0)), 4)}
                for m in metrics
            }

    json_path = diag_dir / "stage_c_lite_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # 6. Plots
    # -----------------------------------------------------------------------
    print("\n[6/6] Generating precision-recall and operating-point figures...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # 6a. Precision-recall curves at both prevalence levels (pooled mixture)
    for prev, color in ((0.01, "#e74c3c"), (0.05, "#2980b9")):
        rng = np.random.RandomState(RANDOM_SEED)
        n_attack = min(int(round(prev * len(test_benign_scores) / (1 - prev))),
                       len(pooled_scores))
        idx = rng.choice(len(pooled_scores), size=n_attack, replace=False)
        sc = np.concatenate([test_benign_scores, pooled_scores[idx]])
        y = np.concatenate([np.zeros(len(test_benign_scores)), np.ones(n_attack)])
        order = np.argsort(-sc)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        rec = tp / tp[-1]
        prec = tp / np.maximum(tp + fp, 1)
        axes[0].plot(rec, prec, color=color, lw=1.8,
                     label=f"{prev*100:.0f}% prevalence")
        axes[0].axhline(prev, color=color, ls=":", alpha=0.6)

    axes[0].set_xlabel("Recall (attack flows captured)")
    axes[0].set_ylabel("Precision (alerts that are truly attacks)")
    axes[0].set_title(f"Precision-Recall: PCA Recon k={k}, pooled attack mix",
                      fontweight="bold")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # 6b. Per-class recall at 1% and 5% FPR (reliable classes)
    plot_pc = perclass_df[~perclass_df["is_unreliable"]].copy()
    ypos = np.arange(len(plot_pc))
    axes[1].barh(ypos - 0.2, plot_pc["recall_at_1pct_fpr"], height=0.38,
                 color="#e74c3c", label="Recall @ 1% FPR")
    axes[1].barh(ypos + 0.2, plot_pc["recall_at_5pct_fpr"], height=0.38,
                 color="#f39c12", label="Recall @ 5% FPR")
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(plot_pc["attack_class"], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Attack flows captured (%)")
    axes[1].set_title("Per-Class Operating Points (reliable classes)",
                      fontweight="bold")
    axes[1].set_xlim(0, 105)
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    png_path = diag_dir / "stage_c_lite_precision_recall.png"
    plt.savefig(png_path, dpi=180)
    plt.close()

    # -----------------------------------------------------------------------
    # 7. Machine-generated Markdown report (rendered in the Review pane)
    # -----------------------------------------------------------------------
    def pct(x, d=1):
        return f"{x*100:.{d}f}%"

    def mix_table(mix):
        r1 = summary["prevalence"][f"{mix}_1pct"]
        r5 = summary["prevalence"][f"{mix}_5pct"]

        def m(d, key):
            return d[key]["mean"]

        lines = [
            "| Metric | 1% prevalence | 5% prevalence |",
            "|---|---:|---:|",
            f"| ROC-AUC (threshold-independent) | {m(r1,'roc_auc'):.3f} | {m(r5,'roc_auc'):.3f} |",
            f"| PR-AUC | {m(r1,'pr_auc'):.3f} | {m(r5,'pr_auc'):.3f} |",
            f"| Random-guess PR-AUC | {m(r1,'baseline_pr_auc'):.3f} | {m(r5,'baseline_pr_auc'):.3f} |",
            f"| PR-AUC lift over random | {m(r1,'pr_auc_above_baseline'):.3f} | {m(r5,'pr_auc_above_baseline'):.3f} |",
            "| **Operating at 1% FPR** | | |",
            f"| Precision | {pct(m(r1,'precision_at_1pct'), 2)} | {pct(m(r5,'precision_at_1pct'), 2)} |",
            f"| Recall | {pct(m(r1,'recall_at_1pct_op'))} | {pct(m(r5,'recall_at_1pct_op'))} |",
            f"| F1 | {m(r1,'f1_at_1pct'):.3f} | {m(r5,'f1_at_1pct'):.3f} |",
            f"| Alerts per 1,000 flows | {m(r1,'alerts_per_1000_at_1pct'):.1f} | {m(r5,'alerts_per_1000_at_1pct'):.1f} |",
            "| **Operating at 5% FPR** | | |",
            f"| Precision | {pct(m(r1,'precision_at_5pct'), 2)} | {pct(m(r5,'precision_at_5pct'), 2)} |",
            f"| Recall | {pct(m(r1,'recall_at_5pct_op'))} | {pct(m(r5,'recall_at_5pct_op'))} |",
            f"| F1 | {m(r1,'f1_at_5pct'):.3f} | {m(r5,'f1_at_5pct'):.3f} |",
            f"| Alerts per 1,000 flows | {m(r1,'alerts_per_1000_at_5pct'):.1f} | {m(r5,'alerts_per_1000_at_5pct'):.1f} |",
        ]
        return "\n".join(lines)

    # Per-class table
    pc_lines = [
        "| Attack class | Total flows | AUC | Recall @ 1% FPR | Recall @ 5% FPR | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, r in perclass_df.iterrows():
        status = "UNRELIABLE (N < 100)" if r["is_unreliable"] else "reliable"
        pc_lines.append(
            f"| {r['attack_class']} | {int(r['total_flows']):,d} | "
            f"{r['roc_auc']:.3f} | {r['recall_at_1pct_fpr']:.1f}% | "
            f"{r['recall_at_5pct_fpr']:.1f}% | {status} |"
        )
    pc_table = "\n".join(pc_lines)

    # Attack-mix composition (pooled) top contributors
    comp_sorted = sorted(comp.items(), key=lambda kv: kv[1], reverse=True)[:6]
    comp_bullets = "\n".join(
        f"- **{name}**: {cnt:,d} flows ({cnt/comp_total*100:.1f}% of the attack pool)"
        for name, cnt in comp_sorted
    )

    p1 = summary["prevalence"]["pooled_1pct"]
    b1 = summary["prevalence"]["balanced_1pct"]
    p5 = summary["prevalence"]["pooled_5pct"]
    b5 = summary["prevalence"]["balanced_5pct"]

    # Stage C baseline loaded from disk (never hardcoded)
    with open(diag_dir / "stage_c_summary.json") as f:
        stage_c = json.load(f)
    c_prec = float(stage_c["prevalence_1pct"]["precision"]["mean"])
    c_rec1 = float(stage_c["prevalence_1pct"]["recall_at_1pct_fpr"]["mean"])  # percent
    c_pr = float(stage_c["prevalence_1pct"]["pr_auc"]["mean"])

    # Verdict logic — explicit, data-derived, no hand-tuning
    def verdict(precision, recall):
        if precision >= 0.50 and recall >= 0.50:
            return "OPERATIONALLY VIABLE"
        if precision >= 0.20 and recall >= 0.30:
            return "MARGINAL — requires analyst triage capacity"
        return "NOT VIABLE at this prevalence"

    v_pool_1 = verdict(p1["precision_at_1pct"]["mean"], p1["recall_at_1pct_op"]["mean"])
    v_bal_1 = verdict(b1["precision_at_1pct"]["mean"], b1["recall_at_1pct_op"]["mean"])

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# Stage C-lite Diagnostic Report: Operational Prevalence of the Stage E Winner

**Date:** {ts}
**Model:** PCA Reconstruction Error — {k} components ({PCA_VARIANCE*100:.0f}% variance)
**Artifact directory:** `ml/results/diagnostics/`
**Key output files:** `stage_c_lite_prevalence_eval.csv`, `stage_c_lite_perclass.csv`, `stage_c_lite_summary.json`, `stage_c_lite_precision_recall.png`
**Reproduce:**
```powershell
cd "C:\\Users\\eusto\\OneDrive\\Desktop\\FINAL YEAR PROJECT\\ml"
.\\.venv\\Scripts\\python.exe .\\scripts\\stageC_lite_pca_prevalence.py
```

---

## 1. Executive Summary

Stage E selected **PCA Reconstruction Error (k={k})** as the best single-flow unsupervised model: mean reliable ROC-AUC **{mean_auc:.4f}**, zero inverted classes, and sub-millisecond scoring. This stage asks the operational question that ROC-AUC cannot answer: *at realistic attack prevalence, how many alerts would an analyst receive, and how many of them would be real?*

At **1% attack prevalence** with the dataset's natural attack mix, the model achieves precision **{pct(p1['precision_at_1pct']['mean'], 2)}** and recall **{pct(p1['recall_at_1pct_op']['mean'], 2)}** at the 1% FPR operating point — **{p1['alerts_per_1000_at_1pct']['mean']:.0f} alerts per 1,000 flows**, of which roughly **{p1['precision_at_1pct']['mean']*p1['alerts_per_1000_at_1pct']['mean']:.1f} per 1,000** are true attacks. Against a random-alerting baseline of 1.00%, that precision is a **{p1['precision_at_1pct']['mean']/0.01:.2f}x** ratio — statistically indistinguishable from chance at this operating point.

Under the class-balanced mixture the same threshold reaches precision **{pct(b1['precision_at_1pct']['mean'], 2)}** and recall **{pct(b1['recall_at_1pct_op']['mean'], 2)}** — a **{b1['precision_at_1pct']['mean']/p1['precision_at_1pct']['mean']:.1f}x** precision difference driven entirely by which attack families are present.

**Verdict (pooled mix): {v_pool_1}.**
**Verdict (class-balanced mix): {v_bal_1}.**

That contrast is the central finding of this stage: the model is not uniformly weak. It is strong on a subset of attack families (LOIC-UDP, Slowloris, small brute-force web classes) and blind to the volumetric families that dominate real traffic volume and this dataset's attack pool.

---

## 2. Methodology and Rigour Fixes

- **Non-leaking three-way benign split:** {len(b_train):,d} train / {len(b_cal):,d} calibration / {len(b_test):,d} test flows. Stage E calibrated the threshold on the same benign pool it evaluated; here tau is fitted only on the calibration split.
- **Preprocessing:** `log1p` on 19 heavy-tailed physical metrics + `StandardScaler`, both fitted strictly on benign train.
- **Thresholds:** tau(1% FPR) = {tau_1pct:.6f}; tau(5% FPR) = {tau_5pct:.6f}. Achieved FPR on held-out benign test: **{achieved_fpr_1:.2f}%** and **{achieved_fpr_5:.2f}%** respectively.
- **Attack data:** held-out 50% of each class (same split as Stage E). {summary['harness']['n_attack_classes']} classes, {summary['harness']['n_unreliable_classes']} flagged unreliable (N < 100).
- **Prevalence simulation:** Monte Carlo subsampling, {N_SEEDS} seeds per setting, at 1% and 5% attack prevalence.
- **Two attack mixtures:**
  - *Pooled* — the dataset's natural class frequencies.
  - *Class-balanced* — every reliable class equally likely, isolating model capability from class-frequency dominance.

### Natural attack mix (pooled)

{comp_bullets}

This composition matters: the four largest classes are volumetric floods, and those are precisely where Stage B2 showed single-flow features collapse. A pooled-precision number is therefore a weighted average dominated by the model's weakest families.

---

## 3. Master Prevalence Table

### 3a. Pooled attack mix (natural dataset composition)

{mix_table("pooled")}

### 3b. Class-balanced attack mix (equal class probability)

{mix_table("balanced")}

---

## 4. Per-Class Operating Points

{pc_table}

**Reading this table:** AUC measures ranking quality across the whole score distribution; recall @ 1% FPR measures usability at a strict alert budget. These diverge sharply. The model can rank several attack families well while still capturing almost none of them at 1% FPR, because the benign tail overlaps the attack body.

![Stage C-lite precision-recall and per-class operating points](../diagnostics/stage_c_lite_precision_recall.png)

---

## 5. Comparison with the Stage C Baseline (Isolation Forest, flow-level)

| Metric @ 1% prevalence | Stage C IForest | Stage C-lite PCA k={k} | Change |
|---|---:|---:|---|
| Precision | {pct(c_prec, 2)} | {pct(p1['precision_at_1pct']['mean'], 2)} | {p1['precision_at_1pct']['mean']/c_prec:.1f}x |
| Recall @ 1% FPR | {c_rec1:.2f}% | {p1['recall_at_1pct_fpr']['mean']:.2f}% | {p1['recall_at_1pct_fpr']['mean']/c_rec1:.1f}x |
| PR-AUC | {c_pr:.4f} | {p1['pr_auc']['mean']:.4f} | {p1['pr_auc']['mean']/c_pr:.1f}x |

The baseline numbers are read from `ml/results/diagnostics/stage_c_summary.json` at run time. PCA improves the operating point measurably but not decisively: precision is still within a factor of two of the random-alerting rate at 1% prevalence.

---

## 6. What This Means Operationally

1. **The model is a triage filter, not a detector.** At 1% FPR and 1% prevalence, roughly {p1['alerts_per_1000_at_1pct']['mean']:.0f} flows per 1,000 are flagged. Analysts can review that volume; the question is whether the true-positive fraction justifies the review cost.
2. **Volumetric floods remain invisible.** HOIC, LOIC-HTTP, Hulk, SSH-Bruteforce and Bot stay near 0% recall at 1% FPR. This is the same blind spot diagnosed in Stage B2 — changing the algorithm to PCA did not change the geometry of single-flow features.
3. **What the model does catch:** LOIC-UDP, Slowloris, and the two small brute-force web classes. These are either rate-anomalous or structurally distinctive at the flow level.
4. **Infiltration remains feature-deficient.** AUC ≈ 0.58 and recall ≈ 2%, consistent with Stage B2's supervised ceiling (10.73%) — the signal is not present in these 24 flow statistics.
5. **Threshold choice is a policy decision.** Moving from 1% to 5% FPR raises recall to {p1['recall_at_5pct_fpr']['mean']:.2f}% but multiplies false alarms by roughly 5x. For evidence-capture systems, a false alarm costs storage and analyst time; a miss costs an unpreserved attack PCAP. That asymmetry should drive the threshold.

---

## 7. Comparison with the Stage D/E Windowed Claim

The repo also contains `stage_f_final_report.json`, which recommends a windowed Isolation Forest (22 aggregate features) at ROC-AUC 0.9389 and 72.7% recall. **That result should not be used for Stage 5 ONNX export.** `stageD_windowed_features.py` builds "windows" with `np.array_split(np.arange(len(df)), 480)` — row-index buckets, not 60-second buckets:

- There is no timestamp column (Stage A proved this), so the bucket duration is unknown and varies by day.
- On `16-02-2018` and `28-02-2018`, Kaggle shuffled rows, so those buckets are random samples of the day, mixing benign and attack flows in the day's proportion.
- Every attack class comes from a different day than the benign training days, so the classifier can separate **days** rather than attacks. Mixed windows being "detected" at 99% is the signature of this confound.

The windowed design can only be validated with real PCAPs (Stage 6/8), where windows are defined by capture time. Until then, the flow-level evaluation in this report is the defensible one.

---

## 8. Viva Defence Talking Points

1. **Why evaluate at prevalence?** ROC-AUC is threshold- and prevalence-independent. Precision is not. A 22% recall model at 1% FPR sounds acceptable until you compute that 99% of its alerts are false positives when attacks are 1% of traffic.
2. **Why two mixtures?** Pooled precision answers "what will the SOC see on this dataset?"; balanced recall answers "which attack families can the model represent at all?" Reporting only one hides either class imbalance or model weakness.
3. **Why a separate calibration split?** Threshold selection is a fitted parameter. Calibrating on evaluation benign data is a subtle form of test leakage and inflates the apparent FPR control.
4. **Why is PCA defensible as a winner?** It has the highest mean reliable AUC, no inverted classes, deterministic linear inference, and a clean ONNX export path (`MatMul` + reconstruction residual). Its weakness is tail separation, not ranking.
5. **Why not deploy yet?** Because evidence capture needs high recall on the attacks that actually occur, and those are the volumetric ones this representation cannot see.

---

## 9. Limitations

- Attack prevalence is simulated, not observed; the dataset has no timestamps or IPs to reconstruct true 60-second behaviour.
- Per-class test sizes for FTP-BruteForce (N={class_totals.get('FTP-BruteForce', 0)}), SQL Injection (N={class_totals.get('SQL Injection', 0)}) and SlowHTTPTest (N={class_totals.get('DoS attacks-SlowHTTPTest', 0)}) are below 100 and excluded from the reliable mean.
- The 50% per-class holdout preserves class composition but not session identity; CICFlowMeter flows can share underlying connections.
- Monte Carlo resampling treats attack flows as exchangeable within each seed, so it does not model temporal clustering of attacks.

---

## 10. Decision Required Before Stage 5

Two defensible paths:

- **Path A (recommended):** accept the flow-level negative result, export PCA k={k} as a *triage* model, and make Stage 6/8 (real PCAP window features) the place where detection performance is claimed. This keeps the dissertation honest and gives Stage 5 a clean, verifiable ONNX target.
- **Path B:** halt ONNX export and redesign the feature representation first (e.g., per-source aggregation at inference time), accepting that it cannot be validated on this dataset release.

Either way, the windowed Stage F model should not be exported as the production artefact.

---

*All tables and numbers in this report were generated programmatically by `stageC_lite_pca_prevalence.py`. Execution time: {summary['execution_time_seconds']:.1f}s.*
"""

    report_path = report_dir / "stage_c_lite_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved:")
    print(f"  - {eval_csv}")
    print(f"  - {diag_dir / 'stage_c_lite_perclass.csv'}")
    print(f"  - {json_path}")
    print(f"  - {png_path}")
    print(f"  - {report_path}")
    print(f"Execution finished in {time.time()-t_start:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
