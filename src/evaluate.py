"""AUC, KS, Gini, and threshold-level precision/recall for credit models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov–Smirnov statistic: max gap between default and non-default CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def gini_coefficient(auc: float) -> float:
    return float(2.0 * auc - 1.0)


def ranking_metrics(y_true, y_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    auc = float(roc_auc_score(y_true, y_score))
    ks = ks_statistic(y_true, y_score)
    return {
        "auc": auc,
        "ks": ks,
        "gini": gini_coefficient(auc),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "n": int(len(y_true)),
        "default_rate": float(y_true.mean()),
    }


def roc_points(y_true, y_score) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


def pr_points(y_true, y_score) -> pd.DataFrame:
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    # threshold array is one shorter than precision/recall
    thr = np.append(thr, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": thr})


def ks_curve(y_true, y_score) -> pd.DataFrame:
    """Population captured vs score rank — the chart credit teams actually use."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_score).astype(float)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    n = len(df)
    df["cum_bad"] = df["y"].cumsum() / max(df["y"].sum(), 1)
    df["cum_good"] = (1 - df["y"]).cumsum() / max((1 - df["y"]).sum(), 1)
    df["pct_pop"] = (np.arange(n) + 1) / n
    df["ks"] = (df["cum_bad"] - df["cum_good"]).abs()
    return df[["pct_pop", "cum_bad", "cum_good", "ks", "p"]]


def threshold_table(y_true, y_score, ead: np.ndarray, lgd: float, thresholds=None) -> pd.DataFrame:
    """Approval-rate / bad-rate / expected-loss tradeoff at PD cutoffs."""
    if thresholds is None:
        thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50]
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score).astype(float)
    ead = np.asarray(ead).astype(float)
    rows = []
    for t in thresholds:
        approved = p < t
        n_app = int(approved.sum())
        if n_app == 0:
            rows.append(
                {
                    "cutoff": t,
                    "approval_rate": 0.0,
                    "approved_bad_rate": np.nan,
                    "defaults_caught": float(y.mean()),
                    "precision_at_reject": np.nan,
                    "expected_loss_approved": 0.0,
                    "el_per_account": np.nan,
                }
            )
            continue
        bad_rate = float(y[approved].mean())
        caught = float(y[~approved].sum() / max(y.sum(), 1))
        rejected = ~approved
        precision_reject = float(y[rejected].mean()) if rejected.any() else np.nan
        el = float((p[approved] * lgd * ead[approved]).sum())
        rows.append(
            {
                "cutoff": t,
                "approval_rate": n_app / len(y),
                "approved_bad_rate": bad_rate,
                "defaults_caught": caught,
                "precision_at_reject": precision_reject,
                "expected_loss_approved": el,
                "el_per_account": el / n_app,
            }
        )
    return pd.DataFrame(rows)
