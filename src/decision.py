"""Map predicted PDs into A–E risk tiers and expected loss."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_TIER_EDGES, LGD, TIER_LABELS
from .features import exposure_at_default


def fit_tier_edges(pd_scores: np.ndarray) -> dict:
    """Development-sample quantile bands so A is safest and E is riskiest."""
    q = np.quantile(pd_scores, [0.20, 0.40, 0.60, 0.80])
    return {
        "A": (0.0, float(q[0])),
        "B": (float(q[0]), float(q[1])),
        "C": (float(q[1]), float(q[2])),
        "D": (float(q[2]), float(q[3])),
        "E": (float(q[3]), 1.01),
    }


def assign_tier(pd_scores, edges: dict | None = None) -> np.ndarray:
    edges = edges or DEFAULT_TIER_EDGES
    p = np.asarray(pd_scores, dtype=float)
    labels = np.full(p.shape, "E", dtype=object)
    for grade, (lo, hi) in edges.items():
        labels[(p >= lo) & (p < hi)] = grade
    labels[p >= edges["E"][0]] = "E"
    return labels


def expected_loss(pd_scores, limit_bal, bill_amt1, lgd: float = LGD) -> np.ndarray:
    ead = exposure_at_default(np.asarray(limit_bal, dtype=float), np.asarray(bill_amt1, dtype=float))
    return np.asarray(pd_scores, dtype=float) * lgd * ead


def tier_summary(y_true, pd_scores, limit_bal, bill_amt1, edges: dict, lgd: float = LGD) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pd_scores).astype(float)
    grades = assign_tier(p, edges)
    ead = exposure_at_default(np.asarray(limit_bal, dtype=float), np.asarray(bill_amt1, dtype=float))
    el = p * lgd * ead
    rows = []
    order = ["A", "B", "C", "D", "E"]
    for g in order:
        mask = grades == g
        n = int(mask.sum())
        lo, hi = edges[g]
        rows.append(
            {
                "tier": g,
                "label": TIER_LABELS[g],
                "pd_min": lo,
                "pd_max": hi,
                "n": n,
                "share": n / len(y) if len(y) else 0.0,
                "observed_default_rate": float(y[mask].mean()) if n else np.nan,
                "avg_pd": float(p[mask].mean()) if n else np.nan,
                "avg_ead": float(ead[mask].mean()) if n else np.nan,
                "total_el": float(el[mask].sum()) if n else 0.0,
                "el_per_account": float(el[mask].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)
