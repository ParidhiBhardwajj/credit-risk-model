"""Credit-style feature engineering: utilization, payment coverage, delinquency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DISPLAY_NAMES

PAY_STATUS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]

CATEGORICAL = ["sex", "education", "marriage"]
NUMERIC = [
    "limit_bal",
    "age",
    "pay_0",
    "bill_amt1",
    "pay_amt1",
    "utilization",
    "utilization_avg",
    "utilization_max",
    "utilization_trend",
    "payment_ratio",
    "payment_ratio_avg",
    "payment_to_limit",
    "delinquent_months",
    "max_delinquency",
    "consecutive_late",
    "revolving_months",
    "over_limit",
    "balance_growth",
    "zero_payment_months",
]

MODEL_FEATURES = CATEGORICAL + NUMERIC
EPS = 1.0


def _ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    d = denom.abs().clip(lower=EPS)
    out = numer / d
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add revolving DTI, payment coverage, and 6-month credit-history features."""
    out = df.copy()
    limit = out["limit_bal"].clip(lower=EPS)

    util = pd.DataFrame({c: out[c] / limit for c in BILL_COLS})
    out["utilization"] = util["bill_amt1"].clip(-1, 5)
    out["utilization_avg"] = util.mean(axis=1).clip(-1, 5)
    out["utilization_max"] = util.max(axis=1).clip(-1, 5)
    out["utilization_trend"] = (util["bill_amt1"] - util["bill_amt6"]).clip(-3, 3)

    # Payment-to-bill: last cash payment vs the prior statement (true coverage ratio).
    out["payment_ratio"] = _ratio(out["pay_amt1"], out["bill_amt2"]).clip(0, 5)
    month_ratios = [_ratio(out[p], out[b]).clip(0, 5) for p, b in zip(PAYAMT_COLS, BILL_COLS)]
    out["payment_ratio_avg"] = pd.concat(month_ratios, axis=1).mean(axis=1)

    # Payment-to-limit stands in for PTI when income is not observed.
    out["payment_to_limit"] = (out["pay_amt1"] / limit).clip(0, 2)

    status = out[PAY_STATUS]
    out["delinquent_months"] = (status > 0).sum(axis=1).astype(int)
    out["max_delinquency"] = status.max(axis=1).astype(int)
    out["revolving_months"] = (status == 0).sum(axis=1).astype(int)
    out["zero_payment_months"] = (out[PAYAMT_COLS] <= 0).sum(axis=1).astype(int)
    out["over_limit"] = (out["bill_amt1"] > out["limit_bal"]).astype(int)
    out["balance_growth"] = _ratio(out["bill_amt1"], out["bill_amt6"]).clip(0, 10)

    consecutive = np.zeros(len(out), dtype=int)
    still = np.ones(len(out), dtype=bool)
    for col in PAY_STATUS:
        late = (out[col].to_numpy() > 0) & still
        consecutive += late.astype(int)
        still &= out[col].to_numpy() > 0
    out["consecutive_late"] = consecutive

    keep = MODEL_FEATURES + (["default"] if "default" in out.columns else [])
    return out[keep]


def exposure_at_default(limit_bal: np.ndarray, bill_amt1: np.ndarray, ccf: float = 0.10) -> np.ndarray:
    """Card EAD ≈ drawn balance + CCF × undrawn limit."""
    drawn = np.maximum(bill_amt1, 0.0)
    undrawn = np.maximum(limit_bal - drawn, 0.0)
    return drawn + ccf * undrawn


def applicant_frame(payload: dict) -> pd.DataFrame:
    """Build a one-row raw table from the Streamlit form, then engineer features."""
    row = {
        "limit_bal": float(payload["limit_bal"]),
        "sex": payload["sex"],
        "education": payload["education"],
        "marriage": payload["marriage"],
        "age": int(payload["age"]),
        "pay_0": int(payload["pay_0"]),
        "bill_amt1": float(payload["bill_amt1"]),
        "pay_amt1": float(payload["pay_amt1"]),
    }
    recent_status = int(payload.get("pay_0", 0))
    recent_bill = float(payload.get("bill_amt1", 0))
    recent_pay = float(payload.get("pay_amt1", 0))
    for i, col in enumerate(PAY_STATUS):
        row[col] = int(payload.get(col, recent_status if i == 0 else payload.get("prior_status", 0)))
    for i, col in enumerate(BILL_COLS):
        decay = 0.97 ** i
        row[col] = float(payload.get(col, recent_bill * decay))
    for i, col in enumerate(PAYAMT_COLS):
        row[col] = float(payload.get(col, recent_pay))
    return engineer(pd.DataFrame([row]))


def pretty(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())
