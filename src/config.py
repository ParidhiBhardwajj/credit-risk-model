"""Paths, underwriting assumptions, and score-band definitions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
METRICS_DIR = ARTIFACTS / "metrics"

RAW_CSV = DATA_RAW / "uci_credit_card_default.csv"
RAW_XLS = DATA_RAW / "uci_credit.xls"

TARGET_COL = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Unsecured revolving LGD — typical credit-card workout recovery is 20–35 cents.
LGD = 0.75

# Credit-conversion factor on undrawn limit (Basel-style card EAD).
CCF_UNDRAWN = 0.10

DISPLAY_NAMES = {
    "limit_bal": "Credit limit",
    "age": "Age",
    "sex": "Sex",
    "education": "Education",
    "marriage": "Marital status",
    "pay_0": "Most recent repayment status",
    "bill_amt1": "Current statement balance",
    "pay_amt1": "Last payment amount",
    "utilization": "Current credit utilization",
    "utilization_avg": "6-month average utilization",
    "utilization_max": "Peak 6-month utilization",
    "utilization_trend": "Utilization trend (recent − oldest)",
    "payment_ratio": "Payment-to-bill ratio",
    "payment_ratio_avg": "6-month average payment-to-bill",
    "payment_to_limit": "Payment-to-limit (PTI analog)",
    "delinquent_months": "Months delinquent (of 6)",
    "max_delinquency": "Worst delinquency status",
    "consecutive_late": "Consecutive late months",
    "revolving_months": "Months revolving (of 6)",
    "over_limit": "Over-limit flag",
    "balance_growth": "6-month balance growth",
    "zero_payment_months": "Months with zero payment",
}

# A is safest. Edges are overwritten at train time from development-sample PD quantiles.
DEFAULT_TIER_EDGES = {
    "A": (0.00, 0.08),
    "B": (0.08, 0.15),
    "C": (0.15, 0.25),
    "D": (0.25, 0.40),
    "E": (0.40, 1.01),
}

TIER_LABELS = {
    "A": "Super prime",
    "B": "Prime",
    "C": "Near prime",
    "D": "Subprime",
    "E": "High risk",
}

TIER_COLORS = {
    "A": "#148F77",
    "B": "#2E86AB",
    "C": "#C4A35A",
    "D": "#E67E22",
    "E": "#C0392B",
}

NAVY = "#1B4F72"
GOLD = "#C4A35A"
TEAL = "#148F77"
RED = "#C0392B"
SLATE = "#5D6D7E"
BLUE = "#2E86AB"
