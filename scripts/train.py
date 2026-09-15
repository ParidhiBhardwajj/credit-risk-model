#!/usr/bin/env python3
"""Train logistic + LightGBM, evaluate AUC/KS, fit risk tiers, save artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TIER_EDGES,
    LGD,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import clean, load_raw
from src.decision import tier_summary
from src.evaluate import ks_curve, ranking_metrics, roc_points, threshold_table
from src.explain import global_importance, shap_matrix, tree_explainer
from src.features import MODEL_FEATURES, engineer, exposure_at_default
from src.models import fit_models, logistic_odds_table, predict_pd


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    cleaned = clean(raw)
    featured = engineer(cleaned)

    # Keep raw exposure fields for EAD / EL (not used as extra model inputs beyond engineer()).
    featured = featured.copy()
    featured["limit_bal_raw"] = cleaned.loc[featured.index, "limit_bal"].to_numpy()
    featured["bill_amt1_raw"] = cleaned.loc[featured.index, "bill_amt1"].to_numpy()

    X = featured[MODEL_FEATURES]
    y = featured["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    models = fit_models(X_tr, y_tr, X_val, y_val)
    pd_test = predict_pd(models, X_test)

    logit_metrics = ranking_metrics(y_test, pd_test["logistic"])
    lgbm_metrics = ranking_metrics(y_test, pd_test["lightgbm"])

    ead_test = exposure_at_default(
        X_test["limit_bal"].to_numpy(), X_test["bill_amt1"].to_numpy()
    )
    cutoffs = threshold_table(y_test, pd_test["lightgbm"], ead_test, LGD)
    cutoffs = cutoffs.dropna(subset=["approved_bad_rate"]).drop_duplicates(
        subset=["approval_rate"], keep="first"
    )

    edges = DEFAULT_TIER_EDGES
    tiers = tier_summary(
        y_test,
        pd_test["lightgbm"],
        X_test["limit_bal"],
        X_test["bill_amt1"],
        edges,
        LGD,
    )

    odds = logistic_odds_table(models["logistic"])
    explainer = tree_explainer(models["lightgbm"])
    # Background + SHAP on a 2,000-row test slice keeps the artifact light.
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    shap_vals = shap_matrix(explainer, shap_sample, models["levels"])
    importance = global_importance(shap_vals, list(X_test.columns))

    scored = X_test.copy()
    scored["default"] = y_test.to_numpy()
    scored["pd_logistic"] = pd_test["logistic"]
    scored["pd_lightgbm"] = pd_test["lightgbm"]
    scored["ead"] = ead_test
    scored["el"] = scored["pd_lightgbm"] * LGD * scored["ead"]

    featured.to_csv(DATA_PROCESSED / "engineered.csv", index=False)
    scored.to_csv(DATA_PROCESSED / "scored_test.csv", index=False)
    odds.to_csv(METRICS_DIR / "odds_ratios.csv", index=False)
    cutoffs.to_csv(METRICS_DIR / "threshold_table.csv", index=False)
    tiers.to_csv(METRICS_DIR / "tier_table.csv", index=False)
    importance.to_csv(METRICS_DIR / "shap_importance.csv", index=False)
    roc_points(y_test, pd_test["logistic"]).to_csv(METRICS_DIR / "roc_logistic.csv", index=False)
    roc_points(y_test, pd_test["lightgbm"]).to_csv(METRICS_DIR / "roc_lightgbm.csv", index=False)
    ks_curve(y_test, pd_test["logistic"]).to_csv(METRICS_DIR / "ks_logistic.csv", index=False)
    ks_curve(y_test, pd_test["lightgbm"]).to_csv(METRICS_DIR / "ks_lightgbm.csv", index=False)

    joblib.dump(models["logistic"], MODELS_DIR / "logistic.joblib")
    joblib.dump(models["lightgbm"], MODELS_DIR / "lightgbm.joblib")
    joblib.dump(
        {
            "levels": models["levels"],
            "features": MODEL_FEATURES,
            "calibrator_logit": models["calibrator_logit"],
            "calibrator_lgbm": models["calibrator_lgbm"],
        },
        MODELS_DIR / "meta.joblib",
    )

    headline = {
        "n_rows": int(len(featured)),
        "default_rate": float(y.mean()),
        "lgd": LGD,
        "logistic": logit_metrics,
        "lightgbm": lgbm_metrics,
        "lift": {
            "auc": lgbm_metrics["auc"] - logit_metrics["auc"],
            "ks": lgbm_metrics["ks"] - logit_metrics["ks"],
        },
        "tier_edges": {k: list(v) for k, v in edges.items()},
        "features": MODEL_FEATURES,
    }
    (METRICS_DIR / "metrics.json").write_text(json.dumps(headline, indent=2))

    print("=== Credit risk model ===")
    print(f"Rows: {headline['n_rows']:,}   default rate: {headline['default_rate']:.1%}")
    print(
        f"Logistic   AUC {logit_metrics['auc']:.3f}   KS {logit_metrics['ks']:.3f}   Gini {logit_metrics['gini']:.3f}"
    )
    print(
        f"LightGBM   AUC {lgbm_metrics['auc']:.3f}   KS {lgbm_metrics['ks']:.3f}   Gini {lgbm_metrics['gini']:.3f}"
    )
    print(f"Lift       AUC +{headline['lift']['auc']:.3f}   KS +{headline['lift']['ks']:.3f}")
    print("\nRisk tiers (test):")
    print(tiers[["tier", "label", "share", "observed_default_rate", "avg_pd"]].to_string(index=False))
    print(f"\nWrote models → {MODELS_DIR}")
    print(f"Wrote metrics → {METRICS_DIR}")


if __name__ == "__main__":
    main()
