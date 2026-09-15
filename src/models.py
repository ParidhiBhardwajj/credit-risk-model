"""Logistic-regression baseline and LightGBM performance model."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .features import CATEGORICAL, NUMERIC


def build_logistic() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocess), ("clf", clf)])


def build_lightgbm(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        min_child_samples=60,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def categorical_levels(X: pd.DataFrame) -> dict[str, list[str]]:
    return {col: sorted(X[col].astype(str).unique().tolist()) for col in CATEGORICAL}


def prepare_lgbm_frame(X: pd.DataFrame, levels: dict[str, list[str]] | None = None) -> pd.DataFrame:
    out = X.copy()
    for col in CATEGORICAL:
        if levels and col in levels:
            out[col] = pd.Categorical(out[col].astype(str), categories=levels[col])
        else:
            out[col] = out[col].astype("category")
    return out


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict[str, Any]:
    logistic = build_logistic()
    logistic.fit(X_train, y_train)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale = neg / max(pos, 1.0)
    booster = build_lightgbm(scale)
    levels = categorical_levels(X_train)
    Xt = prepare_lgbm_frame(X_train, levels)
    Xv = prepare_lgbm_frame(X_valid, levels)
    booster.fit(
        Xt,
        y_train,
        eval_set=[(Xv, y_valid)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    logit_raw = logistic.predict_proba(X_valid)[:, 1]
    lgbm_raw = booster.predict_proba(Xv)[:, 1]
    calibrator_logit = _fit_platt(logit_raw, y_valid.to_numpy())
    calibrator_lgbm = _fit_platt(lgbm_raw, y_valid.to_numpy())

    return {
        "logistic": logistic,
        "lightgbm": booster,
        "levels": levels,
        "calibrator_logit": calibrator_logit,
        "calibrator_lgbm": calibrator_lgbm,
    }


def _fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: map ranking scores to PDs without flattening the distribution."""
    cal = LogisticRegression()
    cal.fit(np.asarray(scores).reshape(-1, 1), y)
    return cal


def _apply_calibrator(calibrator: LogisticRegression | None, scores: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return scores
    return np.clip(
        calibrator.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1],
        1e-6,
        1.0 - 1e-6,
    )


def predict_pd(
    models: dict[str, Any],
    X: pd.DataFrame,
    levels: dict[str, list[str]] | None = None,
) -> dict[str, np.ndarray]:
    levels = levels or models.get("levels")
    logit_raw = models["logistic"].predict_proba(X)[:, 1]
    lgb_raw = models["lightgbm"].predict_proba(prepare_lgbm_frame(X, levels))[:, 1]
    return {
        "logistic": _apply_calibrator(models.get("calibrator_logit"), logit_raw),
        "lightgbm": _apply_calibrator(models.get("calibrator_lgbm"), lgb_raw),
    }


def logistic_odds_table(pipeline: Pipeline) -> pd.DataFrame:
    """Odds ratios on the scaled/one-hot design matrix (numeric: per 1 SD)."""
    prep: ColumnTransformer = pipeline.named_steps["prep"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    names = prep.get_feature_names_out()
    coef = clf.coef_.ravel()
    odds = np.exp(coef)
    out = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coef,
            "odds_ratio": odds,
            "abs_log_odds": np.abs(coef),
        }
    )
    return out.sort_values("abs_log_odds", ascending=False).drop(columns=["abs_log_odds"])
