"""SHAP explanations for the LightGBM credit model."""

from __future__ import annotations

from typing import Any

import warnings

import numpy as np
import pandas as pd
import shap

from .features import pretty
from .models import prepare_lgbm_frame


def tree_explainer(model) -> shap.TreeExplainer:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier")
        return shap.TreeExplainer(model)


def shap_matrix(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    levels: dict | None = None,
) -> np.ndarray:
    framed = prepare_lgbm_frame(X, levels)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier")
        raw = explainer.shap_values(framed)
    if isinstance(raw, list):
        raw = raw[1] if len(raw) > 1 else raw[0]
    if hasattr(raw, "values"):
        raw = raw.values
        if raw.ndim == 3:
            raw = raw[:, :, 1]
    return np.asarray(raw)


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def applicant_drivers(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    top_k: int = 8,
    levels: dict | None = None,
) -> pd.DataFrame:
    values = shap_matrix(explainer, X_row, levels).reshape(-1)
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = expected[1] if np.size(expected) > 1 else float(np.ravel(expected)[0])
    contrib = pd.DataFrame(
        {
            "feature": X_row.columns,
            "feature_label": [pretty(c) for c in X_row.columns],
            "value": X_row.iloc[0].to_numpy(),
            "shap": values,
        }
    )
    contrib["direction"] = np.where(contrib["shap"] > 0, "raises risk", "lowers risk")
    contrib["abs_shap"] = contrib["shap"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).head(top_k)
    contrib.attrs["base_value"] = float(expected)
    return contrib.reset_index(drop=True)


def encode_for_json(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
