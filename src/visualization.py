"""Plotly charts for the credit-risk dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import BLUE, GOLD, NAVY, RED, SLATE, TEAL, TIER_COLORS
from .features import pretty


def _layout(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=NAVY), x=0.0, xanchor="left"),
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.0, font=dict(size=11)),
        margin=dict(l=50, r=24, t=56, b=80),
        font=dict(color="#1C2833"),
    )
    return fig


def plot_default_by_bucket(frame: pd.DataFrame, col: str, title: str, bins: int | None = 8) -> go.Figure:
    work = frame[[col, "default"]].copy()
    if bins and pd.api.types.is_numeric_dtype(work[col]) and work[col].nunique() > 12:
        work["bucket"] = pd.qcut(work[col], q=min(bins, work[col].nunique()), duplicates="drop")
        grouped = work.groupby("bucket", observed=True)["default"].agg(["mean", "count"])
        labels = [str(ix) for ix in grouped.index]
    else:
        grouped = work.groupby(col, observed=True)["default"].agg(["mean", "count"])
        labels = [str(ix) for ix in grouped.index]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=grouped["mean"] * 100,
            marker_color=NAVY,
            name="Default rate",
            hovertemplate="%{x}<br>Default rate: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(title="Default rate (%)")
    fig.update_xaxes(title=pretty(col), tickangle=-25)
    return _layout(fig, title)


def plot_roc(curves: dict[str, pd.DataFrame], aucs: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    colors = {"logistic": SLATE, "lightgbm": TEAL}
    names = {"logistic": "Logistic regression", "lightgbm": "LightGBM"}
    for key, df in curves.items():
        fig.add_trace(
            go.Scatter(
                x=df["fpr"],
                y=df["tpr"],
                mode="lines",
                name=f"{names.get(key, key)} (AUC {aucs[key]:.3f})",
                line=dict(color=colors.get(key, BLUE), width=2.4),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#B0B0B0", width=1, dash="dash"),
        )
    )
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate (default capture)")
    return _layout(fig, "ROC — ranking quality")


def plot_ks(ks_frames: dict[str, pd.DataFrame], ks_values: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    style = {
        "logistic": (SLATE, "Logistic"),
        "lightgbm": (TEAL, "LightGBM"),
    }
    for key, df in ks_frames.items():
        color, name = style.get(key, (BLUE, key))
        fig.add_trace(
            go.Scatter(
                x=df["pct_pop"] * 100,
                y=df["cum_bad"] * 100,
                mode="lines",
                name=f"{name} defaults captured (KS {ks_values[key]:.3f})",
                line=dict(color=color, width=2.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["pct_pop"] * 100,
                y=df["cum_good"] * 100,
                mode="lines",
                name=f"{name} goods captured",
                line=dict(color=color, width=1.4, dash="dot"),
                showlegend=False,
            )
        )
    fig.update_xaxes(title="Population ranked by risk (highest PD first), %")
    fig.update_yaxes(title="Cumulative capture (%)")
    return _layout(fig, "KS curve — separation of defaults vs. non-defaults")


def plot_pr_tradeoff(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=table["cutoff"],
            y=table["approval_rate"] * 100,
            name="Approval rate",
            mode="lines+markers",
            line=dict(color=TEAL, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["cutoff"],
            y=table["approved_bad_rate"] * 100,
            name="Bad rate on approved",
            mode="lines+markers",
            line=dict(color=RED, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["cutoff"],
            y=table["defaults_caught"] * 100,
            name="Defaults screened out",
            mode="lines+markers",
            line=dict(color=GOLD, width=2.2),
        )
    )
    fig.update_xaxes(title="Approve if predicted PD < cutoff")
    fig.update_yaxes(title="Percent")
    return _layout(fig, "Underwriting tradeoff at PD cutoffs")


def plot_expected_loss(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=table["cutoff"].astype(str),
            y=table["expected_loss_approved"],
            marker_color=NAVY,
            name="Expected loss (approved book)",
        )
    )
    fig.update_xaxes(title="PD cutoff")
    fig.update_yaxes(title="Expected loss (NTD)")
    return _layout(fig, "Expected loss on the approved book")


def plot_tier_bars(summary: pd.DataFrame) -> go.Figure:
    colors = [TIER_COLORS.get(t, NAVY) for t in summary["tier"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["tier"] + " — " + summary["label"],
            y=summary["observed_default_rate"] * 100,
            marker_color=colors,
            name="Observed default rate",
            hovertemplate="%{x}<br>Default rate %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["tier"] + " — " + summary["label"],
            y=summary["avg_pd"] * 100,
            name="Average predicted PD",
            mode="markers+lines",
            line=dict(color=SLATE, width=1.6, dash="dot"),
            marker=dict(size=9, color=SLATE),
        )
    )
    fig.update_yaxes(title="Rate (%)")
    return _layout(fig, "Risk tiers: predicted PD vs. realized default rate")


def plot_shap_global(importance: pd.DataFrame, top_k: int = 12) -> go.Figure:
    top = importance.head(top_k).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["mean_abs_shap"],
            y=[pretty(f) for f in top["feature"]],
            orientation="h",
            marker_color=NAVY,
        )
    )
    fig.update_xaxes(title="Mean |SHAP| (impact on log-odds of default)")
    return _layout(fig, "Which features drive risk scores")


def plot_shap_applicant(drivers: pd.DataFrame) -> go.Figure:
    work = drivers.sort_values("shap")
    colors = [RED if v > 0 else TEAL for v in work["shap"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=work["shap"],
            y=work["feature_label"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>SHAP %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#999", line_width=1)
    fig.update_xaxes(title="SHAP value (→ raises default risk)")
    return _layout(fig, "Why this applicant scored this way")


def plot_pd_hist(scores: pd.Series, title: str = "Holdout PD distribution") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores * 100,
            nbinsx=40,
            marker_color=NAVY,
            opacity=0.85,
            name="Applicants",
        )
    )
    fig.update_xaxes(title="Predicted PD (%)")
    fig.update_yaxes(title="Applicants")
    return _layout(fig, title)


def plot_odds(odds: pd.DataFrame, top_k: int = 12) -> go.Figure:
    top = odds.head(top_k).copy()
    top["label"] = (
        top["feature"]
        .str.replace("cat__", "")
        .str.replace("num__", "")
        .str.replace("_", " ")
    )
    top = top.iloc[::-1]
    colors = [RED if or_ > 1 else TEAL for or_ in top["odds_ratio"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["odds_ratio"],
            y=top["label"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=1, line_color="#999", line_width=1, line_dash="dash")
    fig.update_xaxes(title="Odds ratio (>1 raises default odds)", type="log")
    return _layout(fig, "Logistic baseline — odds ratios (numeric per 1 SD)")
