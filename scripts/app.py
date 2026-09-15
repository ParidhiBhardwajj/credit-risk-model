"""
Credit Risk Scoring Dashboard

Score a revolving-credit applicant, compare the logistic baseline to LightGBM,
and translate PDs into A–E risk tiers with expected-loss / approval tradeoffs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.config import LGD, METRICS_DIR, MODELS_DIR, TIER_COLORS, TIER_LABELS
from src.decision import assign_tier, expected_loss
from src.explain import applicant_drivers, tree_explainer
from src.features import MODEL_FEATURES, applicant_frame
from src.models import predict_pd
from src.visualization import (
    plot_default_by_bucket,
    plot_expected_loss,
    plot_ks,
    plot_odds,
    plot_pd_hist,
    plot_pr_tradeoff,
    plot_roc,
    plot_shap_applicant,
    plot_shap_global,
    plot_tier_bars,
)

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.35rem;
        font-weight: 700;
        color: #1B4F72;
        text-align: center;
        padding: 0.4rem 0 0.2rem 0;
    }
    .sub-header {
        text-align: center;
        color: #5D6D7E;
        margin-bottom: 1.2rem;
    }
    .tier-chip {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 0.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAY_LABELS = {
    -2: "No consumption (-2)",
    -1: "Paid in full (-1)",
    0: "Revolving (0)",
    1: "1 month late",
    2: "2 months late",
    3: "3 months late",
    4: "4 months late",
    5: "5+ months late",
}

PRESETS = {
    "Prime revolver": {
        "limit_bal": 250000,
        "age": 38,
        "sex": "Female",
        "education": "Graduate school",
        "marriage": "Married",
        "pay_0": 0,
        "prior_status": 0,
        "bill_amt1": 42000,
        "pay_amt1": 8000,
    },
    "Thin / young file": {
        "limit_bal": 50000,
        "age": 24,
        "sex": "Male",
        "education": "University",
        "marriage": "Single",
        "pay_0": 0,
        "prior_status": 0,
        "bill_amt1": 18000,
        "pay_amt1": 1500,
    },
    "Stressed / past due": {
        "limit_bal": 80000,
        "age": 41,
        "sex": "Male",
        "education": "High school",
        "marriage": "Married",
        "pay_0": 2,
        "prior_status": 2,
        "bill_amt1": 76000,
        "pay_amt1": 0,
    },
}


@st.cache_data
def load_metrics() -> dict:
    path = METRICS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / name)


@st.cache_data
def load_engineered() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "engineered.csv")


@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "scored_test.csv")


@st.cache_resource
def load_models():
    logistic = joblib.load(MODELS_DIR / "logistic.joblib")
    lightgbm = joblib.load(MODELS_DIR / "lightgbm.joblib")
    meta = joblib.load(MODELS_DIR / "meta.joblib")
    return {
        "logistic": logistic,
        "lightgbm": lightgbm,
        "levels": meta["levels"],
        "calibrator_logit": meta.get("calibrator_logit"),
        "calibrator_lgbm": meta.get("calibrator_lgbm"),
        "explainer": tree_explainer(lightgbm),
    }


def _need_train() -> bool:
    return not (MODELS_DIR / "lightgbm.joblib").exists() or not (METRICS_DIR / "metrics.json").exists()


def show_fig(fig) -> None:
    """Draw a Plotly figure without extra Streamlit kwargs.

    Streamlit 1.50's ``plotly_chart`` has no ``width=`` argument yet. Passing
    ``width="stretch"`` lands in ``**kwargs``, which Streamlit treats as a
    Plotly config option and shows as a deprecation overlay on the chart.
    """
    st.plotly_chart(fig)


def metric_pct(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    st.markdown('<div class="main-header">Credit Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">UCI Default of Credit Card Clients · logistic baseline · LightGBM · SHAP · A–E decisioning</div>',
        unsafe_allow_html=True,
    )

    if _need_train():
        st.error("Models not found. From the project root run:  `python scripts/train.py`")
        st.stop()

    metrics = load_metrics()
    models = load_models()
    engineered = load_engineered()
    scored = load_scored()
    odds = load_csv("odds_ratios.csv")
    cutoffs = load_csv("threshold_table.csv")
    tiers = load_csv("tier_table.csv")
    importance = load_csv("shap_importance.csv")
    edges = {k: tuple(v) for k, v in metrics["tier_edges"].items()}

    logit = metrics["logistic"]
    lgbm = metrics["lightgbm"]

    st.sidebar.header("Book snapshot")
    st.sidebar.metric("Applicants", f"{metrics['n_rows']:,}")
    st.sidebar.metric("Observed default rate", metric_pct(metrics["default_rate"]))
    st.sidebar.metric("LGD (assumed)", metric_pct(LGD))
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**LightGBM vs logistic (test)**  \n"
        f"AUC **{lgbm['auc']:.3f}** vs {logit['auc']:.3f}  \n"
        f"KS **{lgbm['ks']:.3f}** vs {logit['ks']:.3f}"
    )
    st.sidebar.caption(
        "Dataset: Yeh & Lien (2009), UCI Default of Credit Card Clients — "
        "30,000 Taiwanese revolving-credit accounts, October 2005."
    )

    tab_overview, tab_drivers, tab_models, tab_score, tab_decision = st.tabs(
        [
            "Overview",
            "Risk drivers",
            "Model comparison",
            "Score an applicant",
            "Decision framework",
        ]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test AUC (LightGBM)", f"{lgbm['auc']:.3f}", f"+{metrics['lift']['auc']:.3f} vs logistic")
        c2.metric("KS statistic", f"{lgbm['ks']:.3f}", f"+{metrics['lift']['ks']:.3f} vs logistic")
        c3.metric("Gini", f"{lgbm['gini']:.3f}")
        c4.metric("PR-AUC", f"{lgbm['pr_auc']:.3f}")
        st.markdown(
            "Accuracy is the wrong headline on a 22% default book. **AUC** ranks who is riskier; "
            "**KS** is the credit-bureau statistic — the maximum gap between the default and "
            "non-default score distributions. Gini = 2·AUC − 1."
        )
        left, right = st.columns(2)
        with left:
            show_fig(plot_tier_bars(tiers))
        with right:
            st.dataframe(
                tiers.assign(
                    share=lambda d: (d["share"] * 100).round(1).astype(str) + "%",
                    observed_default_rate=lambda d: (d["observed_default_rate"] * 100).round(1).astype(str) + "%",
                    avg_pd=lambda d: (d["avg_pd"] * 100).round(1).astype(str) + "%",
                    total_el=lambda d: d["total_el"].round(0),
                )[
                    [
                        "tier",
                        "label",
                        "share",
                        "observed_default_rate",
                        "avg_pd",
                        "total_el",
                    ]
                ].rename(
                    columns={
                        "observed_default_rate": "realized default",
                        "avg_pd": "avg predicted PD",
                        "total_el": "expected loss (NTD)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "Tiers A–E are **policy PD bands** (A < 8%, B 8–15%, C 15–25%, D 25–40%, E ≥ 40%), "
            "not equal-count buckets. A well-built scorecard is **monotone**: realized default "
            "rises with the grade, and average PD sits close to the observed rate. PDs are "
            "Platt-calibrated on the validation fold so they can be used in expected loss."
        )

    with tab_drivers:
        st.markdown(
            "These are the cuts a credit analyst would actually request: utilization (revolving DTI), "
            "payment coverage, and recent delinquency — not just age and limit."
        )
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            show_fig(
                plot_default_by_bucket(engineered, "utilization", "Default rate by current utilization")
            )
        with r1c2:
            show_fig(
                plot_default_by_bucket(engineered, "payment_ratio", "Default rate by payment-to-bill")
            )
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            show_fig(
                plot_default_by_bucket(
                    engineered, "delinquent_months", "Default rate by months delinquent (of last 6)", bins=None
                )
            )
        with r2c2:
            show_fig(
                plot_default_by_bucket(engineered, "limit_bal", "Default rate by credit limit (income-capacity proxy)")
            )
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            show_fig(
                plot_default_by_bucket(engineered, "education", "Default rate by education", bins=None)
            )
        with r3c2:
            show_fig(plot_default_by_bucket(engineered, "age", "Default rate by age"))
        st.markdown("**Engineered features (financial reasoning)**")
        st.markdown(
            """
            | Feature | Why it belongs on a card scorecard |
            | --- | --- |
            | `utilization` / `utilization_avg` | Revolving DTI — balance as a share of limit. High utilization is the classic early-stress signal. |
            | `payment_ratio` | Payment-to-bill. Paying the statement down is the opposite of minimum-pay distress. |
            | `payment_to_limit` | Payment-to-income analog when payroll is unobserved: cash paid vs. granted capacity. |
            | `delinquent_months` / `consecutive_late` | 6-month credit history, not a single late flag. |
            | `utilization_trend` / `balance_growth` | Is the customer digging in or working out? |
            """
        )

    with tab_models:
        roc = {
            "logistic": pd.read_csv(METRICS_DIR / "roc_logistic.csv"),
            "lightgbm": pd.read_csv(METRICS_DIR / "roc_lightgbm.csv"),
        }
        ks = {
            "logistic": pd.read_csv(METRICS_DIR / "ks_logistic.csv"),
            "lightgbm": pd.read_csv(METRICS_DIR / "ks_lightgbm.csv"),
        }
        m1, m2 = st.columns(2)
        with m1:
            show_fig(plot_roc(roc, {"logistic": logit["auc"], "lightgbm": lgbm["auc"]}))
        with m2:
            show_fig(plot_ks(ks, {"logistic": logit["ks"], "lightgbm": lgbm["ks"]}))

        compare = pd.DataFrame(
            [
                {"model": "Logistic regression (baseline)", **logit},
                {"model": "LightGBM (performance)", **lgbm},
            ]
        )
        show = compare[["model", "auc", "ks", "gini", "pr_auc"]].copy()
        for col in ["auc", "ks", "gini", "pr_auc"]:
            show[col] = show[col].map(lambda x: f"{x:.3f}")
        st.dataframe(show, width="stretch", hide_index=True)

        st.markdown(
            "The logistic model is the one a bank risk committee still asks to see: every coefficient "
            "is an odds ratio. Numeric features are standardized, so an odds ratio of 1.40 means "
            "**+1 standard deviation raises default odds 40%**."
        )
        o1, o2 = st.columns(2)
        with o1:
            show_fig(plot_odds(odds))
        with o2:
            show_fig(plot_shap_global(importance))

    with tab_score:
        st.markdown("Enter applicant characteristics. The performance model returns a PD, an A–E tier, and the SHAP drivers behind that score.")
        preset_name = st.selectbox("Load a profile", list(PRESETS.keys()))
        preset = PRESETS[preset_name]

        f1, f2, f3 = st.columns(3)
        with f1:
            limit_bal = st.number_input("Credit limit (NTD)", 10000, 1000000, int(preset["limit_bal"]), step=10000)
            age = st.slider("Age", 21, 75, int(preset["age"]))
            sex = st.selectbox("Sex", ["Female", "Male"], index=0 if preset["sex"] == "Female" else 1)
        with f2:
            education = st.selectbox(
                "Education",
                ["Graduate school", "University", "High school", "Other"],
                index=["Graduate school", "University", "High school", "Other"].index(preset["education"]),
            )
            marriage = st.selectbox(
                "Marital status",
                ["Married", "Single", "Other"],
                index=["Married", "Single", "Other"].index(preset["marriage"]),
            )
            pay_0 = st.selectbox(
                "Most recent repayment status",
                options=list(PAY_LABELS.keys()),
                index=list(PAY_LABELS.keys()).index(preset["pay_0"]),
                format_func=lambda x: PAY_LABELS[x],
            )
        with f3:
            bill_amt1 = st.number_input("Current statement balance", 0, 800000, int(preset["bill_amt1"]), step=1000)
            pay_amt1 = st.number_input("Last payment amount", 0, 500000, int(preset["pay_amt1"]), step=500)
            prior_status = st.selectbox(
                "Prior 5 months (if unknown, copy recent)",
                options=list(PAY_LABELS.keys()),
                index=list(PAY_LABELS.keys()).index(preset["prior_status"]),
                format_func=lambda x: PAY_LABELS[x],
            )

        payload = {
            "limit_bal": limit_bal,
            "age": age,
            "sex": sex,
            "education": education,
            "marriage": marriage,
            "pay_0": pay_0,
            "prior_status": prior_status,
            "bill_amt1": bill_amt1,
            "pay_amt1": pay_amt1,
        }
        X_row = applicant_frame(payload)[MODEL_FEATURES]
        pds = predict_pd(models, X_row, models.get("levels"))
        pd_hat = float(pds["lightgbm"][0])
        pd_logit = float(pds["logistic"][0])
        tier = str(assign_tier([pd_hat], edges)[0])
        el = float(expected_loss([pd_hat], [limit_bal], [bill_amt1])[0])
        util = float(X_row["utilization"].iloc[0])

        color = TIER_COLORS[tier]
        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(
            f'<div class="tier-chip" style="color:{color}">Tier {tier}</div>'
            f'<div style="color:#5D6D7E">{TIER_LABELS[tier]}</div>',
            unsafe_allow_html=True,
        )
        s2.metric("Predicted PD (LightGBM)", f"{pd_hat:.1%}")
        s3.metric("Logistic PD (baseline)", f"{pd_logit:.1%}")
        s4.metric("Expected loss", f"{el:,.0f} NTD")

        st.caption(
            f"Utilization {util:.0%} · payment-to-bill {float(X_row['payment_ratio'].iloc[0]):.0%} · "
            f"EAD uses drawn balance + 10% of undrawn limit; LGD = {LGD:.0%}."
        )

        explainer = models["explainer"]
        drivers = applicant_drivers(explainer, X_row, top_k=8, levels=models.get("levels"))
        show_fig(plot_shap_applicant(drivers))
        st.dataframe(
            drivers[["feature_label", "value", "shap", "direction"]].rename(
                columns={"feature_label": "driver", "value": "applicant value", "shap": "SHAP"}
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_decision:
        st.markdown(
            "A classifier becomes a **credit policy** when you pick a cutoff. Approve applicants "
            "below a PD threshold; everyone above it is referred or declined. Expected loss on the "
            "approved book is PD × LGD × EAD."
        )
        show_fig(plot_pr_tradeoff(cutoffs))
        d1, d2 = st.columns((2, 1))
        with d1:
            show_fig(plot_expected_loss(cutoffs))
        with d2:
            st.markdown("**Pick a cutoff**")
            options = [float(x) for x in cutoffs["cutoff"]]
            default = 0.15 if 0.15 in options else options[min(4, len(options) - 1)]
            cutoff = st.select_slider("Approve if PD <", options=options, value=default)
            row = cutoffs.loc[cutoffs["cutoff"].astype(float) == float(cutoff)].iloc[0]
            st.metric("Approval rate", f"{row['approval_rate']:.1%}")
            st.metric("Bad rate among approved", f"{row['approved_bad_rate']:.1%}")
            st.metric("Share of defaults screened out", f"{row['defaults_caught']:.1%}")
            st.metric("EL on approved book", f"{row['expected_loss_approved']:,.0f} NTD")

        st.dataframe(
            cutoffs.assign(
                approval_rate=lambda d: (d["approval_rate"] * 100).round(1).astype(str) + "%",
                approved_bad_rate=lambda d: (d["approved_bad_rate"] * 100).round(1).astype(str) + "%",
                defaults_caught=lambda d: (d["defaults_caught"] * 100).round(1).astype(str) + "%",
                expected_loss_approved=lambda d: d["expected_loss_approved"].round(0),
            ).rename(
                columns={
                    "cutoff": "PD cutoff",
                    "approval_rate": "approval rate",
                    "approved_bad_rate": "approved bad rate",
                    "defaults_caught": "defaults screened out",
                    "expected_loss_approved": "EL approved (NTD)",
                }
            )[
                [
                    "PD cutoff",
                    "approval rate",
                    "approved bad rate",
                    "defaults screened out",
                    "EL approved (NTD)",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Holding LGD at 75% (unsecured revolving). Tightening the cutoff (moving left) "
            "cuts expected loss and approval rate together — that is the underwriting frontier."
        )

        st.markdown("**Holdout PD distribution**")
        st.caption("Calibrated LightGBM PDs on the 20% test set, used to build the cutoff table above.")
        show_fig(plot_pd_hist(scored["pd_lightgbm"]))


if __name__ == "__main__":
    main()
