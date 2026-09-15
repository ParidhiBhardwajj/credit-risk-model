# Credit Risk Scoring

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An underwriting-style credit risk model: an interpretable logistic baseline, a LightGBM performance model, SHAP drivers for a single applicant, and A–E risk tiers tied to approval rate and expected loss.

**(holdout n = 6,000):** LightGBM **AUC 0.776 / KS 0.418** vs. logistic **0.752 / 0.392** (**+0.024 AUC, +0.026 KS**). Policy tiers are monotone: realized default is **5.3% in A vs. 61.2% in E**. Approving PD < 15% takes **45% of the book** at an **8.9% bad rate** (book default is 22.1%) and screens out **82% of defaults**.

## Why this dataset

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009) — 30,000 Taiwanese revolving-credit accounts, October 2005, **22.1%** default next month. It is the standard academic credit-default benchmark: six months of repayment status, bills, and payments, plus limit and demographics.

Home Credit Default Risk is richer (bureau + prior applications) but is a multi-table Kaggle dump. This project keeps the full underwriting workflow — features, baseline, KS, SHAP, cutoffs — on a dataset anyone can clone and rerun.

Income is not observed. Credit limit stands in for repayment capacity; **utilization** is revolving DTI (balance / limit) and **payment-to-limit** is the payment-to-income analog.

## What it does

1. **Explore** — default rates by utilization, payment-to-bill, delinquency, limit, education, and age.
2. **Engineer** — utilization (current / average / peak / trend), payment coverage, payment-to-limit, delinquent months, consecutive lates, over-limit, balance growth.
3. **Baseline** — logistic regression with class weights. Odds ratios are reported per 1 SD for numeric features (the model a risk committee still asks to see).
4. **Performance model** — LightGBM, Platt-calibrated on a validation fold so PDs can be used as probabilities, not just ranks.
5. **Evaluate** — AUC, **KS-statistic**, Gini (2·AUC − 1), PR-AUC. Accuracy is the wrong headline on a 22% default book.
6. **Decide** — A–E tiers from PD bands, plus an approval / bad-rate / expected-loss frontier at PD cutoffs. EL = PD × LGD × EAD, with LGD = 75% and EAD = drawn balance + 10% of undrawn limit.
7. **Explain** — SHAP values for an individual applicant (why this score), not only global importance.

## Holdout scorecard

Stratified 80/20 split. LightGBM uses 15% of the train fold for early stopping and Platt scaling.

| Model | AUC | KS | Gini | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression (baseline) | 0.752 | 0.392 | 0.504 | 0.504 |
| LightGBM (calibrated) | **0.776** | **0.418** | **0.552** | **0.550** |
| Lift | +0.024 | +0.026 | +0.048 | +0.046 |

## Risk tiers (test set)

| Tier | Policy band | Share of book | Avg PD | Realized default |
| --- | --- | ---: | ---: | ---: |
| A Super prime | PD < 8% | 16.4% | 7.2% | **5.3%** |
| B Prime | 8–15% | 28.9% | 11.4% | 11.0% |
| C Near prime | 15–25% | 27.4% | 18.5% | 18.0% |
| D Subprime | 25–40% | 11.6% | 32.2% | 30.4% |
| E High risk | PD ≥ 40% | 15.7% | 56.5% | **61.2%** |

Average PD tracks realized default in every grade — that is the check that the score is a probability, not just a rank.

## Underwriting frontier (approve if PD < cutoff)

| PD cutoff | Approval rate | Bad rate on approved | Defaults screened out |
| ---: | ---: | ---: | ---: |
| 8% | 16.4% | 5.3% | 96.1% |
| 10% | 27.6% | 6.4% | 92.1% |
| **15%** | **45.3%** | **8.9%** | **81.7%** |
| 20% | 65.9% | 11.7% | 65.3% |
| 30% | 77.1% | 13.1% | 54.5% |

A 15% cutoff more than halves the book default rate (22.1% → 8.9%) while still approving nearly half of applicants.

## Engineered features

| Feature | Credit reading |
| --- | --- |
| `utilization` / `utilization_avg` / `utilization_max` | Revolving DTI — balance as a share of limit. |
| `utilization_trend` / `balance_growth` | Is the customer digging in or working out? |
| `payment_ratio` / `payment_ratio_avg` | Payment-to-bill coverage (cash vs. last statement). |
| `payment_to_limit` | Payment-to-income analog when payroll is unobserved. |
| `delinquent_months` / `max_delinquency` / `consecutive_late` | 6-month credit history, not a single late flag. |
| `over_limit` / `zero_payment_months` | Stress flags. |

SHAP on the holdout sample ranks **delinquent months**, most recent repayment status (`pay_0`), last payment, current balance, and limit as the top drivers — the same variables a card underwriter would start with.

Logistic odds ratios on correlated utilization terms should be read as a group; LightGBM + SHAP is the decision model.

## Tech stack

- **Python 3.9+**
- **pandas / numpy** — cleaning and feature engineering
- **scikit-learn** — logistic pipeline, Platt calibration, metrics
- **LightGBM** — gradient-boosted performance model
- **SHAP** — TreeExplainer for applicant-level drivers
- **Plotly + Streamlit** — dashboard (same stack as the portfolio-risk project)

## Quick start

```bash
git clone https://github.com/ParidhiBhardwajj/credit-risk-model.git
cd credit-risk-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run scripts/app.py
```

Or from the project root:

```bash
./run.sh
```

Then open [http://localhost:8501](http://localhost:8501).

Trained models ship in `artifacts/models/`. To rebuild them from the UCI file:

```bash
python scripts/train.py
```

## Project structure

```
credit-risk-model/
├── data/raw/uci_credit_card_default.csv
├── data/processed/              # engineered features + scored holdout
├── artifacts/models/            # logistic, LightGBM, calibrators
├── artifacts/metrics/           # AUC/KS, odds ratios, cutoff & tier tables
├── scripts/
│   ├── app.py                   # Streamlit dashboard
│   └── train.py                 # train, evaluate, save artifacts
├── src/
│   ├── data.py                  # load / clean
│   ├── features.py              # utilization, PTI analog, delinquency
│   ├── models.py                # logistic + LightGBM + Platt
│   ├── evaluate.py              # AUC, KS, Gini, threshold table
│   ├── decision.py              # A–E tiers, expected loss
│   ├── explain.py               # SHAP
│   └── visualization.py         # Plotly charts
├── requirements.txt
├── run.sh
└── README.md
```

## Dashboard tabs

1. **Overview** — headline AUC/KS and the A–E grade table
2. **Risk drivers** — default rate by utilization, payment coverage, delinquency, limit, education, age
3. **Model comparison** — ROC, KS curve, logistic odds ratios, global SHAP
4. **Score an applicant** — form → PD, tier, expected loss, top SHAP drivers
5. **Decision framework** — cutoff slider with approval rate, approved bad rate, defaults screened out, and EL

## Method notes

- **Target:** `default payment next month` (1 = default). Prevalence 22.1%.
- **Split:** 80/20 stratified; 15% of train is validation for early stopping and Platt scaling. All reported metrics are holdout.
- **KS:** max |TPR − FPR| on the ROC, i.e. the largest gap between the default and non-default score CDFs. This is the statistic credit bureaus quote.
- **Gini:** 2·AUC − 1.
- **Calibration:** Platt scaling (`LogisticRegression` on the raw model score, fit on validation). Ranking metrics are almost unchanged; PDs become usable in EL = PD × LGD × EAD.
- **LGD:** 75% (unsecured revolving). **EAD:** current statement + 10% credit-conversion factor on undrawn limit.
- **Tiers:** fixed PD bands (not equal-count buckets), which is how a policy team would publish a grade.
- This is a demonstration scorecard on public research data, not a production underwriting system. It does not use bureau files, income verification, or through-the-cycle calibration.


## License

MIT. See [LICENSE](LICENSE).

Dataset: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480. Hosted by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

## Author

**Paridhi Bhardwaj** 
