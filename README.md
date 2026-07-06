# Customer Retention / Churn Prediction

A customer churn prediction system built on the IBM Telco Customer Churn dataset. The project covers the full workflow from raw data to a business-ready model: data cleaning, exploratory analysis, feature engineering, model comparison, threshold tuning, and final evaluation with concrete retention recommendations.

## Problem statement

Telecom companies lose significant revenue to customer churn. This project builds a model to flag customers who are at risk of leaving, so a business can proactively intervene (retention offers, support outreach, contract incentives) before they churn — rather than reacting after the fact.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset) — 7,043 customer records with 21 columns covering demographics, account details, subscribed services, and churn status (~27% churn rate).

## Project structure

```
customer-retention-prediction/
├── data/
│   ├── raw/                       # Original dataset
│   │   └── Telco-Customer-Churn.csv
│   └── processed/                 # Cleaned dataset + train/test splits (splits are gitignored, regenerate via notebook 03)
│       └── telco_churn_clean.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Load, inspect, clean the raw data
│   ├── 02_eda.ipynb                  # Exploratory analysis of churn drivers
│   ├── 03_feature_engineering.ipynb  # Encoding + train/test split
│   ├── 04_modeling.ipynb             # Train & compare Logistic Regression / Random Forest, SMOTE, threshold tuning
│   └── 05_evaluation.ipynb           # Final model evaluation + business recommendations
├── requirements.txt
└── README.md
```

## Key EDA findings

- **Tenure** is the strongest churn driver — customers churn far more in their first few months
- **Month-to-month contracts** churn at a much higher rate than 1- or 2-year contracts
- **Fiber optic internet** customers churn more than DSL or no-internet customers
- **Electronic check** payers churn the most; automatic payment methods correlate with retention
- **Tech support** availability is strongly protective against churn

## Modeling approach

Five models/variants were trained and compared, optimizing for **recall on the churn class** rather than raw accuracy — in this business problem, missing an at-risk customer is more costly than a false alarm.

| Model | Accuracy | Recall (Churn) | ROC-AUC |
|---|---|---|---|
| **Logistic Regression (class-weighted)** | 0.74 | **0.79** | 0.84 |
| Logistic Regression + SMOTE | 0.75 | 0.63 | 0.81 |
| Random Forest + SMOTE | 0.78 | 0.60 | 0.82 |
| Logistic Regression (baseline) | 0.81 | 0.56 | 0.84 |
| Random Forest (baseline) | 0.79 | 0.50 | 0.82 |

**Selected model:** Logistic Regression with `class_weight='balanced'` — it catches ~79% of customers who actually churn, and stays interpretable, which matters for a model that will drive retention-outreach decisions. SMOTE oversampling was tested but didn't outperform simple class weighting.

## Bug fix from the original version

The original feature engineering step accidentally included `customerID` — a unique per-row identifier — as a categorical feature. Once one-hot encoded, this silently expanded the feature space to **~7,000 columns** (one per customer), which is both meaningless noise and a major performance/memory cost. This has been fixed by dropping `customerID` before encoding, bringing the feature set down to a legitimate **30 columns** and making the feature-importance output actually interpretable (contract type, internet service, and payment method now surface as the top drivers, instead of random customer IDs).

## Business recommendations

1. **Target new, month-to-month customers first** with an onboarding/check-in program in the first few months, and incentives to move to longer contracts.
2. **Investigate fiber optic and electronic-check segments** for service reliability or payment-experience friction.
3. **Promote tech support and autopay adoption** — both are strongly associated with lower churn and are low-cost retention levers.


Run the notebooks in order (01 → 05). Notebook 03 regenerates `X_train.csv` / `X_test.csv` / `y_train.csv` / `y_test.csv`, which are intentionally excluded from version control as derived, reproducible artifacts.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib


