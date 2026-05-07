#!/usr/bin/env python3
"""
BP 2025 Logistic Regression — Market Only vs Market + Sentiment
=========================================================================
Compares two logistic regression models on the exact same sample:
  Model A: BP 2025 market features only
  Model B: BP 2025 market features + FinBERT & VADER sentiment scores

Install requirements (run once):
    pip install pandas numpy scikit-learn

Input files (place in the same folder as this script):
    market_features_with_calendar.csv
    bp_2025_sentiment_scored.csv

Output files:
    bp2025_model_A_market_only_performance.csv
    bp2025_model_B_market_sentiment_performance.csv
    bp2025_model_A_predictions.csv
    bp2025_model_B_predictions.csv
    bp2025_coefficients_market_only.csv
    bp2025_coefficients_market_sentiment.csv
    bp2025_comparison.csv
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MARKET_CSV    = "market_features_with_calendar.csv"
SENTIMENT_CSV = "bp_2025_sentiment_scored.csv"
TICKER        = "BP.L"
YEAR          = 2025
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
MAX_ITER      = 3000

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load & filter market data to BP 2025
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: Loading and filtering market data (BP 2025)")
print("=" * 65)

market_df = pd.read_csv(MARKET_CSV)
market_df["Date"] = pd.to_datetime(market_df["Date"])

company_df = market_df[
    (market_df["Stock"] == TICKER) &
    (market_df["Date"].dt.year == YEAR)
].copy().reset_index(drop=True)

print(f"{TICKER} {YEAR} rows : {len(company_df)}")
print(f"Date range         : {company_df['Date'].min().date()} → {company_df['Date'].max().date()}")
print(f"Target counts      :\n{company_df['Target'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Load sentiment, aggregate to one row per day
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 2: Loading BP 2025 sentiment scores")
print("=" * 65)

sent_df = pd.read_csv(SENTIMENT_CSV)
sent_df["date"] = pd.to_datetime(sent_df["date"])

sent_daily = (
    sent_df
    .groupby("date", as_index=False)
    .agg(
        finbert_score   = ("finbert_score",   "mean"),
        finbert_numeric = ("finbert_numeric", "mean"),
        vader_compound  = ("vader_compound",  "mean"),
        vader_pos       = ("vader_pos",       "mean"),
        vader_neg       = ("vader_neg",       "mean"),
        headline_count  = ("vader_compound",  "count"),
    )
)
sent_daily = sent_daily.rename(columns={"date": "Date"})
print(f"Unique sentiment dates: {len(sent_daily)}")
print(f"Sentiment columns     : finbert_score, finbert_numeric, vader_compound, vader_pos, vader_neg, headline_count\n")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Merge sentiment into market rows
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 3: Merging datasets (backward asof join, 30-day tolerance)")
print("=" * 65)

company_df = company_df.sort_values("Date").reset_index(drop=True)
sent_daily = sent_daily.sort_values("Date").reset_index(drop=True)

merged_df = pd.merge_asof(
    company_df,
    sent_daily,
    on="Date",
    direction="backward",
    tolerance=pd.Timedelta("30D"),
)

merged_df["finbert_score"]   = merged_df["finbert_score"].fillna(0.5)
merged_df["finbert_numeric"] = merged_df["finbert_numeric"].fillna(0.0)
merged_df["vader_compound"]  = merged_df["vader_compound"].fillna(0.0)
merged_df["vader_pos"]       = merged_df["vader_pos"].fillna(0.0)
merged_df["vader_neg"]       = merged_df["vader_neg"].fillna(0.0)
merged_df["headline_count"]  = merged_df["headline_count"].fillna(0).astype(int)

print(f"Merged rows           : {len(merged_df)}")
print(f"Rows with news signal : {(merged_df['headline_count'] > 0).sum()}")
print(f"Rows forward-filled   : {(merged_df['headline_count'] == 0).sum()}\n")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Build feature sets
# ─────────────────────────────────────────────────────────────────
SENTIMENT_FEATURES = [
    "finbert_score", "finbert_numeric",
    "vader_compound", "vader_pos", "vader_neg",
    "headline_count",
]

X_market = merged_df.drop(
    columns=["Date", "Stock", "Target"] + SENTIMENT_FEATURES
).copy()

X_full = merged_df.drop(columns=["Date", "Stock", "Target"]).copy()

y = merged_df["Target"].copy()

X_market = X_market.replace([np.inf, -np.inf], np.nan)
X_full   = X_full.replace([np.inf, -np.inf], np.nan)

valid_idx = X_full.dropna().index.intersection(X_market.dropna().index)
X_market  = X_market.loc[valid_idx].copy()
X_full    = X_full.loc[valid_idx].copy()
y         = y.loc[valid_idx].copy()

print(f"Dataset shape after cleaning:")
print(f"  Market only : {X_market.shape}")
print(f"  Full (M+S)  : {X_full.shape}")
print(f"\nTarget distribution:")
print(y.value_counts(normalize=True).round(4).to_string())

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Train/Test split (same split for both models)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 5: Train/Test split (stratified shuffle, test_size=0.2)")
print("=" * 65)

X_train_m, X_test_m, y_train, y_test = train_test_split(
    X_market, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=y,
)

X_train_f = X_full.loc[X_train_m.index]
X_test_f  = X_full.loc[X_test_m.index]

print(f"Train rows : {len(X_train_m)}")
print(f"Test rows  : {len(X_test_m)}")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — Helper: scale, train, evaluate
# ─────────────────────────────────────────────────────────────────
def run_logistic(X_tr, X_te, y_tr, y_te, label):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)

    model = LogisticRegression(
        max_iter=MAX_ITER,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, zero_division=0,
                                target_names=["Down (0)", "Up (1)"]))

    metrics = {
        "Model":     label,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
    }

    pred_df = X_te.copy()
    pred_df["Actual"]                = y_te.values
    pred_df["Predicted"]             = y_pred
    pred_df["Predicted_Probability"] = y_prob

    coef_df = pd.DataFrame({
        "Feature":     X_tr.columns,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", ascending=False)

    return metrics, pred_df, coef_df

# ─────────────────────────────────────────────────────────────────
# STEP 7 — Model A: Market Only
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 6: MODEL A — Market Features Only")
print("=" * 65)

res_A, preds_A, coef_A = run_logistic(
    X_train_m, X_test_m, y_train, y_test,
    "BP 2025 — Logistic Regression (Market Only)"
)

# ─────────────────────────────────────────────────────────────────
# STEP 8 — Model B: Market + Sentiment
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 7: MODEL B — Market + Sentiment Features")
print("=" * 65)

res_B, preds_B, coef_B = run_logistic(
    X_train_f, X_test_f, y_train, y_test,
    "BP 2025 — Logistic Regression (Market + Sentiment)"
)

# ─────────────────────────────────────────────────────────────────
# STEP 9 — Side-by-side comparison
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SIDE-BY-SIDE COMPARISON")
print("=" * 65)

compare_df = pd.DataFrame([res_A, res_B]).set_index("Model")
delta = (
    compare_df.loc["BP 2025 — Logistic Regression (Market + Sentiment)"]
    - compare_df.loc["BP 2025 — Logistic Regression (Market Only)"]
).round(4)
compare_df.loc["Δ (B − A)"] = delta
print(compare_df.round(4).to_string())

# ─────────────────────────────────────────────────────────────────
# STEP 10 — Save outputs
# ─────────────────────────────────────────────────────────────────
pd.DataFrame([res_A]).to_csv("bp2025_model_A_market_only_performance.csv",      index=False)
pd.DataFrame([res_B]).to_csv("bp2025_model_B_market_sentiment_performance.csv", index=False)
preds_A.to_csv("bp2025_model_A_predictions.csv",          index=False)
preds_B.to_csv("bp2025_model_B_predictions.csv",          index=False)
coef_A.to_csv("bp2025_coefficients_market_only.csv",      index=False)
coef_B.to_csv("bp2025_coefficients_market_sentiment.csv", index=False)
compare_df.reset_index().to_csv("bp2025_comparison.csv",  index=False)

print("\nFiles saved:")
print("  bp2025_model_A_market_only_performance.csv")
print("  bp2025_model_B_market_sentiment_performance.csv")
print("  bp2025_model_A_predictions.csv")
print("  bp2025_model_B_predictions.csv")
print("  bp2025_coefficients_market_only.csv")
print("  bp2025_coefficients_market_sentiment.csv")
print("  bp2025_comparison.csv")
