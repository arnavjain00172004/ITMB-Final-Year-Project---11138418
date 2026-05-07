#!/usr/bin/env python3
"""
HSBC 2021 - Logistic Regression: Market Only vs Market + Sentiment

Steps:
  1. Load market_features_with_calendar.csv
  2. Filter to HSBC (HSBA.L), year 2021
  3. Load hsbc_2021_sentiment_scored.csv (FinBERT + VADER scores)
  4. Merge on date using a forward-fill so every trading day gets
     the most recent available sentiment signal
  5. Run Logistic Regression (market only) — baseline
  6. Run Logistic Regression (market + sentiment) — revised
  7. Print and compare: Accuracy, Precision, Recall, F1
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ── Config ────────────────────────────────────────────────────────────────────
MARKET_CSV    = "market_features_with_calendar.csv"
SENTIMENT_CSV = "HSBC_2021_Sentiment.csv"
TICKER        = "HSBA.L"
YEAR          = 2021
RANDOM_STATE  = 42
TEST_SIZE     = 0.2   # last 20% of rows used as test (time-ordered split)

# ── 1. Load & Filter Market Data ──────────────────────────────────────────────
print("=" * 60)
print(" STEP 1: Loading market data")
print("=" * 60)

market_df = pd.read_csv(MARKET_CSV, parse_dates=["Date"])
print(f"Full dataset: {len(market_df):,} rows, {market_df['Stock'].nunique()} stocks")

hsbc_df = market_df[
    (market_df["Stock"] == TICKER) &
    (market_df["Date"].dt.year == YEAR)
].copy().reset_index(drop=True)

print(f"HSBA.L 2021 rows: {len(hsbc_df)}")
print(f"Date range: {hsbc_df['Date'].min().date()} → {hsbc_df['Date'].max().date()}")
print(f"Target distribution:\n{hsbc_df['Target'].value_counts().to_string()}\n")

# ── 2. Load Sentiment Data ────────────────────────────────────────────────────
print("=" * 60)
print(" STEP 2: Loading sentiment scores")
print("=" * 60)

sent_df = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
print(f"Sentiment rows: {len(sent_df)}")
print(f"Columns: {list(sent_df.columns)}\n")

# Keep only the sentiment columns we need
sent_cols = ["date", "vader_compound", "finbert_score", "finbert_numeric"]
sent_df = sent_df[sent_cols].copy()

# Daily average — some dates have multiple headlines
sent_daily = (
    sent_df
    .groupby("date", as_index=False)
    .agg(
        vader_compound   = ("vader_compound",  "mean"),
        finbert_score    = ("finbert_score",   "mean"),
        finbert_numeric  = ("finbert_numeric", "mean"),
        headline_count   = ("vader_compound",  "count"),
    )
)
sent_daily.rename(columns={"date": "Date"}, inplace=True)
print(f"Unique sentiment dates: {len(sent_daily)}")

# ── 3. Merge via Forward-Fill ─────────────────────────────────────────────────
print("=" * 60)
print(" STEP 3: Merging market data with sentiment (forward-fill)")
print("=" * 60)

# Sort both by date
hsbc_df   = hsbc_df.sort_values("Date").reset_index(drop=True)
sent_daily = sent_daily.sort_values("Date").reset_index(drop=True)

# Merge then forward-fill sentiment into trading days with no news
merged_df = pd.merge_asof(
    hsbc_df,
    sent_daily,
    on="Date",
    direction="backward",   # use most recent prior sentiment
    tolerance=pd.Timedelta("30 days"),  # don't carry stale signals > 30 days
)

# Fill any remaining NaN sentiment (before first headline) with neutral values
merged_df["vader_compound"]  = merged_df["vader_compound"].fillna(0.0)
merged_df["finbert_score"]   = merged_df["finbert_score"].fillna(0.5)
merged_df["finbert_numeric"] = merged_df["finbert_numeric"].fillna(0.0)
merged_df["headline_count"]  = merged_df["headline_count"].fillna(0).astype(int)

print(f"Merged rows: {len(merged_df)}")
print(f"Rows with sentiment signal: {(merged_df['headline_count'] > 0).sum()}")
print(f"Rows forward-filled (no news that day): {(merged_df['headline_count'] == 0).sum()}\n")

# ── 4. Define Feature Sets ────────────────────────────────────────────────────
print("=" * 60)
print(" STEP 4: Defining features")
print("=" * 60)

MARKET_FEATURES = [
    "Lag1_Return", "MA5", "MA10", "Volatility10", "Volume_Change",
    "FTSE_Return", "FTSE_MA5", "FTSE_MA10", "FTSE_Volatility10",
    "STOXX_Return", "STOXX_MA5", "STOXX_MA10", "STOXX_Volatility10",
    "Day_of_Week_Monday", "Day_of_Week_Thursday",
    "Day_of_Week_Tuesday", "Day_of_Week_Wednesday",
]

SENTIMENT_FEATURES = [
    "vader_compound",
    "finbert_score",
    "finbert_numeric",
    "headline_count",
]

TARGET = "Target"

# Drop rows with any NaN in features
model_df = merged_df.dropna(subset=MARKET_FEATURES + [TARGET]).copy()
print(f"Rows after dropping NaN: {len(model_df)}")

# Convert bool day-of-week columns to int if needed
for col in ["Day_of_Week_Monday","Day_of_Week_Thursday","Day_of_Week_Tuesday","Day_of_Week_Wednesday"]:
    model_df[col] = model_df[col].astype(int)

# ── 5. Time-ordered Train/Test Split (no shuffle) ────────────────────────────
print("=" * 60)
print(" STEP 5: Time-ordered train/test split")
print("=" * 60)

split_idx = int(len(model_df) * (1 - TEST_SIZE))
train_df  = model_df.iloc[:split_idx]
test_df   = model_df.iloc[split_idx:]

print(f"Train: {len(train_df)} rows  ({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
print(f"Test:  {len(test_df)}  rows  ({test_df['Date'].min().date()} → {test_df['Date'].max().date()})\n")

# ── 6. Helper: train & evaluate logistic regression ─────────────────────────
def run_lr(X_train, X_test, y_train, y_test, label):
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_train)
    X_te    = scaler.transform(X_test)
    model   = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    return {
        "Label":     label,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
    }, y_pred, model, scaler

y_train = train_df[TARGET]
y_test  = test_df[TARGET]

# ── 7. Model A: Market Only ───────────────────────────────────────────────────
print("=" * 60)
print(" STEP 6: Model A — Logistic Regression (Market Features Only)")
print("=" * 60)

res_A, pred_A, model_A, _ = run_lr(
    train_df[MARKET_FEATURES],
    test_df[MARKET_FEATURES],
    y_train, y_test,
    "Market Only"
)

print(f"  Accuracy  : {res_A['Accuracy']:.4f}")
print(f"  Precision : {res_A['Precision']:.4f}")
print(f"  Recall    : {res_A['Recall']:.4f}")
print(f"  F1 Score  : {res_A['F1']:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, pred_A, target_names=["Down (0)", "Up (1)"]))
print("  Confusion Matrix:")
print(confusion_matrix(y_test, pred_A))

# ── 8. Model B: Market + Sentiment ───────────────────────────────────────────
print("=" * 60)
print(" STEP 7: Model B — Logistic Regression (Market + Sentiment Features)")
print("=" * 60)

ALL_FEATURES = MARKET_FEATURES + SENTIMENT_FEATURES

res_B, pred_B, model_B, _ = run_lr(
    train_df[ALL_FEATURES],
    test_df[ALL_FEATURES],
    y_train, y_test,
    "Market + Sentiment"
)

print(f"  Accuracy  : {res_B['Accuracy']:.4f}")
print(f"  Precision : {res_B['Precision']:.4f}")
print(f"  Recall    : {res_B['Recall']:.4f}")
print(f"  F1 Score  : {res_B['F1']:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, pred_B, target_names=["Down (0)", "Up (1)"]))
print("  Confusion Matrix:")
print(confusion_matrix(y_test, pred_B))

# ── 9. Side-by-Side Comparison ────────────────────────────────────────────────
print("=" * 60)
print(" COMPARISON: Market Only  vs  Market + Sentiment")
print("=" * 60)

compare_df = pd.DataFrame([res_A, res_B]).set_index("Label")
compare_df["Δ Accuracy"]  = compare_df["Accuracy"]  - compare_df["Accuracy"].iloc[0]
compare_df["Δ Precision"] = compare_df["Precision"] - compare_df["Precision"].iloc[0]
compare_df["Δ Recall"]    = compare_df["Recall"]    - compare_df["Recall"].iloc[0]
compare_df["Δ F1"]        = compare_df["F1"]        - compare_df["F1"].iloc[0]

print(compare_df.round(4).to_string())

compare_df.to_csv("hsbc_2021_lr_comparison.csv")
print("\n[INFO] Comparison saved → hsbc_2021_lr_comparison.csv")
