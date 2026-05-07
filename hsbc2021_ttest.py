#!/usr/bin/env python3
"""
HSBC 2021 — Independent Samples T-Tests: Sentiment Scores vs Stock Movement
============================================================================
Tests whether the mean sentiment scores (vader_compound, finbert_score,
finbert_numeric) differ significantly between trading days where the stock
went UP (Target=1) vs DOWN (Target=0).

8 T-tests are performed — one per sentiment score / market feature pair:
  1.  vader_compound       — Up days vs Down days
  2.  finbert_score        — Up days vs Down days
  3.  finbert_numeric      — Up days vs Down days
  4.  vader_pos            — Up days vs Down days
  5.  vader_neg            — Up days vs Down days
  6.  Lag1_Return          — Up days vs Down days (market baseline)
  7.  Volatility10         — Up days vs Down days (market baseline)
  8.  vader_compound * MA5 — Interaction proxy: sentiment x momentum

A significant result (p < 0.05) means the mean value of that variable
is significantly different between Up and Down days — it carries a
separating signal for price direction prediction.

Datasets needed (same folder as this script):
    market_features_with_calendar.csv
    HSBC_2021_Sentiment.csv

Install requirements (run once):
    pip install pandas numpy scipy

Outputs:
    hsbc2021_ttest_results.csv          — all 8 test statistics and p-values
    hsbc2021_ttest_group_means.csv      — mean values per group (Up vs Down)
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MARKET_CSV    = "market_features_with_calendar.csv"
SENTIMENT_CSV = "HSBC_2021_Sentiment.csv"
TICKER        = "HSBA.L"
YEAR          = 2021

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load & filter market data
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: Loading and filtering HSBC 2021 market data")
print("=" * 65)

market_df = pd.read_csv(MARKET_CSV)
market_df["Date"] = pd.to_datetime(market_df["Date"])

hsbc_df = market_df[
    (market_df["Stock"] == TICKER) &
    (market_df["Date"].dt.year == YEAR)
].copy().reset_index(drop=True)

print(f"HSBC 2021 trading rows : {len(hsbc_df)}")
print(f"Target distribution    :")
print(hsbc_df["Target"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Load & aggregate sentiment
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 2: Loading HSBC 2021 sentiment scores")
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
sent_daily.rename(columns={"date": "Date"}, inplace=True)
print(f"Unique sentiment dates : {len(sent_daily)}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Merge (backward asof, 30-day tolerance)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3: Merging datasets")
print("=" * 65)

hsbc_df    = hsbc_df.sort_values("Date").reset_index(drop=True)
sent_daily = sent_daily.sort_values("Date").reset_index(drop=True)

merged_df = pd.merge_asof(
    hsbc_df, sent_daily,
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

# Interaction feature: sentiment scaled by short-term momentum
merged_df["vader_x_MA5"] = merged_df["vader_compound"] * merged_df["MA5"]

print(f"Merged rows : {len(merged_df)}")

# Split into Up and Down groups
up_days   = merged_df[merged_df["Target"] == 1]
down_days = merged_df[merged_df["Target"] == 0]
print(f"Up days   : {len(up_days)}")
print(f"Down days : {len(down_days)}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — 8 Independent Samples T-Tests
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4: Running 8 T-Tests (Up days vs Down days)")
print("=" * 65)

TESTS = [
    ("vader_compound",  "VADER Compound Score          (sentiment intensity)"),
    ("finbert_score",   "FinBERT Confidence Score       (model certainty)"),
    ("finbert_numeric", "FinBERT Numeric Score          (+1/0/-1 polarity)"),
    ("vader_pos",       "VADER Positive Sub-Score       (positive sentiment share)"),
    ("vader_neg",       "VADER Negative Sub-Score       (negative sentiment share)"),
    ("Lag1_Return",     "Lag1 Return                    (prior day market return)"),
    ("Volatility10",    "10-Day Volatility              (market risk level)"),
    ("vader_x_MA5",     "VADER Compound × MA5           (sentiment × momentum interaction)"),
]

results    = []
group_rows = []

for col, label in TESTS:
    up_vals   = up_days[col].dropna()
    down_vals = down_days[col].dropna()

    # Levene test for equal variances to choose correct t-test variant
    _, lev_p = levene(up_vals, down_vals)
    equal_var = lev_p >= 0.05

    t_stat, p_val = ttest_ind(up_vals, down_vals, equal_var=equal_var)
    sig = "YES ✓" if p_val < 0.05 else "NO ✗"

    up_mean   = round(up_vals.mean(), 6)
    down_mean = round(down_vals.mean(), 6)
    diff      = round(up_mean - down_mean, 6)

    print(f"\n── T-Test {len(results)+1}: {label} ──")
    print(f"  Mean (Up days)   : {up_mean:.6f}")
    print(f"  Mean (Down days) : {down_mean:.6f}")
    print(f"  Difference       : {diff:.6f}")
    print(f"  Equal variance   : {equal_var} (Levene p={lev_p:.4f})")
    print(f"  T-Statistic      : {t_stat:.4f}")
    print(f"  P-Value          : {p_val:.6f}")
    print(f"  Significant      : {sig}")

    results.append({
        "Test_Number":      len(results) + 1,
        "Variable":         col,
        "Description":      label.strip(),
        "Mean_Up_Days":     up_mean,
        "Mean_Down_Days":   down_mean,
        "Mean_Difference":  diff,
        "Equal_Variance":   equal_var,
        "T_Statistic":      round(t_stat, 4),
        "P_Value":          round(p_val, 6),
        "Significant":      p_val < 0.05,
        "Interpretation":   "Means differ significantly" if p_val < 0.05 else "No significant difference",
    })

    for direction, vals in [("Up", up_vals), ("Down", down_vals)]:
        group_rows.append({
            "Variable":   col,
            "Direction":  direction,
            "N":          len(vals),
            "Mean":       round(vals.mean(), 6),
            "Std":        round(vals.std(),  6),
            "Min":        round(vals.min(),  6),
            "Max":        round(vals.max(),  6),
        })

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY OF ALL 8 T-TESTS")
print("=" * 65)
res_df = pd.DataFrame(results)
summary_cols = ["Test_Number", "Variable", "Mean_Up_Days", "Mean_Down_Days",
                "T_Statistic", "P_Value", "Significant", "Interpretation"]
print(res_df[summary_cols].to_string(index=False))

sig_count = res_df["Significant"].sum()
print(f"\n  {sig_count} out of 8 tests significant at p < 0.05")

print("""
── What the results mean ───────────────────────────────────────
  T-Test 1 (vader_compound): Did Up days have more positive VADER
    sentiment than Down days? Significant = sentiment intensity
    differs meaningfully between rising and falling price days.

  T-Test 2 (finbert_score): Was FinBERT more confident on Up vs Down
    days? Significant = model certainty differs by price direction.

  T-Test 3 (finbert_numeric): Did Up days correlate with more
    positive FinBERT polarity labels (+1) vs Down days (-1)?

  T-Test 4 (vader_pos): Was the positive sentiment share higher on
    Up days? Directly tests positive tone vs price movement.

  T-Test 5 (vader_neg): Was the negative sentiment share higher on
    Down days? Directly tests negative tone vs price decline.

  T-Test 6 (Lag1_Return): Does the prior day return differ between
    Up and Down days? Baseline test — momentum effect.

  T-Test 7 (Volatility10): Is market volatility level different on
    Up vs Down days? High volatility can suppress predictability.

  T-Test 8 (vader_x_MA5): Does the interaction of sentiment and
    momentum (VADER × MA5) differ between Up and Down days?
    Significant = sentiment is more informative in trending markets.
""")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — Save outputs
# ─────────────────────────────────────────────────────────────────
res_df.to_csv("hsbc2021_ttest_results.csv", index=False)
pd.DataFrame(group_rows).to_csv("hsbc2021_ttest_group_means.csv", index=False)

print("Files saved:")
print("  hsbc2021_ttest_results.csv")
print("  hsbc2021_ttest_group_means.csv")
