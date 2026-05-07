#!/usr/bin/env python3
"""
HSBC 2021 — Chi-Squared Test: Sentiment Labels vs Stock Movement Direction
==========================================================================
Tests whether the distribution of FinBERT / VADER sentiment labels
is statistically independent of the stock's next-day price direction
(Target: 1 = up, 0 = down).

A significant result (p < 0.05) means sentiment category and price
direction are NOT independent — i.e. sentiment has a statistically
meaningful association with stock movement.

Datasets needed (same folder as this script):
    market_features_with_calendar.csv   — market features dataset
    HSBC_2021_Sentiment.csv             — HSBC 2021 scored sentiment

Install requirements (run once):
    pip install pandas numpy scipy

Outputs:
    hsbc2021_chi2_results.csv           — test statistics and p-values
    hsbc2021_chi2_contingency_tables.csv — observed counts per category pair
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

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

# Aggregate to daily level: majority vote for labels, mean for scores
def majority(s):
    return s.value_counts().idxmax()

sent_daily = (
    sent_df
    .groupby("date", as_index=False)
    .agg(
        finbert_label   = ("finbert_label",  majority),
        vader_label     = ("vader_label",    majority),
        finbert_numeric = ("finbert_numeric","mean"),
        vader_compound  = ("vader_compound", "mean"),
        finbert_score   = ("finbert_score",  "mean"),
        headline_count  = ("vader_compound", "count"),
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

hsbc_df   = hsbc_df.sort_values("Date").reset_index(drop=True)
sent_daily = sent_daily.sort_values("Date").reset_index(drop=True)

merged_df = pd.merge_asof(
    hsbc_df, sent_daily,
    on="Date",
    direction="backward",
    tolerance=pd.Timedelta("30D"),
)

merged_df["finbert_label"] = merged_df["finbert_label"].fillna("neutral")
merged_df["vader_label"]   = merged_df["vader_label"].fillna("neutral")
merged_df["Target_Label"]  = merged_df["Target"].map({1: "Up", 0: "Down"})

print(f"Merged rows : {len(merged_df)}")
print(f"Rows with sentiment signal: {merged_df['finbert_label'].ne('neutral').sum()}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Chi-Squared Tests
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4: Chi-Squared Tests")
print("=" * 65)

results = []
contingency_records = []

def run_chi2(label_col, target_col, test_name):
    ct = pd.crosstab(merged_df[label_col], merged_df[target_col])
    chi2, p, dof, expected = chi2_contingency(ct)
    sig = "YES ✓" if p < 0.05 else "NO ✗"

    print(f"\n── {test_name} ───────────────────────────────────────────")
    print(f"  Contingency Table (observed counts):")
    print(ct.to_string())
    print(f"\n  Chi-Squared Statistic : {chi2:.4f}")
    print(f"  Degrees of Freedom    : {dof}")
    print(f"  P-Value               : {p:.6f}")
    print(f"  Significant (p<0.05)  : {sig}")

    results.append({
        "Test":          test_name,
        "Chi2_Statistic": round(chi2, 4),
        "Degrees_of_Freedom": dof,
        "P_Value":        round(p, 6),
        "Significant":    p < 0.05,
        "Interpretation": "Association found" if p < 0.05 else "No significant association",
    })

    for sent_cat in ct.index:
        for move_cat in ct.columns:
            contingency_records.append({
                "Test": test_name,
                "Sentiment_Category": sent_cat,
                "Price_Direction":    move_cat,
                "Observed_Count":     ct.loc[sent_cat, move_cat],
            })

run_chi2("finbert_label", "Target_Label", "FinBERT Label vs Price Direction (Up/Down)")
run_chi2("vader_label",   "Target_Label", "VADER Label vs Price Direction (Up/Down)")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Summary & Interpretation
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
res_df = pd.DataFrame(results)
print(res_df[["Test", "Chi2_Statistic", "P_Value", "Significant", "Interpretation"]].to_string(index=False))

print("""
── What the results mean ───────────────────────────────────────
  p < 0.05 (Significant = YES):
    Sentiment label category and price direction are NOT independent.
    Positive/negative/neutral FinBERT or VADER labels are distributed
    differently across Up vs Down days — sentiment carries a statistically
    meaningful signal for stock movement direction.

  p >= 0.05 (Significant = NO):
    No statistically significant association detected between sentiment
    label and price direction in this sample. This could reflect:
      - Small sample size (HSBC 2021 has ~250 trading days, limited news)
      - Sentiment noise from forward-filled neutral days
      - Market efficiency absorbing news before the next trading day
""")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — Save outputs
# ─────────────────────────────────────────────────────────────────
res_df.to_csv("hsbc2021_chi2_results.csv", index=False)
pd.DataFrame(contingency_records).to_csv("hsbc2021_chi2_contingency_tables.csv", index=False)

print("Files saved:")
print("  hsbc2021_chi2_results.csv")
print("  hsbc2021_chi2_contingency_tables.csv")
