"""
Merge HSBC 2021 market data with daily sentiment data
=====================================================
- Filter the master market dataset to HSBA.L (HSBC's LSE ticker) for 2021
- Aggregate sentiment to one row per day (some days have multiple headlines)
- Left-merge on date so every trading day is preserved, even when there is
  no news for that day (sentiment columns will be NaN on those days)

Run from the same folder as the two input CSVs, or edit the paths below.
"""

import os
import pandas as pd

# ---------- Paths (relative to this script) ----------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MARKET_FILE    = os.path.join(BASE_DIR, "market_features_with_calendar.csv")
SENTIMENT_FILE = os.path.join(BASE_DIR, "HSBC_2021_Sentiment.csv")
OUTPUT_FILE    = os.path.join(BASE_DIR, "HSBC_2021_market_sentiment_merged.csv")

# ---------- 1. Load and filter market data ----------
market = pd.read_csv(MARKET_FILE)
market["Date"] = pd.to_datetime(market["Date"])

hsbc_2021 = market[
    (market["Stock"] == "HSBA.L") &
    (market["Date"].dt.year == 2021)
].copy().sort_values("Date").reset_index(drop=True)

print(f"Market data — HSBA.L 2021: {len(hsbc_2021)} trading days "
      f"({hsbc_2021['Date'].min().date()} to "
      f"{hsbc_2021['Date'].max().date()})")

# ---------- 2. Load sentiment, strip BOM, parse dates ----------
sentiment = pd.read_csv(SENTIMENT_FILE)
sentiment.columns = [c.lstrip("\ufeff") for c in sentiment.columns]
sentiment["date"] = pd.to_datetime(sentiment["date"])

print(f"Sentiment data: {len(sentiment)} headlines across "
      f"{sentiment['date'].nunique()} unique dates")

# ---------- 3. Aggregate sentiment to one row per date ----------
# Some days have multiple headlines, so we collapse them sensibly:
#   - numeric scores  -> mean
#   - categorical labels (finbert_label, vader_label) -> majority vote
#   - models_agree -> True only if ALL headlines that day showed agreement
#   - headline / summary -> joined with " | " so all are preserved
#   - headline_count -> new column, useful for filtering / weighting later

def majority(s):
    """Return the most common value; ties resolved by first occurrence."""
    return s.value_counts().idxmax() if len(s) else None

agg_spec = {
    "headline":         lambda s: " | ".join(s.astype(str)),
    "summary":          lambda s: " | ".join(s.astype(str)),
    "finbert_label":    majority,
    "finbert_score":    "mean",
    "finbert_numeric":  "mean",
    "vader_compound":   "mean",
    "vader_pos":        "mean",
    "vader_neg":        "mean",
    "vader_neu":        "mean",
    "vader_label":      majority,
    "models_agree":     "all",
}

sentiment_daily = (
    sentiment
    .groupby("date", as_index=False)
    .agg(agg_spec)
)
# Add headline count for transparency
sentiment_daily["headline_count"] = (
    sentiment.groupby("date").size().reset_index(name="n")["n"].values
)

# Rename to make merged columns easy to identify downstream
sentiment_daily = sentiment_daily.rename(columns={
    "date":             "Date",
    "headline":         "headlines",
    "summary":          "summaries",
})

print(f"Sentiment after daily aggregation: {len(sentiment_daily)} rows")

# ---------- 4. Merge (left join keeps every trading day) ----------
merged = hsbc_2021.merge(sentiment_daily, on="Date", how="left")

# ---------- 5. Quick coverage report ----------
days_with_news = merged["headline_count"].notna().sum()
print(
    f"\nMerged dataset: {len(merged)} rows, "
    f"{days_with_news} days with news "
    f"({days_with_news / len(merged) * 100:.1f}% coverage), "
    f"{len(merged) - days_with_news} days without news"
)

# Sentiment dates that didn't match any trading day (weekends, holidays).
# Worth surfacing — those headlines drop out of the merge.
unmatched = sentiment_daily[~sentiment_daily["Date"].isin(merged["Date"])]
if not unmatched.empty:
    print(f"\n{len(unmatched)} sentiment day(s) didn't fall on a trading day "
          f"and were dropped:")
    for d in unmatched["Date"].dt.date.tolist():
        print(f"  - {d}")
    print("(Consider rolling these forward to the next trading day if you "
          "want them captured — see notes at the bottom of this script.)")

# ---------- 6. Save ----------
merged.to_csv(OUTPUT_FILE, index=False)
print(f"\nWrote: {OUTPUT_FILE}")
print(f"Columns: {list(merged.columns)}")

# ---------------------------------------------------------------------------
# OPTIONAL — roll non-trading-day news forward to the next trading day
# ---------------------------------------------------------------------------
# If a piece of news drops on a Saturday, its market impact is realised the
# following Monday. To capture that, replace the merge step above with:
#
#   merged = pd.merge_asof(
#       hsbc_2021.sort_values("Date"),
#       sentiment_daily.sort_values("Date"),
#       on="Date",
#       direction="backward",   # carries news to the next trading day
#       tolerance=pd.Timedelta(days=3),  # don't carry news >3 days
#   )
#
# This is a defensible modelling choice but it changes the semantics of
# the dataset, so I've left the strict same-day merge as the default.