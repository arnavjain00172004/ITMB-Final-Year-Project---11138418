import yfinance as yf
import pandas as pd

indices = {
    "FTSE": "^FTSE",
    "STOXX": "^STOXX50E"
}

start_date = "2020-01-01"
end_date = "2026-01-01"

all_data = []

for index_name, ticker in indices.items():
    print(f"Downloading {index_name} ({ticker})...")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        print(f"No data for {index_name}")
        continue

    # Flatten columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Keep only needed columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Index"] = index_name

    # Reorder
    df = df[["Date", "Index", "Open", "High", "Low", "Close", "Volume"]]

    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

# Make sure types are correct
final_df["Date"] = pd.to_datetime(final_df["Date"])
for col in ["Open", "High", "Low", "Close", "Volume"]:
    final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

# Drop any bad rows
final_df = final_df.dropna(subset=["Date", "Open", "High", "Low", "Close"])

final_df.to_csv("market_index_data_final.csv", index=False)

print("\nDone.")
print(final_df.head())
print(final_df.tail())
print(final_df.shape)
print(final_df["Index"].value_counts())
print(final_df.dtypes)