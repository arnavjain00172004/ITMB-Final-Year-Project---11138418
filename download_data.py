import yfinance as yf
import pandas as pd

stocks = [
    "HSBA.L",
    "BP.L",
    "AZN.L",
    "ULVR.L",
    "SAP.DE",
    "SIE.DE",
    "MC.PA",
    "TTE.PA"
]

start_date = "2020-01-01"
end_date = "2026-01-01"

all_data = []

for stock in stocks:
    print(f"Downloading {stock}...")

    df = yf.download(
        stock,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        print(f"No data for {stock}")
        continue

    # If columns come as MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index so Date becomes a normal column
    df = df.reset_index()

    # Keep only needed columns
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[keep_cols].copy()

    # Add stock ticker
    df["Stock"] = stock

    # Reorder columns
    df = df[["Date", "Stock", "Open", "High", "Low", "Close", "Volume"]]

    all_data.append(df)

# Combine all stocks vertically
final_df = pd.concat(all_data, ignore_index=True)

# Save clean dataset
final_df.to_csv("market_data_final.csv", index=False)

print("\nDone.")
print(final_df.head())
print(final_df.shape)
print(final_df.columns.tolist())