import pandas as pd

# Load market data
input_file = "market_data_final.csv"
df = pd.read_csv(input_file)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort values by stock and date (important for time series calculations)
df = df.sort_values(["Stock", "Date"]).copy()

# Group by stock so each company's features are calculated separately
grouped = df.groupby("Stock", group_keys=False)

# 1. Daily return
df["Return"] = grouped["Close"].pct_change()

# 2. Lagged return (previous day's return)
df["Lag1_Return"] = grouped["Return"].shift(1)

# 3. 5-day moving average of closing price
df["MA5"] = grouped["Close"].transform(lambda x: x.rolling(5).mean())

# 4. 10-day moving average of closing price
df["MA10"] = grouped["Close"].transform(lambda x: x.rolling(10).mean())

# 5. 10-day rolling volatility (standard deviation of returns)
df["Volatility10"] = grouped["Return"].transform(lambda x: x.rolling(10).std())

# 6. Volume change
df["Volume_Change"] = grouped["Volume"].pct_change()

# 7. Target variable: next day's return direction
next_day_return = grouped["Return"].shift(-1)
df["Target"] = (next_day_return > 0).astype(int)

# Remove rows with missing values created by rolling calculations
final_df = df.dropna().copy()

# Select final columns for modelling
final_df = final_df[[
    "Date",
    "Stock",
    "Return",
    "Lag1_Return",
    "MA5",
    "MA10",
    "Volatility10",
    "Volume_Change",
    "Target",
]]

# Save processed dataset
output_file = "market_features_dataset.csv"
final_df.to_csv(output_file, index=False)

print("Feature engineering completed.")
print(final_df.head())
print("Dataset shape:", final_df.shape)
print("Saved to:", output_file)
