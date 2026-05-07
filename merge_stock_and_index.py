import pandas as pd

# Load datasets
stocks = pd.read_csv("market_features_dataset.csv")
indices = pd.read_csv("market_index_features.csv")

# Convert dates
stocks["Date"] = pd.to_datetime(stocks["Date"])
indices["Date"] = pd.to_datetime(indices["Date"])

# Split index data
ftse = indices[indices["Index"] == "FTSE"].copy()
stoxx = indices[indices["Index"] == "STOXX"].copy()

# Rename columns so both can coexist after merge
ftse = ftse.rename(columns={
    "Index_Return": "FTSE_Return",
    "Index_MA5": "FTSE_MA5",
    "Index_MA10": "FTSE_MA10",
    "Index_Volatility10": "FTSE_Volatility10"
})

stoxx = stoxx.rename(columns={
    "Index_Return": "STOXX_Return",
    "Index_MA5": "STOXX_MA5",
    "Index_MA10": "STOXX_MA10",
    "Index_Volatility10": "STOXX_Volatility10"
})

# Keep only needed columns
ftse = ftse[["Date", "FTSE_Return", "FTSE_MA5", "FTSE_MA10", "FTSE_Volatility10"]]
stoxx = stoxx[["Date", "STOXX_Return", "STOXX_MA5", "STOXX_MA10", "STOXX_Volatility10"]]

# Merge both index datasets into stock dataset by date
merged = stocks.merge(ftse, on="Date", how="left")
merged = merged.merge(stoxx, on="Date", how="left")

# Drop rows with any missing values after merge
merged = merged.dropna().copy()

# Save final merged dataset
merged.to_csv("market_features_with_indices.csv", index=False)

print("Merge completed.")
print(merged.head())
print("Dataset shape:", merged.shape)
print("Saved to: market_features_with_indices.csv")