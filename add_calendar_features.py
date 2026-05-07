import pandas as pd

# Load dataset
df = pd.read_csv("market_features_with_indices.csv")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Create day of week
df["Day_of_Week"] = df["Date"].dt.day_name()

# One-hot encode day of week
df = pd.get_dummies(df, columns=["Day_of_Week"], drop_first=True)

# Save updated dataset
df.to_csv("market_features_with_calendar.csv", index=False)

print("Calendar features added successfully.")
print(df.head())
print("Dataset shape:", df.shape)