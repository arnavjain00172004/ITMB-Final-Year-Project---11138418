import pandas as pd

# ============================
# 1. LOAD DATA
# ============================
file_path = "market_features_with_calendar.csv"

df = pd.read_csv(file_path)

print("\nDataset Loaded")
print("Shape:", df.shape)
print(df.head())


# ============================
# 2. DESCRIPTIVE STATISTICS
# ============================

# Select only numeric columns (exclude target if needed later)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Full descriptive statistics
desc_stats = numeric_df.describe().T  # transpose for readability

# Add extra stats
desc_stats['median'] = numeric_df.median()
desc_stats['skewness'] = numeric_df.skew()
desc_stats['kurtosis'] = numeric_df.kurtosis()

# Save
desc_stats.to_csv("descriptive_statistics.csv")

print("\nDescriptive Statistics Saved: descriptive_statistics.csv")
print(desc_stats.head())


# ============================
# 3. TARGET FREQUENCY DISTRIBUTION
# ============================

# Frequency count
target_counts = df['Target'].value_counts().sort_index()

# Percentage distribution
target_percentage = df['Target'].value_counts(normalize=True).sort_index() * 100

# Combine into one table
target_dist = pd.DataFrame({
    "Count": target_counts,
    "Percentage (%)": target_percentage
})

# Save
target_dist.to_csv("target_distribution.csv")

print("\nTarget Distribution Saved: target_distribution.csv")
print(target_dist)


# ============================
# 4. OPTIONAL: DAY OF WEEK DISTRIBUTION
# ============================

# If your dataset has day-of-week dummies
day_cols = [col for col in df.columns if "Day_of_Week" in col]

if day_cols:
    day_distribution = df[day_cols].sum().sort_values(ascending=False)
    day_distribution.to_csv("day_of_week_distribution.csv")

    print("\nDay of Week Distribution Saved: day_of_week_distribution.csv")
    print(day_distribution)


# ============================
# 5. BASIC INTERPRETATION OUTPUT
# ============================

print("\nQuick Insights:")
print(f"Total rows: {len(df)}")
print(f"Up days (Target=1): {target_counts.get(1, 0)}")
print(f"Down days (Target=0): {target_counts.get(0, 0)}")

imbalance = abs(target_percentage.get(1, 0) - target_percentage.get(0, 0))
print(f"Class imbalance difference: {imbalance:.2f}%")