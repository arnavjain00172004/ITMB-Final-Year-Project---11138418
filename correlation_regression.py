import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

# ============================
# 1. LOAD DATA
# ============================
file_path = "market_features_with_calendar.csv"
df = pd.read_csv(file_path)

print("\nDataset Loaded")
print("Shape:", df.shape)

# ============================
# 2. SELECT NUMERIC COLUMNS
# ============================
numeric_df = df.select_dtypes(include=["float64", "int64"]).copy()

# ============================
# 3. CORRELATION MATRIX
# ============================
corr_matrix = numeric_df.corr()

# Save full correlation matrix
corr_matrix.to_csv("correlation_matrix.csv")
print("\nCorrelation matrix saved: correlation_matrix.csv")

# Plot heatmap
plt.figure(figsize=(14, 10))
im = plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im)

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix - Market Data Only")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("Correlation heatmap saved: correlation_heatmap.png")

# Return-focused correlations
return_corr = corr_matrix["Return"].sort_values(ascending=False)
return_corr.to_csv("return_correlations.csv", header=["Correlation_with_Return"])

print("\nTop correlations with Return:")
print(return_corr.head(10))

print("\nLowest correlations with Return:")
print(return_corr.tail(10))

# ============================
# 4. LINEAR REGRESSION
# ============================
# Predict continuous Return using market-only features
X = df.drop(columns=["Date", "Stock", "Target", "Return"])
y = df["Return"]

# Keep numeric features only
X = X.select_dtypes(include=["float64", "int64"]).copy()

# Replace inf with NaN and drop invalid rows
X = X.replace([np.inf, -np.inf], np.nan)
valid_index = X.dropna().index
X = X.loc[valid_index].copy()
y = y.loc[valid_index].copy()

print("\nLinear regression dataset shape:", X.shape)

# ============================
# 5. SKLEARN LINEAR REGRESSION
# ============================
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))  # fixed here

print("\nLinear Regression Results (sklearn)")
print("-----------------------------------")
print(f"R-squared: {r2:.6f}")
print(f"RMSE     : {rmse:.6f}")

# Save coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
}).sort_values(by="Coefficient", ascending=False)

coef_df.to_csv("linear_regression_coefficients.csv", index=False)
print("Linear regression coefficients saved: linear_regression_coefficients.csv")

# ============================
# 6. STATSMODELS OLS SUMMARY
# ============================
X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()

with open("linear_regression_summary.txt", "w") as f:
    f.write(ols_model.summary().as_text())

print("OLS summary saved: linear_regression_summary.txt")

# ============================
# 7. SAVE PREDICTIONS
# ============================
pred_df = X.copy()
pred_df["Actual_Return"] = y.values
pred_df["Predicted_Return"] = y_pred
pred_df.to_csv("linear_regression_predictions.csv", index=False)

print("Predictions saved: linear_regression_predictions.csv")

# ============================
# 8. PLOT ACTUAL VS PREDICTED
# ============================
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Return")
plt.ylabel("Predicted Return")
plt.title("Linear Regression: Actual vs Predicted Return")
plt.tight_layout()
plt.savefig("linear_regression_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

print("Actual vs predicted plot saved: linear_regression_actual_vs_predicted.png")