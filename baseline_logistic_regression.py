import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("market_features_with_calendar.csv")

# -----------------------------
# 2. Prepare features and target
# -----------------------------
X = df.drop(columns=["Date", "Stock", "Target"])
y = df["Target"]

# Replace inf values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Keep only rows without missing values
valid_index = X.dropna().index
X = X.loc[valid_index].copy()
y = y.loc[valid_index].copy()

print("Dataset shape after cleaning:", X.shape)
print("\nTarget distribution:")
print(y.value_counts(normalize=True))

# -----------------------------
# 3. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# -----------------------------
# 4. Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train Logistic Regression
# -----------------------------
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# 6. Predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 7. Evaluation metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\nScaled Logistic Regression Results")
print("----------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# 8. Coefficients (optional interpretability)
# -----------------------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

coef_df.to_csv("logistic_coefficients.csv", index=False)

# -----------------------------
# 9. Save model performance for Tableau
# -----------------------------
performance_df = pd.DataFrame({
    "Model": ["Logistic Regression (Scaled, Balanced)"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
})

performance_df.to_csv("model_performance_logistic.csv", index=False)

# -----------------------------
# 10. Save predictions for Tableau
# -----------------------------
predictions_df = X_test.copy()
predictions_df["Actual"] = y_test.values
predictions_df["Predicted"] = y_pred
predictions_df["Predicted_Probability"] = y_prob

predictions_df.to_csv("model_predictions_logistic.csv", index=False)

print("\nFiles exported:")
print("- model_performance_logistic.csv")
print("- model_predictions_logistic.csv")
print("- logistic_coefficients.csv")