import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Keep only valid rows
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
# 4. Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# 6. Evaluation metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\nRandom Forest Results")
print("---------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# 7. Save model performance
# -----------------------------
performance_df = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
})

performance_df.to_csv("model_performance_random_forest.csv", index=False)

# -----------------------------
# 8. Save predictions
# -----------------------------
predictions_df = X_test.copy()
predictions_df["Actual"] = y_test.values
predictions_df["Predicted"] = y_pred
predictions_df["Predicted_Probability"] = y_prob

predictions_df.to_csv("model_predictions_random_forest.csv", index=False)

# -----------------------------
# 9. Save feature importance
# -----------------------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("feature_importance_random_forest.csv", index=False)

print("\nFiles exported:")
print("- model_performance_random_forest.csv")
print("- model_predictions_random_forest.csv")
print("- feature_importance_random_forest.csv")