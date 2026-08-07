"""
Cognifyz ML Internship - Task 1
Predict Restaurant Ratings
--------------------------------
Objective: Build a machine learning model to predict the aggregate rating
of a restaurant based on other features.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset_.csv"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

# Restaurants with 0 votes have Aggregate rating = 0 ("Not rated").
# These aren't genuine ratings, so we drop them for the regression task.
df = df[df["Votes"] > 0].copy()
print(f"Shape after removing unrated restaurants: {df.shape}")

# ---------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# ---------------------------------------------------------------
# 3. Feature selection & encoding
# ---------------------------------------------------------------
features = [
    "Country Code", "Average Cost for two", "Price range", "Votes",
    "Has Table booking", "Has Online delivery", "Is delivering now",
    "Currency", "City",
]
target = "Aggregate rating"

data = df[features + [target]].copy()

binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now"]
for col in binary_cols:
    data[col] = data[col].map({"Yes": 1, "No": 0})

# Label-encode high-cardinality categoricals
for col in ["Currency", "City"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

X = data[features]
y = data[target]

# ---------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------
# 5. Train models
# ---------------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)

tree_reg = DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE)
tree_reg.fit(X_train, y_train)
tree_pred = tree_reg.predict(X_test)

# ---------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------
results = {
    "Linear Regression": (lin_pred, lin_reg),
    "Decision Tree Regression": (tree_pred, tree_reg),
}

results_path = BASE_DIR / "results.txt"
with open(results_path, "w") as f:
    for name, (pred, _) in results.items():
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        line = f"{name}: MSE={mse:.4f}, R2={r2:.4f}"
        print(line)
        f.write(line + "\n")

# ---------------------------------------------------------------
# 7. Feature importance (Decision Tree)
# ---------------------------------------------------------------
importances = pd.Series(tree_reg.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances (Decision Tree):")
print(importances)
importances.to_csv(BASE_DIR / "feature_importance.csv")

plt.figure(figsize=(8, 5))
importances.plot(kind="bar", color="teal")
plt.title("Feature Importance - Restaurant Rating Prediction")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(BASE_DIR / "feature_importance.png", dpi=150)
plt.close()

# Actual vs predicted plot for the better model
plt.figure(figsize=(6, 6))
plt.scatter(y_test, tree_pred, alpha=0.3, s=10)
plt.plot([0, 5], [0, 5], "r--")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Decision Tree: Actual vs Predicted Ratings")
plt.tight_layout()
plt.savefig(BASE_DIR / "actual_vs_predicted.png", dpi=150)
plt.close()

print("\nDone. See results.txt, feature_importance.png, actual_vs_predicted.png")
