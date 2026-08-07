"""
Cognifyz ML Internship - Task 3
Cuisine Classification
--------------------------------
Objective: Develop a machine learning model to classify restaurants
based on their cuisines.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset_.csv"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# Each restaurant can list multiple cuisines (e.g. "French, Japanese").
# For a clean classification target we use the FIRST (primary) cuisine listed.
df["Primary Cuisine"] = df["Cuisines"].apply(lambda x: x.split(",")[0].strip())

# Restrict to the top 10 most common cuisines to keep classes well-represented
# (many of the 1800+ combinations only appear once, which isn't learnable).
top_cuisines = df["Primary Cuisine"].value_counts().head(10).index
df = df[df["Primary Cuisine"].isin(top_cuisines)].copy()
print("Classes used:", list(top_cuisines))
print(f"Filtered shape: {df.shape}")

# ---------------------------------------------------------------
# 2. Feature selection & encoding
# ---------------------------------------------------------------
features = [
    "Country Code", "City", "Average Cost for two", "Price range",
    "Votes", "Aggregate rating", "Has Table booking", "Has Online delivery",
]
target = "Primary Cuisine"

data = df[features + [target]].copy()

binary_cols = ["Has Table booking", "Has Online delivery"]
for col in binary_cols:
    data[col] = data[col].map({"Yes": 1, "No": 0})

city_le = LabelEncoder()
data["City"] = city_le.fit_transform(data["City"].astype(str))

target_le = LabelEncoder()
data[target] = target_le.fit_transform(data[target])

X = data[features]
y = data[target]

# ---------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

rf_clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# ---------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------
results_path = BASE_DIR / "results.txt"
with open(results_path, "w") as f:
    for name, pred in [("Logistic Regression", log_pred), ("Random Forest", rf_pred)]:
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
        line = f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}"
        print(line)
        f.write(line + "\n")

    f.write("\nRandom Forest classification report:\n")
    report = classification_report(y_test, rf_pred, target_names=target_le.classes_, zero_division=0)
    print(report)
    f.write(report)

# ---------------------------------------------------------------
# 6. Confusion matrix for the best model
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.xlabel("Predicted Cuisine")
plt.ylabel("Actual Cuisine")
plt.title("Random Forest: Cuisine Classification Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# Per-cuisine analysis: which cuisines are hardest to classify
per_class_acc = []
for i, cuisine in enumerate(target_le.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        acc_i = accuracy_score(y_test[mask], rf_pred[mask])
        per_class_acc.append((cuisine, acc_i, mask.sum()))
per_class_df = pd.DataFrame(per_class_acc, columns=["Cuisine", "Accuracy", "Support"]).sort_values("Accuracy")
per_class_df.to_csv(BASE_DIR / "per_cuisine_accuracy.csv", index=False)
print("\nPer-cuisine accuracy (hardest first):")
print(per_class_df)

print("\nDone. See results.txt, confusion_matrix.png, per_cuisine_accuracy.csv")
