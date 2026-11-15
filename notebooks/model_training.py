# %% [markdown]
# 05 - Model Training and Evaluation
# Uses chronological train/test splitting.

# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# %%
# Project paths

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Load dataset

df = pd.read_csv(DATA)

df["Date"] = pd.to_datetime(df["Date"])

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# %%
# Features

features = [
    "Rainfall",
    "Rainfall_3d",
    "Rainfall_7d",
    "Rainfall_Lag1",
    "Rainfall_Lag2",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Pressure",
    "Month"
]


# %%
# Check required columns

required_columns = features + ["Flood_Risk", "Date"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in model_ready.csv: {missing_columns}"
    )


# %%
# Remove missing values

df = df.dropna(
    subset=features + ["Flood_Risk"]
).copy()


# %%
# Check target classes

print("\nFlood Risk Classes:")
print(df["Flood_Risk"].value_counts().sort_index())


# %%
# Chronological train/test split

split_date = pd.Timestamp("2024-01-01")

train = df[df["Date"] < split_date].copy()
test = df[df["Date"] >= split_date].copy()

print("\nTraining data:", train.shape)
print("Testing data :", test.shape)


# %%
# Check train/test classes

print("\nTraining classes:")
print(sorted(train["Flood_Risk"].unique()))

print("\nTesting classes:")
print(sorted(test["Flood_Risk"].unique()))


# %%
# Prepare X and y

X_train = train[features]
y_train = train["Flood_Risk"]

X_test = test[features]
y_test = test["Flood_Risk"]


# %%
# Models

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# %%
# Train models

results = []
predictions = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    predictions[name] = pred

    p, r, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": p,
        "Recall": r,
        "F1": f1
    })


# %%
# Model comparison

results_df = pd.DataFrame(results)

print("\n================ MODEL RESULTS ================")
print(results_df)


# %%
# Random Forest evaluation

rf = models["Random Forest"]

rf_pred = predictions["Random Forest"]


# %%
# Classification report

print("\n========== RANDOM FOREST CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        rf_pred,
        labels=[0, 1, 2],
        target_names=[
            "Low",
            "Medium",
            "High"
        ],
        zero_division=0
    )
)


# %%
# Confusion matrix

cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=[0, 1, 2]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Low",
        "Medium",
        "High"
    ],
    yticklabels=[
        "Low",
        "Medium",
        "High"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()
plt.show()


# %%
# Feature importance

importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(
    ascending=False
)


plt.figure(figsize=(10, 5))

importance.plot(
    kind="bar"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Importance"
)

plt.tight_layout()
plt.show()


# %%
# Save model

model_path = MODEL_DIR / "flood_model.pkl"

joblib.dump(
    rf,
    model_path
)


# %%
# Save features

features_path = MODEL_DIR / "features.pkl"

joblib.dump(
    features,
    features_path
)


# %%
# Save model comparison

results_path = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)


# %%
# Final message

print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("Saved model:")
print(model_path)

print("\nSaved features:")
print(features_path)

print("\nSaved comparison:")
print(results_path)