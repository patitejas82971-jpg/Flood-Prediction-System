# %% [markdown]
# 04 - Feature Engineering
# Creates accumulated rainfall, lag variables, month, and a prototype target.
# Replace the prototype target with observed historical flood-event labels for
# a validated research project.

# %%
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

INPUT = PROJECT_ROOT / "data" / "processed" / "cleaned_weather.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"

df = pd.read_csv(INPUT)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

df["Rainfall_3d"] = df["Rainfall"].rolling(3).sum()
df["Rainfall_7d"] = df["Rainfall"].rolling(7).sum()
df["Rainfall_Lag1"] = df["Rainfall"].shift(1)
df["Rainfall_Lag2"] = df["Rainfall"].shift(2)
df["Month"] = df["Date"].dt.month
df = df.dropna().reset_index(drop=True)

def prototype_risk(row):
    if row["Rainfall_7d"] >= 250 or row["Rainfall_3d"] >= 150:
        return 2
    if row["Rainfall_7d"] >= 150 or row["Rainfall_3d"] >= 80:
        return 1
    return 0

df["Flood_Risk"] = df.apply(prototype_risk, axis=1)
df.to_csv(OUTPUT, index=False)

print("Model-ready shape:", df.shape)
print(df["Flood_Risk"].value_counts().sort_index())
print("NOTE: Flood_Risk is a prototype rainfall-risk label, not a validated flood-event label.")
