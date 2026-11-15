# %% [markdown]
# 03 - Exploratory Data Analysis

# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cleaned_weather.csv")
df["Date"] = pd.to_datetime(df["Date"])

print(df.describe())

# %%
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Rainfall"])
plt.title("Daily Rainfall Over Time")
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(9, 5))
sns.histplot(df["Rainfall"], bins=30, kde=True)
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(9, 6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# %%
monthly = df.groupby(df["Date"].dt.month)["Rainfall"].mean()
plt.figure(figsize=(9, 5))
monthly.plot(kind="bar")
plt.title("Average Rainfall by Month")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.tight_layout()
plt.show()
