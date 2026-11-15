# %% [markdown]
# 02 - Data Cleaning

# %%
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW = PROJECT_ROOT / "data" / "raw" / "weather_data.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
numeric = ["Rainfall", "Temperature", "Humidity", "Wind_Speed", "Pressure"]

for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates().dropna(subset=["Date"])

for col in numeric:
    df[col] = df[col].interpolate().ffill().bfill()

df = df.sort_values("Date").reset_index(drop=True)

out = PROCESSED / "cleaned_weather.csv"
df.to_csv(out, index=False)

print("Cleaned shape:", df.shape)
print("Saved:", out)
