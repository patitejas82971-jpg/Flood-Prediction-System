# %% [markdown]
# 01 - Data Collection
# Downloads daily historical weather data from Open-Meteo.
# Change latitude/longitude for your chosen study area.

# %%
from pathlib import Path
import requests
import pandas as pd

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Andhra Pradesh (Amaravati / Vijayawada) coordinates
latitude = 16.5062
longitude = 80.6480

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "precipitation_sum",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "pressure_msl_mean"
    ],
    "timezone": "auto"
}

url = "https://archive-api.open-meteo.com/v1/archive"
response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df = df.rename(columns={
    "time": "Date",
    "precipitation_sum": "Rainfall",
    "temperature_2m_mean": "Temperature",
    "relative_humidity_2m_mean": "Humidity",
    "wind_speed_10m_max": "Wind_Speed",
    "pressure_msl_mean": "Pressure"
})

out = RAW_DIR / "weather_data.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows to {out}")
print(df.head())
