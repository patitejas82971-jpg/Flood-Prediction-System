import pandas as pd

FEATURES = [
    "Rainfall", "Rainfall_3d", "Rainfall_7d",
    "Rainfall_Lag1", "Rainfall_Lag2",
    "Temperature", "Humidity", "Wind_Speed",
    "Pressure", "Month"
]

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric = ["Rainfall", "Temperature", "Humidity", "Wind_Speed", "Pressure"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates().dropna(subset=["Date"])
    for col in numeric:
        df[col] = df[col].interpolate().ffill().bfill()
    return df.sort_values("Date").reset_index(drop=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    df["Rainfall_3d"] = df["Rainfall"].rolling(3).sum()
    df["Rainfall_7d"] = df["Rainfall"].rolling(7).sum()
    df["Rainfall_Lag1"] = df["Rainfall"].shift(1)
    df["Rainfall_Lag2"] = df["Rainfall"].shift(2)
    df["Month"] = df["Date"].dt.month
    return df.dropna().reset_index(drop=True)

def add_prototype_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    # Educational prototype label; replace with observed flood-event labels.
    df = df.copy()
    def risk(row):
        if row["Rainfall_7d"] >= 250 or row["Rainfall_3d"] >= 150:
            return 2
        if row["Rainfall_7d"] >= 150 or row["Rainfall_3d"] >= 80:
            return 1
        return 0
    df["Flood_Risk"] = df.apply(risk, axis=1)
    return df
