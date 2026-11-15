# 🌊 Flood Prediction & Early Warning System

A Data Science & Machine Learning web dashboard for estimating flood-risk levels (**Low**, **Medium**, **High**) across **Andhra Pradesh** and custom locations using real-time and historical weather data from **Open-Meteo**.

## ✨ Features
- **📍 Live Location Flood Prediction:**
  - **Auto-Detect Location (Andhra Pradesh):** Supports all 26 Andhra Pradesh districts with one-click live weather retrieval.
  - **Search Any City / District:** Geocodes any city or town via the Open-Meteo Geocoding API.
  - **Custom GPS Coordinates:** Supports manual Latitude & Longitude entry.
  - **Live Map & 7-Day Breakdown:** Displays an interactive location map and the past 7 days + today's daily rainfall and weather records.
- **🤖 Machine Learning Classification:**
  - Compares Logistic Regression, Random Forest, and Gradient Boosting models on chronological train/test splits.
  - Uses 10 meteorological features: `Rainfall`, `Rainfall_3d`, `Rainfall_7d`, `Rainfall_Lag1`, `Rainfall_Lag2`, `Temperature`, `Humidity`, `Wind_Speed`, `Pressure`, and `Month`.
- **📊 Historical AP Rainfall Analysis:**
  - Interactive charts for the last 365 days of rainfall, monthly averages, and summary statistics.

## 📁 Project Structure
```text
├── app/
│   └── app.py                  # Streamlit web dashboard
├── data/
│   ├── raw/                    # Raw historical weather data (weather_data.csv)
│   └── processed/              # Cleaned & feature-engineered datasets
├── models/
│   ├── flood_model.pkl         # Trained Random Forest classifier
│   ├── features.pkl            # Model feature list
│   └── model_comparison.csv    # Evaluation metrics across models
├── notebooks/
│   ├── data_collection.py      # 01 - Open-Meteo historical data download
│   ├── data_cleaning.py        # 02 - Data cleaning & interpolation
│   ├── eda.py                  # 03 - Exploratory Data Analysis plots
│   ├── feature_engineering.py  # 04 - Rolling/lag features & prototype labels
│   └── model_training.py       # 05 - Model training & evaluation
├── src/
│   ├── preprocessing.py        # Reusable data cleaning & feature helpers
│   └── prediction.py           # Model loading & inference helpers
└── requirements.txt            # Python dependencies
```

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app/app.py
   ```

3. **(Optional) Re-run the full data & training pipeline:**
   ```bash
   python notebooks/data_collection.py
   python notebooks/data_cleaning.py
   python notebooks/feature_engineering.py
   python notebooks/model_training.py
   ```
