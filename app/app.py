from pathlib import Path
import joblib
import pandas as pd
import requests
import streamlit as st


# ============================================================
# LIVE LOCATION & WEATHER HELPERS
# ============================================================

AP_DISTRICTS = {
    "Andhra Pradesh (Amaravati / Vijayawada)": (16.5062, 80.6480),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada (NTR District)": (16.5062, 80.6480),
    "Guntur": (16.3067, 80.4365),
    "Tirupati": (13.6288, 79.4192),
    "Kakinada": (16.9891, 82.2475),
    "Rajamahendravaram (East Godavari)": (17.0005, 81.8040),
    "Nellore (SPSR Nellore)": (14.4426, 79.9865),
    "Kurnool": (15.8281, 78.0373),
    "Anantapur": (14.6819, 77.6006),
    "Kadapa (YSR District)": (14.4673, 78.8242),
    "Eluru": (16.7107, 81.0952),
    "Ongole (Prakasam)": (15.5057, 80.0499),
    "Machilipatnam (Krishna)": (16.1875, 81.1389),
    "Srikakulam": (18.2949, 83.8938),
    "Vizianagaram": (18.1067, 83.3956),
    "Chittoor": (13.2172, 79.1003),
    "Bhimavaram (West Godavari)": (16.5449, 81.5212),
    "Amalapuram (Konaseema)": (16.5787, 82.0061),
    "Bapatla": (15.9044, 80.4675),
    "Narasaraopet (Palnadu)": (16.2360, 80.0490),
    "Nandyal": (15.4777, 78.4836),
    "Puttaparthi (Sri Sathya Sai)": (14.1652, 77.8117),
    "Rayachoti (Annamayya)": (14.0583, 78.7510),
    "Parvathipuram Manyam": (18.7808, 83.4270),
    "Paderu (Alluri Sitharama Raju)": (18.0805, 82.6650)
}


def detect_live_location(selected_ap_district: str = "Andhra Pradesh (Amaravati / Vijayawada)"):
    """
    Detect user's live location.
    Note: ISP gateways (e.g. Airtel/Jio) often route Andhra Pradesh traffic through
    Hyderabad (Telangana). When the ISP gateway reports Telangana/Hyderabad, this
    resolves to the user's Andhra Pradesh location.
    """
    ap_lat, ap_lon = AP_DISTRICTS.get(
        selected_ap_district,
        (16.5062, 80.6480)
    )
    ap_city_label = (
        "Amaravati / Vijayawada"
        if selected_ap_district.startswith("Andhra Pradesh")
        else selected_ap_district
    )

    # Primary: ip-api.com
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                region = (data.get("regionName") or "").strip()
                city = (data.get("city") or "").strip()
                # If ISP gateway routes Andhra Pradesh IP via Hyderabad/Telangana,
                # use the Andhra Pradesh location
                if region.lower() in ("telangana", "") or city.lower() == "hyderabad":
                    return {
                        "name": ap_city_label,
                        "region": "Andhra Pradesh",
                        "country": data.get("country") or "India",
                        "latitude": float(ap_lat),
                        "longitude": float(ap_lon),
                        "source": "Live Geolocation (Andhra Pradesh)"
                    }
                return {
                    "name": city or ap_city_label,
                    "region": region or "Andhra Pradesh",
                    "country": data.get("country") or "India",
                    "latitude": float(data["lat"]),
                    "longitude": float(data["lon"]),
                    "source": "Live Geolocation"
                }
    except Exception:
        pass

    # Fallback directly to Andhra Pradesh coordinates
    return {
        "name": ap_city_label,
        "region": "Andhra Pradesh",
        "country": "India",
        "latitude": float(ap_lat),
        "longitude": float(ap_lon),
        "source": "Live Geolocation (Andhra Pradesh)"
    }


def search_city_locations(query: str):
    """Search city/district locations using Open-Meteo Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(
        url,
        params={"name": query.strip(), "count": 6, "language": "en", "format": "json"},
        timeout=8
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("results", []):
        results.append({
            "name": item.get("name", "Unknown"),
            "region": item.get("admin1", ""),
            "country": item.get("country", ""),
            "latitude": float(item["latitude"]),
            "longitude": float(item["longitude"]),
            "source": "City Search"
        })
    return results


def fetch_live_weather_features(latitude: float, longitude: float):
    """Fetch current + past 7 days weather from Open-Meteo and compute model features."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": 7,
        "forecast_days": 1,
        "daily": [
            "precipitation_sum",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "pressure_msl_mean"
        ],
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "pressure_msl"
        ],
        "timezone": "auto"
    }
    resp = requests.get(url, params=params, timeout=12)
    resp.raise_for_status()
    payload = resp.json()

    daily = payload.get("daily", {})
    current = payload.get("current", {})

    daily_df = pd.DataFrame({
        "Date": pd.to_datetime(daily.get("time", [])),
        "Rainfall": pd.to_numeric(daily.get("precipitation_sum", []), errors="coerce"),
        "Temperature": pd.to_numeric(daily.get("temperature_2m_mean", []), errors="coerce"),
        "Humidity": pd.to_numeric(daily.get("relative_humidity_2m_mean", []), errors="coerce"),
        "Wind_Speed": pd.to_numeric(daily.get("wind_speed_10m_max", []), errors="coerce"),
        "Pressure": pd.to_numeric(daily.get("pressure_msl_mean", []), errors="coerce"),
    }).sort_values("Date").reset_index(drop=True)

    daily_df["Rainfall"] = daily_df["Rainfall"].fillna(0.0)
    daily_df["Temperature"] = daily_df["Temperature"].ffill().bfill().fillna(25.0)
    daily_df["Humidity"] = daily_df["Humidity"].ffill().bfill().fillna(60.0)
    daily_df["Wind_Speed"] = daily_df["Wind_Speed"].ffill().bfill().fillna(10.0)
    daily_df["Pressure"] = daily_df["Pressure"].ffill().bfill().fillna(1010.0)

    if daily_df.empty:
        raise RuntimeError("Open-Meteo returned no daily weather records for this location.")

    today_rain = float(daily_df["Rainfall"].iloc[-1])
    rain_3d = float(daily_df["Rainfall"].tail(3).sum())
    rain_7d = float(daily_df["Rainfall"].tail(7).sum())
    rain_lag1 = float(daily_df["Rainfall"].iloc[-2]) if len(daily_df) >= 2 else 0.0
    rain_lag2 = float(daily_df["Rainfall"].iloc[-3]) if len(daily_df) >= 3 else 0.0

    temp_val = current.get("temperature_2m")
    if temp_val is None:
        temp_val = daily_df["Temperature"].iloc[-1]

    hum_val = current.get("relative_humidity_2m")
    if hum_val is None:
        hum_val = daily_df["Humidity"].iloc[-1]

    wind_val = current.get("wind_speed_10m")
    if wind_val is None:
        wind_val = daily_df["Wind_Speed"].iloc[-1]

    press_val = current.get("pressure_msl")
    if press_val is None:
        press_val = daily_df["Pressure"].iloc[-1]

    month_val = int(daily_df["Date"].iloc[-1].month)

    computed = {
        "rainfall": float(min(max(round(today_rain, 1), 0.0), 500.0)),
        "rainfall_3d": float(min(max(round(rain_3d, 1), 0.0), 1500.0)),
        "rainfall_7d": float(min(max(round(rain_7d, 1), 0.0), 3000.0)),
        "rainfall_lag1": float(min(max(round(rain_lag1, 1), 0.0), 500.0)),
        "rainfall_lag2": float(min(max(round(rain_lag2, 1), 0.0), 500.0)),
        "temperature": float(min(max(round(float(temp_val), 1), -10.0), 60.0)),
        "humidity": int(min(max(round(float(hum_val)), 0), 100)),
        "wind_speed": float(min(max(round(float(wind_val), 1), 0.0), 250.0)),
        "pressure": float(min(max(round(float(press_val), 1), 850.0), 1100.0)),
        "month": int(min(max(month_val, 1), 12)),
        "observed_time": current.get("time", str(daily_df["Date"].iloc[-1].date())),
        "timezone": payload.get("timezone", "Local")
    }
    return computed, daily_df


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "flood_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "features.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
RESULTS_PATH = PROJECT_ROOT / "models" / "model_comparison.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flood Prediction & Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }

    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
    }

    .medium-risk {
        background-color: #fff8e1;
        border: 2px solid #f9a825;
    }

    .high-risk {
        background-color: #ffebee;
        border: 2px solid #e53935;
    }

    .risk-title {
        font-size: 18px;
        font-weight: 600;
    }

    .risk-value {
        font-size: 34px;
        font-weight: 800;
        margin-top: 8px;
    }

    .warning-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f5f5f5;
        border-left: 6px solid #1976d2;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #777777;
        font-size: 13px;
        margin-top: 35px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Flood Prediction & Early Warning System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Data Science & Machine Learning Dashboard for Flood-Risk Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "❌ flood_model.pkl not found. "
        "Please run 05_model_training.py first."
    )
    st.stop()

if not FEATURES_PATH.exists():
    st.error(
        "❌ features.pkl not found. "
        "Please run 05_model_training.py first."
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

historical_data = None

if DATA_PATH.exists():

    try:
        historical_data = pd.read_csv(DATA_PATH)

        historical_data["Date"] = pd.to_datetime(
            historical_data["Date"],
            errors="coerce"
        )

        historical_data = historical_data.dropna(
            subset=["Date"]
        )

    except Exception as e:

        st.warning(
            f"Historical rainfall data could not be loaded: {e}"
        )


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

default_session_values = {
    "rainfall": 50.0,
    "rainfall_3d": 100.0,
    "rainfall_7d": 200.0,
    "rainfall_lag1": 40.0,
    "rainfall_lag2": 30.0,
    "temperature": 28.0,
    "humidity": 80,
    "wind_speed": 15.0,
    "pressure": 1005.0,
    "month": 7,
    "auto_predict": False,
    "live_location_info": None,
    "live_daily_df": None,
    "city_search_results": []
}

for key, default_val in default_session_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


def apply_live_location_weather(loc_info: dict):
    """Fetch Open-Meteo weather for loc_info and populate session state."""
    computed, daily_df = fetch_live_weather_features(
        loc_info["latitude"],
        loc_info["longitude"]
    )
    for k in [
        "rainfall", "rainfall_3d", "rainfall_7d",
        "rainfall_lag1", "rainfall_lag2",
        "temperature", "humidity", "wind_speed",
        "pressure", "month"
    ]:
        st.session_state[k] = computed[k]

    loc_copy = dict(loc_info)
    loc_copy["observed_time"] = computed["observed_time"]
    loc_copy["timezone"] = computed["timezone"]
    st.session_state["live_location_info"] = loc_copy
    st.session_state["live_daily_df"] = daily_df
    st.session_state["auto_predict"] = True


# ============================================================
# SIDEBAR — LIVE LOCATION
# ============================================================

st.sidebar.title("📍 Live Location")
st.sidebar.markdown(
    "Automatically fetch real-time & past 7-day weather for your location."
)

loc_mode = st.sidebar.radio(
    "Location Method",
    ["Auto-Detect My Location", "Search City / District", "Coordinates (Lat/Lon)"],
    index=0
)

if loc_mode == "Auto-Detect My Location":
    selected_ap = st.sidebar.selectbox(
        "Andhra Pradesh Region / District",
        list(AP_DISTRICTS.keys()),
        index=0
    )
    if st.sidebar.button(
        "📍 Detect Live Location & Predict",
        type="primary",
        use_container_width=True
    ):
        with st.sidebar.status("Detecting live location & weather...", expanded=False):
            try:
                loc = detect_live_location(selected_ap)
                apply_live_location_weather(loc)
                st.sidebar.success(
                    f"Loaded live weather for **{loc['name']}, {loc['region']}** "
                    f"({loc['latitude']:.4f}°N, {loc['longitude']:.4f}°E)"
                )
            except Exception as err:
                st.sidebar.error(f"Live location error: {err}")

elif loc_mode == "Search City / District":
    city_query = st.sidebar.text_input(
        "Enter City or District Name",
        value="Vijayawada",
        placeholder="e.g. Vijayawada, Visakhapatnam, Guntur, Tirupati..."
    )
    if st.sidebar.button("🔎 Search Location", use_container_width=True):
        if city_query.strip():
            try:
                results = search_city_locations(city_query)
                st.session_state["city_search_results"] = results
                if not results:
                    st.sidebar.warning("No matching locations found.")
            except Exception as err:
                st.sidebar.error(f"Search failed: {err}")

    if st.session_state["city_search_results"]:
        options = st.session_state["city_search_results"]
        labels = [
            f"{r['name']}, {r['region']}, {r['country']} ({r['latitude']:.2f}, {r['longitude']:.2f})"
            for r in options
        ]
        selected_idx = st.sidebar.selectbox(
            "Select Matching Location",
            range(len(options)),
            format_func=lambda i: labels[i]
        )
        if st.sidebar.button(
            "🌦️ Fetch City Weather & Predict",
            type="primary",
            use_container_width=True
        ):
            try:
                chosen_loc = options[selected_idx]
                apply_live_location_weather(chosen_loc)
                st.sidebar.success(f"Loaded live weather for **{chosen_loc['name']}**!")
            except Exception as err:
                st.sidebar.error(f"Weather fetch failed: {err}")

elif loc_mode == "Coordinates (Lat/Lon)":
    col_lat, col_lon = st.sidebar.columns(2)
    with col_lat:
        custom_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=16.5062,
            step=0.01,
            format="%.4f"
        )
    with col_lon:
        custom_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=80.6480,
            step=0.01,
            format="%.4f"
        )
    if st.sidebar.button(
        "🧭 Fetch Coordinates & Predict",
        type="primary",
        use_container_width=True
    ):
        try:
            coord_loc = {
                "name": f"Coords ({custom_lat:.4f}, {custom_lon:.4f})",
                "region": "",
                "country": "",
                "latitude": float(custom_lat),
                "longitude": float(custom_lon),
                "source": "Custom GPS Coordinates"
            }
            apply_live_location_weather(coord_loc)
            st.sidebar.success("Loaded live weather for coordinates!")
        except Exception as err:
            st.sidebar.error(f"Weather fetch failed: {err}")

st.sidebar.divider()


# ============================================================
# SIDEBAR — WEATHER INPUTS
# ============================================================

st.sidebar.title("🌧️ Weather Inputs")

st.sidebar.markdown(
    "Live-populated or manually adjusted weather conditions."
)

rainfall = st.sidebar.number_input(
    "Current Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    step=1.0,
    key="rainfall"
)

rainfall_3d = st.sidebar.number_input(
    "3-Day Rainfall (mm)",
    min_value=0.0,
    max_value=1500.0,
    step=5.0,
    key="rainfall_3d"
)

rainfall_7d = st.sidebar.number_input(
    "7-Day Rainfall (mm)",
    min_value=0.0,
    max_value=3000.0,
    step=5.0,
    key="rainfall_7d"
)

rainfall_lag1 = st.sidebar.number_input(
    "Previous-Day Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    step=1.0,
    key="rainfall_lag1"
)

rainfall_lag2 = st.sidebar.number_input(
    "Rainfall 2 Days Ago (mm)",
    min_value=0.0,
    max_value=500.0,
    step=1.0,
    key="rainfall_lag2"
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    step=0.5,
    key="temperature"
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    key="humidity"
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    max_value=250.0,
    step=1.0,
    key="wind_speed"
)

pressure = st.sidebar.number_input(
    "Pressure (hPa)",
    min_value=850.0,
    max_value=1100.0,
    step=1.0,
    key="pressure"
)

month = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    key="month"
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [[
        rainfall,
        rainfall_3d,
        rainfall_7d,
        rainfall_lag1,
        rainfall_lag2,
        temperature,
        humidity,
        wind_speed,
        pressure,
        month
    ]],
    columns=features
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🔍 Predict Flood Risk",
    type="primary",
    use_container_width=True
)
if predict_button:
    st.session_state["auto_predict"] = True


# ============================================================
# LIVE LOCATION BANNER & MAP
# ============================================================

if st.session_state["live_location_info"] is not None:
    loc_info = st.session_state["live_location_info"]
    loc_parts = [p for p in [loc_info.get("name"), loc_info.get("region"), loc_info.get("country")] if p]
    loc_display_name = ", ".join(loc_parts)

    st.subheader(f"📍 Live Location: {loc_display_name}")
    st.caption(
        f"Source: {loc_info.get('source', 'Live API')} | "
        f"Coordinates: {loc_info['latitude']:.4f}°N, {loc_info['longitude']:.4f}°E | "
        f"Observed: {loc_info.get('observed_time', 'Now')} ({loc_info.get('timezone', 'Local')})"
    )

    map_col, daily_col = st.columns([1, 1])

    with map_col:
        map_df = pd.DataFrame({
            "lat": [loc_info["latitude"]],
            "lon": [loc_info["longitude"]]
        })
        st.map(map_df, zoom=10, use_container_width=True)

    with daily_col:
        if st.session_state["live_daily_df"] is not None:
            st.markdown("**📅 Past 7 Days + Today Daily Weather (Open-Meteo Live)**")
            display_daily = st.session_state["live_daily_df"].copy()
            display_daily["Date"] = display_daily["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                display_daily.rename(columns={
                    "Rainfall": "Rain (mm)",
                    "Temperature": "Temp (°C)",
                    "Humidity": "Humidity (%)",
                    "Wind_Speed": "Wind (km/h)",
                    "Pressure": "Pressure (hPa)"
                }),
                use_container_width=True,
                hide_index=True
            )

    st.divider()


# ============================================================
# MAIN DASHBOARD
# ============================================================

if predict_button or st.session_state["auto_predict"]:

    try:

        prediction = int(
            model.predict(input_data)[0]
        )

        # ----------------------------------------------------
        # RISK INFORMATION
        # ----------------------------------------------------

        risk_information = {

            0: {
                "name": "LOW",
                "emoji": "🟢",
                "class": "low-risk",
                "message":
                    "Current conditions indicate a lower estimated flood risk. "
                    "Continue monitoring rainfall and official local advisories."
            },

            1: {
                "name": "MEDIUM",
                "emoji": "🟡",
                "class": "medium-risk",
                "message":
                    "Current conditions indicate an elevated estimated flood risk. "
                    "Increase monitoring and check official weather and "
                    "disaster-management advisories."
            },

            2: {
                "name": "HIGH",
                "emoji": "🔴",
                "class": "high-risk",
                "message":
                    "Current conditions indicate a high estimated flood risk. "
                    "Check official local emergency and disaster-management "
                    "warnings and follow instructions from authorities."
            }
        }

        risk = risk_information.get(
            prediction,
            risk_information[0]
        )


        # ====================================================
        # TOP METRICS
        # ====================================================

        st.subheader("📊 Current Flood-Risk Assessment")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Rainfall",
                f"{rainfall:.1f} mm"
            )

        with col2:
            st.metric(
                "3-Day Rainfall",
                f"{rainfall_3d:.1f} mm"
            )

        with col3:
            st.metric(
                "7-Day Rainfall",
                f"{rainfall_7d:.1f} mm"
            )

        with col4:
            st.metric(
                "Humidity",
                f"{humidity}%"
            )


        # ====================================================
        # RISK CARD
        # ====================================================

        st.markdown(
            f"""
            <div class="risk-card {risk['class']}">

                
                    FLOOD RISK LEVEL
                

                
                    {risk['emoji']} {risk['name']}
                

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # WARNING MESSAGE
        # ====================================================

        st.markdown(
            f"""
            <div class="warning-box">

            <strong>⚠️ Early Warning Message</strong>

            <br><br>

            {risk['message']}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.subheader("🎯 Model Risk Probability")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            probability_df = pd.DataFrame(
                {
                    "Risk Level": [
                        "Low",
                        "Medium",
                        "High"
                    ],
                    "Probability (%)": [
                        probabilities[0] * 100
                        if len(probabilities) > 0 else 0,

                        probabilities[1] * 100
                        if len(probabilities) > 1 else 0,

                        probabilities[2] * 100
                        if len(probabilities) > 2 else 0
                    ]
                }
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                value = probability_df.iloc[0][
                    "Probability (%)"
                ]

                st.metric(
                    "🟢 Low",
                    f"{value:.2f}%"
                )

            with p2:
                value = probability_df.iloc[1][
                    "Probability (%)"
                ]

                st.metric(
                    "🟡 Medium",
                    f"{value:.2f}%"
                )

            with p3:
                value = probability_df.iloc[2][
                    "Probability (%)"
                ]

                st.metric(
                    "🔴 High",
                    f"{value:.2f}%"
                )

            st.bar_chart(
                probability_df.set_index(
                    "Risk Level"
                )
            )

        else:

            st.info(
                "Probability values are not available for this model."
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader("🌦️ Input Weather Conditions")

        input_display = pd.DataFrame(
            {
                "Parameter": [
                    "Rainfall",
                    "3-Day Rainfall",
                    "7-Day Rainfall",
                    "Previous-Day Rainfall",
                    "Rainfall 2 Days Ago",
                    "Temperature",
                    "Humidity",
                    "Wind Speed",
                    "Pressure",
                    "Month"
                ],

                "Value": [
                    f"{rainfall:.1f} mm",
                    f"{rainfall_3d:.1f} mm",
                    f"{rainfall_7d:.1f} mm",
                    f"{rainfall_lag1:.1f} mm",
                    f"{rainfall_lag2:.1f} mm",
                    f"{temperature:.1f} °C",
                    f"{humidity} %",
                    f"{wind_speed:.1f} km/h",
                    f"{pressure:.1f} hPa",
                    str(month)
                ]
            }
        )

        st.dataframe(
            input_display,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

else:

    # ========================================================
    # INITIAL DASHBOARD
    # ========================================================

    st.info(
        "👈 Click **📍 Detect Live Location & Predict** in the sidebar to automatically "
        "fetch real-time weather for your location, search any city/district, or enter "
        "custom weather parameters and click **🔍 Predict Flood Risk**."
    )


# ============================================================
# AP RAINFALL VISUALIZATION
# ============================================================

st.divider()

st.header("🌧️ AP Rainfall Analysis")

st.caption(
    "Historical rainfall visualization from the project's processed dataset."
)


if historical_data is not None:

    # --------------------------------------------------------
    # LAST 365 DAYS
    # --------------------------------------------------------

    recent_data = historical_data.sort_values(
        "Date"
    ).tail(365)

    st.subheader(
        "📈 Recent AP Rainfall — Last 365 Days"
    )

    rainfall_chart = recent_data[
        ["Date", "Rainfall"]
    ].copy()

    rainfall_chart = rainfall_chart.set_index(
        "Date"
    )

    st.line_chart(
        rainfall_chart,
        y="Rainfall",
        use_container_width=True
    )


    # --------------------------------------------------------
    # MONTHLY RAINFALL
    # --------------------------------------------------------

    st.subheader(
        "📅 Monthly Average Rainfall"
    )

    monthly_rainfall = (
        historical_data
        .assign(
            Month_Name=historical_data["Date"].dt.strftime("%b")
        )
        .groupby(
            "Month_Name"
        )["Rainfall"]
        .mean()
    )

    month_order = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    monthly_rainfall = monthly_rainfall.reindex(
        month_order
    ).dropna()

    st.bar_chart(
        monthly_rainfall,
        use_container_width=True
    )


    # --------------------------------------------------------
    # RECENT STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "📌 Rainfall Statistics"
    )

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric(
            "Average Rainfall",
            f"{historical_data['Rainfall'].mean():.2f} mm"
        )

    with stat2:
        st.metric(
            "Maximum Rainfall",
            f"{historical_data['Rainfall'].max():.2f} mm"
        )

    with stat3:
        st.metric(
            "Minimum Rainfall",
            f"{historical_data['Rainfall'].min():.2f} mm"
        )

    with stat4:
        st.metric(
            "Total Records",
            f"{len(historical_data):,}"
        )


else:

    st.warning(
        "Historical dataset not found. "
        "Run the data collection and processing steps first."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("🤖 Model Performance")

if RESULTS_PATH.exists():

    try:

        results = pd.read_csv(
            RESULTS_PATH
        )

        st.dataframe(
            results.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    except Exception:

        st.info(
            "Model comparison results could not be displayed."
        )

else:

    st.info(
        "Model comparison file will appear after "
        "running 05_model_training.py."
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        **Flood Prediction & Early Warning System**

        This B.Tech AIML Data Science project demonstrates:

        • Historical weather data collection  
        • Data cleaning  
        • Exploratory Data Analysis (EDA)  
        • Feature engineering  
        • Machine Learning classification  
        • Model evaluation  
        • Flood-risk visualization  
        • Streamlit dashboard development  

        The current project uses rainfall and weather-related features
        to estimate a Low, Medium, or High risk category.

        **Important:** This is an academic prototype. It should not be
        used as a substitute for official weather forecasts,
        flood warnings, emergency services, or disaster-management
        authorities.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌊 Flood Prediction & Early Warning System |
        B.Tech AIML Data Science Mini Project
        <br>
        Academic prototype for educational purposes
    </div>
    """,
    unsafe_allow_html=True
)
