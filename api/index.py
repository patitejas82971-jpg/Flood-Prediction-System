from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "models" / "vercel_bundle.json"

_BUNDLE_CACHE = None


def get_bundle():
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is None:
        with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
            _BUNDLE_CACHE = json.load(f)
    return _BUNDLE_CACHE


def predict_from_trees(values):
    bundle = get_bundle()
    trees = bundle["trees"]
    classes = bundle["classes"]
    n_classes = len(classes)

    avg_probs = [0.0] * n_classes
    for tree in trees:
        left = tree["left"]
        right = tree["right"]
        feat = tree["feature"]
        thresh = tree["threshold"]
        val = tree["value"]

        node = 0
        while left[node] != -1:
            f_idx = feat[node]
            if values[f_idx] <= thresh[node]:
                node = left[node]
            else:
                node = right[node]
        node_probs = val[node]
        for c_idx in range(n_classes):
            avg_probs[c_idx] += node_probs[c_idx]

    n_trees = float(len(trees)) if trees else 1.0
    avg_probs = [p / n_trees for p in avg_probs]

    # Map to 3 classes [Low (0), Medium (1), High (2)]
    prob_3 = [0.0, 0.0, 0.0]
    for idx, cls_val in enumerate(classes):
        if 0 <= cls_val <= 2:
            prob_3[cls_val] = avg_probs[idx]

    # Check high-risk threshold rule from feature_engineering.py if training split lacked class 2
    rain_3d = values[1]
    rain_7d = values[2]
    if 2 not in classes and (rain_7d >= 250.0 or rain_3d >= 150.0):
        high_p = min(0.92, max(0.65, prob_3[1]))
        med_p = round((1.0 - high_p) * 0.7, 4)
        low_p = round(max(0.0, 1.0 - high_p - med_p), 4)
        prob_3 = [low_p, med_p, high_p]

    pred_class = int( max(range(3), key=lambda i: prob_3[i]) )
    return pred_class, [round(p * 100.0, 2) for p in prob_3]


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>🌊 Flood Prediction & Early Warning System</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #ffffff;
      --sidebar-bg: #f8f9fb;
      --border: #e5e7eb;
      --text: #111827;
      --muted: #6b7280;
      --primary: #ff4b4b;
      --blue: #1976d2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: var(--text);
      background: var(--bg);
      display: flex;
      min-height: 100vh;
    }
    .sidebar {
      width: 340px;
      min-width: 340px;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border);
      padding: 22px 18px;
      overflow-y: auto;
      max-height: 100vh;
      position: sticky;
      top: 0;
    }
    .main {
      flex: 1;
      padding: 32px 42px;
      max-width: 1200px;
      margin: 0 auto;
      overflow-x: hidden;
    }
    @media (max-width: 900px) {
      body { flex-direction: column; }
      .sidebar { width: 100%; min-width: 100%; max-height: none; position: static; }
      .main { padding: 20px; }
    }
    .main-title { font-size: 34px; font-weight: 700; margin-bottom: 4px; }
    .subtitle { font-size: 16px; color: var(--muted); margin-bottom: 24px; }
    .sidebar h2 { font-size: 20px; margin: 0 0 6px 0; }
    .sidebar p { font-size: 13px; color: var(--muted); margin: 0 0 14px 0; }
    .divider { border: 0; border-top: 1px solid var(--border); margin: 18px 0; }
    .field { margin-bottom: 12px; }
    .field label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 4px; }
    .field input, .field select {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font-size: 14px;
      background: #fff;
    }
    .btn {
      width: 100%;
      padding: 10px 14px;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s;
      margin-top: 6px;
    }
    .btn:hover { opacity: 0.92; }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-secondary { background: #e5e7eb; color: #1f2937; }
    .radio-group { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; font-size: 13.5px; }
    .radio-group label { display: flex; align-items: center; gap: 8px; cursor: pointer; }
    .status-msg {
      margin-top: 10px;
      padding: 9px 12px;
      border-radius: 8px;
      font-size: 13px;
      display: none;
    }
    .status-ok { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; display: block; }
    .status-err { background: #ffebee; color: #b71c1c; border: 1px solid #ef9a9a; display: block; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 18px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
    @media (max-width: 768px) {
      .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; }
    }
    .metric-card {
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fafafa;
    }
    .metric-label { font-size: 13px; color: var(--muted); }
    .metric-val { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .risk-card {
      padding: 24px;
      border-radius: 15px;
      text-align: center;
      margin-bottom: 18px;
      font-weight: 700;
      font-size: 24px;
    }
    .low-risk { background-color: #e8f5e9; border: 2px solid #43a047; }
    .medium-risk { background-color: #fff8e1; border: 2px solid #f9a825; }
    .high-risk { background-color: #ffebee; border: 2px solid #e53935; }
    .warning-box {
      padding: 18px;
      border-radius: 12px;
      background-color: #f5f5f5;
      border-left: 6px solid var(--blue);
      margin-bottom: 22px;
      line-height: 1.5;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13.5px; margin-bottom: 18px; }
    th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--border); }
    th { background: #f3f4f6; font-weight: 600; }
    #map { height: 290px; width: 100%; border-radius: 10px; border: 1px solid var(--border); }
    .footer { text-align: center; color: #777; font-size: 13px; margin-top: 35px; padding: 15px; }
  </style>
</head>
<body>
  <aside class="sidebar">
    <h2>📍 Live Location</h2>
    <p>Automatically fetch real-time & past 7-day weather for your location.</p>

    <div class="radio-group">
      <label><input type="radio" name="loc_mode" value="ap" checked onchange="switchLocMode('ap')"> Auto-Detect My Location</label>
      <label><input type="radio" name="loc_mode" value="city" onchange="switchLocMode('city')"> Search City / District</label>
      <label><input type="radio" name="loc_mode" value="coords" onchange="switchLocMode('coords')"> Coordinates (Lat/Lon)</label>
    </div>

    <div id="mode-ap">
      <div class="field">
        <label>Andhra Pradesh Region / District</label>
        <select id="ap-district"></select>
      </div>
      <button class="btn btn-primary" onclick="detectAndPredictAP()">📍 Detect Live Location & Predict</button>
    </div>

    <div id="mode-city" style="display:none;">
      <div class="field">
        <label>Enter City or District Name</label>
        <input type="text" id="city-query" value="Vijayawada" placeholder="e.g. Vijayawada, Visakhapatnam..." />
      </div>
      <button class="btn btn-secondary" onclick="searchCity()">🔎 Search Location</button>
      <div id="city-results-wrap" style="display:none; margin-top:10px;">
        <div class="field">
          <label>Select Matching Location</label>
          <select id="city-select"></select>
        </div>
        <button class="btn btn-primary" onclick="fetchSelectedCity()">🌦️ Fetch City Weather & Predict</button>
      </div>
    </div>

    <div id="mode-coords" style="display:none;">
      <div class="grid-2" style="gap:8px; margin-bottom:6px;">
        <div class="field">
          <label>Latitude</label>
          <input type="number" id="custom-lat" value="16.5062" step="0.01" />
        </div>
        <div class="field">
          <label>Longitude</label>
          <input type="number" id="custom-lon" value="80.6480" step="0.01" />
        </div>
      </div>
      <button class="btn btn-primary" onclick="fetchCoordsWeather()">🧭 Fetch Coordinates & Predict</button>
    </div>

    <div id="loc-status" class="status-msg"></div>

    <hr class="divider" />

    <h2>🌧️ Weather Inputs</h2>
    <p>Live-populated or manually adjusted weather conditions.</p>

    <div class="field"><label>Current Rainfall (mm)</label><input type="number" id="in_rain" value="50.0" step="1" min="0" max="500"></div>
    <div class="field"><label>3-Day Rainfall (mm)</label><input type="number" id="in_rain3" value="100.0" step="5" min="0" max="1500"></div>
    <div class="field"><label>7-Day Rainfall (mm)</label><input type="number" id="in_rain7" value="200.0" step="5" min="0" max="3000"></div>
    <div class="field"><label>Previous-Day Rainfall (mm)</label><input type="number" id="in_lag1" value="40.0" step="1" min="0" max="500"></div>
    <div class="field"><label>Rainfall 2 Days Ago (mm)</label><input type="number" id="in_lag2" value="30.0" step="1" min="0" max="500"></div>
    <div class="field"><label>Temperature (°C)</label><input type="number" id="in_temp" value="28.0" step="0.5" min="-10" max="60"></div>
    <div class="field"><label>Humidity (%): <span id="hum_val">80</span>%</label><input type="range" id="in_hum" value="80" min="0" max="100" oninput="document.getElementById('hum_val').innerText=this.value"></div>
    <div class="field"><label>Wind Speed (km/h)</label><input type="number" id="in_wind" value="15.0" step="1" min="0" max="250"></div>
    <div class="field"><label>Pressure (hPa)</label><input type="number" id="in_press" value="1005.0" step="1" min="850" max="1100"></div>
    <div class="field"><label>Month: <span id="month_val">7</span></label><input type="range" id="in_month" value="7" min="1" max="12" oninput="document.getElementById('month_val').innerText=this.value"></div>

    <button class="btn btn-primary" onclick="runPrediction()">🔍 Predict Flood Risk</button>
  </aside>

  <main class="main">
    <div class="main-title">🌊 Flood Prediction & Early Warning System</div>
    <div class="subtitle">Data Science & Machine Learning Dashboard for Flood-Risk Analysis</div>

    <div id="live-loc-section" style="display:none;">
      <h2 id="live-loc-title" style="margin-bottom:4px;">📍 Live Location</h2>
      <p id="live-loc-caption" style="color:var(--muted); font-size:13.5px; margin-top:0; margin-bottom:14px;"></p>
      <div class="grid-2">
        <div><div id="map"></div></div>
        <div>
          <div style="font-weight:600; margin-bottom:8px; font-size:14px;">📅 Past 7 Days + Today Daily Weather (Open-Meteo Live)</div>
          <div style="max-height:265px; overflow-y:auto; border:1px solid var(--border); border-radius:8px;">
            <table style="margin:0;">
              <thead>
                <tr><th>Date</th><th>Rain (mm)</th><th>Temp (°C)</th><th>Humidity (%)</th><th>Wind (km/h)</th><th>Pressure (hPa)</th></tr>
              </thead>
              <tbody id="live-daily-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
      <hr class="divider" />
    </div>

    <div id="initial-banner" class="warning-box" style="border-left-color:#0284c7; background:#f0f9ff;">
      👈 Click <strong>📍 Detect Live Location & Predict</strong> in the sidebar to automatically fetch real-time weather for your Andhra Pradesh location, search any city/district, or enter custom weather parameters and click <strong>🔍 Predict Flood Risk</strong>.
    </div>

    <div id="prediction-section" style="display:none;">
      <h2>📊 Current Flood-Risk Assessment</h2>
      <div class="grid-4">
        <div class="metric-card"><div class="metric-label">Current Rainfall</div><div class="metric-val" id="m_rain">0.0 mm</div></div>
        <div class="metric-card"><div class="metric-label">3-Day Rainfall</div><div class="metric-val" id="m_rain3">0.0 mm</div></div>
        <div class="metric-card"><div class="metric-label">7-Day Rainfall</div><div class="metric-val" id="m_rain7">0.0 mm</div></div>
        <div class="metric-card"><div class="metric-label">Humidity</div><div class="metric-val" id="m_hum">0%</div></div>
      </div>

      <div id="risk-card" class="risk-card low-risk">
        <div style="font-size:16px; font-weight:600; margin-bottom:6px;">FLOOD RISK LEVEL</div>
        <div id="risk-badge" style="font-size:32px; font-weight:800;">🟢 LOW</div>
      </div>

      <div class="warning-box">
        <strong>⚠️ Early Warning Message</strong><br><br>
        <span id="risk-msg"></span>
      </div>

      <h2>🎯 Model Risk Probability</h2>
      <div class="grid-3">
        <div class="metric-card"><div class="metric-label">🟢 Low</div><div class="metric-val" id="p_low">0.00%</div></div>
        <div class="metric-card"><div class="metric-label">🟡 Medium</div><div class="metric-val" id="p_med">0.00%</div></div>
        <div class="metric-card"><div class="metric-label">🔴 High</div><div class="metric-val" id="p_high">0.00%</div></div>
      </div>
      <div style="height:220px; margin-bottom:24px;"><canvas id="probChart"></canvas></div>

      <h2>🌦️ Input Weather Conditions</h2>
      <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody id="input-summary-tbody"></tbody>
      </table>
    </div>

    <hr class="divider" />
    <h2>🌧️ AP Rainfall Analysis</h2>
    <p style="color:var(--muted); font-size:14px; margin-top:-8px;">Historical rainfall visualization from the project's processed dataset.</p>

    <h3>📈 Recent AP Rainfall — Last 365 Days</h3>
    <div style="height:250px; margin-bottom:24px;"><canvas id="recentChart"></canvas></div>

    <h3>📅 Monthly Average Rainfall</h3>
    <div style="height:240px; margin-bottom:24px;"><canvas id="monthlyChart"></canvas></div>

    <h3>📌 Rainfall Statistics</h3>
    <div class="grid-4">
      <div class="metric-card"><div class="metric-label">Average Rainfall</div><div class="metric-val" id="st_avg">--</div></div>
      <div class="metric-card"><div class="metric-label">Maximum Rainfall</div><div class="metric-val" id="st_max">--</div></div>
      <div class="metric-card"><div class="metric-label">Minimum Rainfall</div><div class="metric-val" id="st_min">--</div></div>
      <div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-val" id="st_cnt">--</div></div>
    </div>

    <hr class="divider" />
    <h2>🤖 Model Performance</h2>
    <table>
      <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody id="model-comp-tbody"></tbody>
    </table>

    <div class="footer">
      🌊 Flood Prediction & Early Warning System | B.Tech AIML Data Science Mini Project<br>
      Academic prototype for educational purposes
    </div>
  </main>

  <script>
    const AP_DISTRICTS = {
      "Andhra Pradesh (Amaravati / Vijayawada)": [16.5062, 80.6480],
      "Visakhapatnam": [17.6868, 83.2185],
      "Vijayawada (NTR District)": [16.5062, 80.6480],
      "Guntur": [16.3067, 80.4365],
      "Tirupati": [13.6288, 79.4192],
      "Kakinada": [16.9891, 82.2475],
      "Rajamahendravaram (East Godavari)": [17.0005, 81.8040],
      "Nellore (SPSR Nellore)": [14.4426, 79.9865],
      "Kurnool": [15.8281, 78.0373],
      "Anantapur": [14.6819, 77.6006],
      "Kadapa (YSR District)": [14.4673, 78.8242],
      "Eluru": [16.7107, 81.0952],
      "Ongole (Prakasam)": [15.5057, 80.0499],
      "Machilipatnam (Krishna)": [16.1875, 81.1389],
      "Srikakulam": [18.2949, 83.8938],
      "Vizianagaram": [18.1067, 83.3956],
      "Chittoor": [13.2172, 79.1003],
      "Bhimavaram (West Godavari)": [16.5449, 81.5212],
      "Amalapuram (Konaseema)": [16.5787, 82.0061],
      "Bapatla": [15.9044, 80.4675],
      "Narasaraopet (Palnadu)": [16.2360, 80.0490],
      "Nandyal": [15.4777, 78.4836],
      "Puttaparthi (Sri Sathya Sai)": [14.1652, 77.8117],
      "Rayachoti (Annamayya)": [14.0583, 78.7510],
      "Parvathipuram Manyam": [18.7808, 83.4270],
      "Paderu (Alluri Sitharama Raju)": [18.0805, 82.6650]
    };

    const apSelect = document.getElementById("ap-district");
    Object.keys(AP_DISTRICTS).forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      apSelect.appendChild(opt);
    });

    let mapInstance = null;
    let mapMarker = null;
    let probChartInstance = null;
    let citySearchResults = [];

    function switchLocMode(mode) {
      document.getElementById("mode-ap").style.display = mode === "ap" ? "block" : "none";
      document.getElementById("mode-city").style.display = mode === "city" ? "block" : "none";
      document.getElementById("mode-coords").style.display = mode === "coords" ? "block" : "none";
    }

    function setStatus(msg, isErr = false) {
      const el = document.getElementById("loc-status");
      el.className = "status-msg " + (isErr ? "status-err" : "status-ok");
      el.innerHTML = msg;
    }

    async function fetchOpenMeteoAndPredict(locInfo) {
      setStatus("Fetching live weather from Open-Meteo...");
      try {
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${locInfo.latitude}&longitude=${locInfo.longitude}&past_days=7&forecast_days=1&daily=precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,pressure_msl_mean&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl&timezone=auto`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error("Open-Meteo request failed");
        const data = await resp.json();

        const daily = data.daily || {};
        const current = data.current || {};
        const times = daily.time || [];
        const rains = (daily.precipitation_sum || []).map(v => Number(v || 0));
        const temps = (daily.temperature_2m_mean || []).map(v => Number(v ?? 25));
        const hums = (daily.relative_humidity_2m_mean || []).map(v => Number(v ?? 60));
        const winds = (daily.wind_speed_10m_max || []).map(v => Number(v ?? 10));
        const press = (daily.pressure_msl_mean || []).map(v => Number(v ?? 1010));

        const n = rains.length;
        const todayRain = n > 0 ? rains[n - 1] : 0;
        const rain3d = rains.slice(Math.max(0, n - 3)).reduce((a, b) => a + b, 0);
        const rain7d = rains.slice(Math.max(0, n - 7)).reduce((a, b) => a + b, 0);
        const lag1 = n >= 2 ? rains[n - 2] : 0;
        const lag2 = n >= 3 ? rains[n - 3] : 0;

        const tempVal = current.temperature_2m ?? (n > 0 ? temps[n - 1] : 28);
        const humVal = Math.round(current.relative_humidity_2m ?? (n > 0 ? hums[n - 1] : 80));
        const windVal = current.wind_speed_10m ?? (n > 0 ? winds[n - 1] : 15);
        const pressVal = current.pressure_msl ?? (n > 0 ? press[n - 1] : 1005);
        const monthVal = times.length > 0 ? (new Date(times[times.length - 1]).getMonth() + 1) : 7;

        document.getElementById("in_rain").value = todayRain.toFixed(1);
        document.getElementById("in_rain3").value = rain3d.toFixed(1);
        document.getElementById("in_rain7").value = rain7d.toFixed(1);
        document.getElementById("in_lag1").value = lag1.toFixed(1);
        document.getElementById("in_lag2").value = lag2.toFixed(1);
        document.getElementById("in_temp").value = Number(tempVal).toFixed(1);
        document.getElementById("in_hum").value = humVal;
        document.getElementById("hum_val").innerText = humVal;
        document.getElementById("in_wind").value = Number(windVal).toFixed(1);
        document.getElementById("in_press").value = Number(pressVal).toFixed(1);
        document.getElementById("in_month").value = monthVal;
        document.getElementById("month_val").innerText = monthVal;

        // Render Live Location Map & Table
        document.getElementById("live-loc-section").style.display = "block";
        const locTitle = [locInfo.name, locInfo.region, locInfo.country].filter(Boolean).join(", ");
        document.getElementById("live-loc-title").innerText = `📍 Live Location: ${locTitle}`;
        document.getElementById("live-loc-caption").innerText =
          `Source: ${locInfo.source} | Coordinates: ${locInfo.latitude.toFixed(4)}°N, ${locInfo.longitude.toFixed(4)}°E | Observed: ${current.time || "Now"} (${data.timezone || "Local"})`;

        if (!mapInstance) {
          mapInstance = L.map("map").setView([locInfo.latitude, locInfo.longitude], 10);
          L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "&copy; OpenStreetMap"
          }).addTo(mapInstance);
          mapMarker = L.marker([locInfo.latitude, locInfo.longitude]).addTo(mapInstance);
        } else {
          mapInstance.setView([locInfo.latitude, locInfo.longitude], 10);
          mapMarker.setLatLng([locInfo.latitude, locInfo.longitude]);
          setTimeout(() => mapInstance.invalidateSize(), 100);
        }

        const tbody = document.getElementById("live-daily-tbody");
        tbody.innerHTML = "";
        for (let i = 0; i < times.length; i++) {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${times[i]}</td><td>${rains[i].toFixed(1)}</td><td>${temps[i].toFixed(1)}</td><td>${hums[i]}</td><td>${winds[i].toFixed(1)}</td><td>${press[i].toFixed(1)}</td>`;
          tbody.appendChild(tr);
        }

        setStatus(`Loaded live weather for <strong>${locInfo.name}, ${locInfo.region || "India"}</strong> (${locInfo.latitude.toFixed(4)}°N, ${locInfo.longitude.toFixed(4)}°E)`);
        await runPrediction();
      } catch (err) {
        setStatus("Weather fetch error: " + err.message, true);
      }
    }

    async function detectAndPredictAP() {
      const selected = document.getElementById("ap-district").value;
      const coords = AP_DISTRICTS[selected] || [16.5062, 80.6480];
      const cityLabel = selected.startsWith("Andhra Pradesh") ? "Amaravati / Vijayawada" : selected;
      await fetchOpenMeteoAndPredict({
        name: cityLabel,
        region: "Andhra Pradesh",
        country: "India",
        latitude: coords[0],
        longitude: coords[1],
        source: "Live Geolocation (Andhra Pradesh)"
      });
    }

    async function searchCity() {
      const q = document.getElementById("city-query").value.trim();
      if (!q) return;
      try {
        const resp = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=6&language=en&format=json`);
        const data = await resp.json();
        citySearchResults = data.results || [];
        const sel = document.getElementById("city-select");
        sel.innerHTML = "";
        if (citySearchResults.length === 0) {
          setStatus("No matching locations found.", true);
          document.getElementById("city-results-wrap").style.display = "none";
          return;
        }
        citySearchResults.forEach((r, idx) => {
          const opt = document.createElement("option");
          opt.value = idx;
          opt.textContent = `${r.name}, ${r.admin1 || ""}, ${r.country || ""} (${r.latitude.toFixed(2)}, ${r.longitude.toFixed(2)})`;
          sel.appendChild(opt);
        });
        document.getElementById("city-results-wrap").style.display = "block";
      } catch (e) {
        setStatus("City search failed: " + e.message, true);
      }
    }

    async function fetchSelectedCity() {
      const idx = Number(document.getElementById("city-select").value);
      const r = citySearchResults[idx];
      if (!r) return;
      await fetchOpenMeteoAndPredict({
        name: r.name,
        region: r.admin1 || "",
        country: r.country || "",
        latitude: Number(r.latitude),
        longitude: Number(r.longitude),
        source: "City Search"
      });
    }

    async function fetchCoordsWeather() {
      const lat = Number(document.getElementById("custom-lat").value);
      const lon = Number(document.getElementById("custom-lon").value);
      await fetchOpenMeteoAndPredict({
        name: `Coords (${lat.toFixed(4)}, ${lon.toFixed(4)})`,
        region: "Andhra Pradesh",
        country: "India",
        latitude: lat,
        longitude: lon,
        source: "Custom GPS Coordinates"
      });
    }

    async function runPrediction() {
      const values = [
        Number(document.getElementById("in_rain").value),
        Number(document.getElementById("in_rain3").value),
        Number(document.getElementById("in_rain7").value),
        Number(document.getElementById("in_lag1").value),
        Number(document.getElementById("in_lag2").value),
        Number(document.getElementById("in_temp").value),
        Number(document.getElementById("in_hum").value),
        Number(document.getElementById("in_wind").value),
        Number(document.getElementById("in_press").value),
        Number(document.getElementById("in_month").value)
      ];

      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values })
      });
      const res = await resp.json();

      document.getElementById("initial-banner").style.display = "none";
      document.getElementById("prediction-section").style.display = "block";

      document.getElementById("m_rain").innerText = `${values[0].toFixed(1)} mm`;
      document.getElementById("m_rain3").innerText = `${values[1].toFixed(1)} mm`;
      document.getElementById("m_rain7").innerText = `${values[2].toFixed(1)} mm`;
      document.getElementById("m_hum").innerText = `${values[6]}%`;

      const riskInfo = {
        0: {
          name: "LOW", emoji: "🟢", cls: "low-risk",
          msg: "Current conditions indicate a lower estimated flood risk. Continue monitoring rainfall and official local advisories."
        },
        1: {
          name: "MEDIUM", emoji: "🟡", cls: "medium-risk",
          msg: "Current conditions indicate an elevated estimated flood risk. Increase monitoring and check official weather and disaster-management advisories."
        },
        2: {
          name: "HIGH", emoji: "🔴", cls: "high-risk",
          msg: "Current conditions indicate a high estimated flood risk. Check official local emergency and disaster-management warnings and follow instructions from authorities."
        }
      }[res.prediction] || { name: "LOW", emoji: "🟢", cls: "low-risk", msg: "" };

      const rc = document.getElementById("risk-card");
      rc.className = "risk-card " + riskInfo.cls;
      document.getElementById("risk-badge").innerText = `${riskInfo.emoji} ${riskInfo.name}`;
      document.getElementById("risk-msg").innerText = riskInfo.msg;

      const probs = res.probabilities || [100, 0, 0];
      document.getElementById("p_low").innerText = `${probs[0].toFixed(2)}%`;
      document.getElementById("p_med").innerText = `${probs[1].toFixed(2)}%`;
      document.getElementById("p_high").innerText = `${probs[2].toFixed(2)}%`;

      if (probChartInstance) probChartInstance.destroy();
      probChartInstance = new Chart(document.getElementById("probChart"), {
        type: "bar",
        data: {
          labels: ["Low", "Medium", "High"],
          datasets: [{
            label: "Probability (%)",
            data: probs,
            backgroundColor: ["#43a047", "#f9a825", "#e53935"]
          }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } } }
      });

      const labels = [
        ["Rainfall", `${values[0].toFixed(1)} mm`],
        ["3-Day Rainfall", `${values[1].toFixed(1)} mm`],
        ["7-Day Rainfall", `${values[2].toFixed(1)} mm`],
        ["Previous-Day Rainfall", `${values[3].toFixed(1)} mm`],
        ["Rainfall 2 Days Ago", `${values[4].toFixed(1)} mm`],
        ["Temperature", `${values[5].toFixed(1)} °C`],
        ["Humidity", `${values[6]} %`],
        ["Wind Speed", `${values[7].toFixed(1)} km/h`],
        ["Pressure", `${values[8].toFixed(1)} hPa`],
        ["Month", `${values[9]}`]
      ];
      const tbody = document.getElementById("input-summary-tbody");
      tbody.innerHTML = labels.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    }

    async function loadDashboardBundle() {
      const resp = await fetch("/api/bundle");
      const data = await resp.json();
      const hist = data.historical;

      document.getElementById("st_avg").innerText = `${hist.stats.avg.toFixed(2)} mm`;
      document.getElementById("st_max").innerText = `${hist.stats.max.toFixed(2)} mm`;
      document.getElementById("st_min").innerText = `${hist.stats.min.toFixed(2)} mm`;
      document.getElementById("st_cnt").innerText = hist.stats.count.toLocaleString();

      new Chart(document.getElementById("recentChart"), {
        type: "line",
        data: {
          labels: hist.recent_dates,
          datasets: [{ label: "Daily Rainfall (mm)", data: hist.recent_rainfall, borderColor: "#1976d2", pointRadius: 0, borderWidth: 1.5 }]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });

      new Chart(document.getElementById("monthlyChart"), {
        type: "bar",
        data: {
          labels: hist.monthly_labels,
          datasets: [{ label: "Avg Monthly Rainfall (mm)", data: hist.monthly_values, backgroundColor: "#3b82f6" }]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });

      const compBody = document.getElementById("model-comp-tbody");
      compBody.innerHTML = (data.model_comparison || []).map(r => `
        <tr>
          <td>${r.Model}</td>
          <td>${(r.Accuracy * 100).toFixed(2)}%</td>
          <td>${(r.Precision * 100).toFixed(2)}%</td>
          <td>${(r.Recall * 100).toFixed(2)}%</td>
          <td>${(r.F1 * 100).toFixed(2)}%</td>
        </tr>
      `).join("");
    }

    loadDashboardBundle();
  </script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/bundle":
            bundle = get_bundle()
            payload = {
                "historical": bundle["historical"],
                "model_comparison": bundle["model_comparison"]
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
            values = [float(x) for x in data.get("values", [0]*10)]
            pred, probs = predict_from_trees(values)
            body = json.dumps({
                "prediction": pred,
                "probabilities": probs
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

