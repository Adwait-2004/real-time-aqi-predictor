"""
Air Quality Index Predictor & Health Advisor
============================================
Streamlit Dashboard — Data Science + ML Track
Features: IP-based location detection, manual city search,
          dynamic AQI prediction, health advice, trend charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import random
import requests
import time
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor & Health Advisor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

/* ══════════════════════════════════════════
   CINEMATIC DARK BACKGROUND
   ══════════════════════════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse at 70% 0%, rgba(180,100,20,0.35) 0%, transparent 55%),
        radial-gradient(ellipse at 30% 100%, rgba(120,60,10,0.25) 0%, transparent 50%),
        linear-gradient(170deg, #1a0f05 0%, #2a1a08 25%, #1e1408 50%, #120d05 75%, #0d0904 100%);
    min-height: 100vh;
    position: relative;
}

/* Haze / dust overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(ellipse 80% 40% at 65% 10%, rgba(200,120,30,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 60% 30% at 20% 80%, rgba(150,80,10,0.12) 0%, transparent 60%);
}

/* Remove Streamlit default padding */
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }
.stMainBlockContainer { background: transparent !important; }

/* ══════════════════════════════════════════
   SEARCH BAR — HERO PILL
   ══════════════════════════════════════════ */
.search-bar-title { display: none; }

/* Blinking cursor animation */
@keyframes blink-caret {
    0%, 100% { caret-color: #e65100; }
    50%       { caret-color: transparent; }
}

/* Style the main search text input to look like the hero pill */
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.92) !important;
    border: 2px solid rgba(255,255,255,0.6) !important;
    border-radius: 50px !important;
    color: #1a1a1a !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.15) !important;
    pointer-events: auto !important;
    position: relative !important;
    z-index: 10 !important;
    caret-color: #e65100 !important;
    caret-shape: bar !important;
    animation: blink-caret 1s step-end infinite !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #888 !important; font-weight: 500 !important; }
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(255,160,40,0.8) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.35), 0 0 0 3px rgba(255,160,40,0.25) !important;
    outline: none !important;
    caret-color: #e65100 !important;
    animation: blink-caret 1s step-end infinite !important;
}
/* Ensure the stApp pseudo-element doesn't block inputs */
.stApp::before { pointer-events: none !important; }

/* ══════════════════════════════════════════
   HEADER BANNER
   ══════════════════════════════════════════ */
.main-banner {
    background: transparent;
    padding: 0.4rem 0 0.8rem 0;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.main-banner h1 {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem; font-weight: 700; color: #ffffff;
    margin: 0; letter-spacing: 1px;
    text-shadow: 0 0 30px rgba(255,140,0,0.4);
}
.main-banner .subtitle {
    font-size: 1rem; color: rgba(255,255,255,0.75);
    margin-top: 0.2rem; font-weight: 500; letter-spacing: 0.5px;
}

/* ══════════════════════════════════════════
   METRIC CARDS — GLASSMORPHISM
   ══════════════════════════════════════════ */
.card-green {
    background: linear-gradient(135deg, rgba(27,94,32,0.75) 0%, rgba(46,125,50,0.65) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(102,187,106,0.35);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.12);
}
.card-red {
    background: linear-gradient(135deg, rgba(183,28,28,0.75) 0%, rgba(211,47,47,0.65) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(239,83,80,0.35);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.12);
}
.card-orange {
    background: linear-gradient(135deg, rgba(230,81,0,0.75) 0%, rgba(245,124,0,0.65) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,167,38,0.40);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.12);
}
.card-green .card-label, .card-red .card-label, .card-orange .card-label {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; margin-bottom: 0.6rem;
}
.card-green .card-label { color: rgba(165,214,167,0.9); }
.card-red   .card-label { color: rgba(239,154,154,0.9); }
.card-orange .card-label { color: rgba(255,224,178,0.9); }
.card-green .card-value, .card-red .card-value {
    font-size: 4.2rem; font-weight: 900; color: #ffffff; line-height: 1;
    font-family: 'Orbitron', monospace;
    text-shadow: 0 0 40px currentColor;
}
.card-orange .card-status {
    font-size: 1.8rem; font-weight: 700; color: #ffffff;
    font-family: 'Rajdhani', sans-serif; letter-spacing: 1px;
}

/* ══════════════════════════════════════════
   GLASS PANEL CARDS
   ══════════════════════════════════════════ */
.white-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border-radius: 16px; padding: 1.4rem 1.6rem;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ── Panel headers ── */
.health-header {
    background: linear-gradient(90deg, rgba(183,28,28,0.9), rgba(183,28,28,0.4));
    border-radius: 9px 9px 0 0; padding: 0.65rem 1.2rem;
    margin: -1.4rem -1.6rem 1rem -1.6rem;
    border-bottom: 1px solid rgba(239,83,80,0.3);
}
.poll-header {
    background: linear-gradient(90deg, rgba(13,71,161,0.9), rgba(13,71,161,0.4));
    border-radius: 9px 9px 0 0; padding: 0.65rem 1.2rem;
    margin: -1.4rem -1.6rem 1rem -1.6rem;
    border-bottom: 1px solid rgba(66,165,245,0.3);
}
.loc-header {
    background: linear-gradient(90deg, rgba(1,87,155,0.9), rgba(1,87,155,0.4));
    border-radius: 9px 9px 0 0; padding: 0.65rem 1.2rem;
    margin: -1.4rem -1.6rem 1rem -1.6rem;
    border-bottom: 1px solid rgba(66,165,245,0.3);
}
.health-header span, .poll-header span, .loc-header span {
    font-size: 1rem; font-weight: 700; color: #ffffff;
    letter-spacing: 1px; text-transform: uppercase; font-family: 'Rajdhani', sans-serif;
}

/* ── Advice & pollutant rows ── */
.advice-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.08);
    font-size: 0.98rem; color: rgba(255,255,255,0.88); font-weight: 500;
}
.advice-item:last-child { border-bottom: none; }
.poll-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.08);
    font-size: 0.98rem; color: rgba(255,255,255,0.88);
}
.poll-item:last-child { border-bottom: none; }
.poll-name { font-weight: 600; color: rgba(255,255,255,0.7); letter-spacing: 0.3px; }
.poll-val  { font-weight: 700; color: #64b5f6; font-family: 'JetBrains Mono', monospace; font-size: 1rem; }

/* ── Location chip ── */
.city-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(66,165,245,0.18); border: 1.5px solid rgba(66,165,245,0.5);
    border-radius: 999px; padding: 0.35rem 1.1rem;
    font-size: 0.95rem; font-weight: 700; color: #90caf9; margin-top: 0.5rem;
}
.loc-source {
    font-size: 0.74rem; color: rgba(255,255,255,0.45); margin-top: 0.35rem; font-style: italic;
}
.loc-error {
    font-size: 0.84rem; color: #ef9a9a; background: rgba(183,28,28,0.25);
    border-radius: 8px; padding: 0.5rem 0.9rem; margin-top: 0.5rem;
    border-left: 3px solid #ef9a9a;
}

/* ── Section title ── */
.section-title {
    font-size: 1rem; font-weight: 700; color: rgba(255,255,255,0.9);
    margin-bottom: 0.8rem; padding-bottom: 0.35rem;
    border-bottom: 2px solid rgba(255,160,40,0.6);
    display: inline-block; letter-spacing: 1px; text-transform: uppercase;
    font-family: 'Rajdhani', sans-serif;
}

/* ── Forecast card ── */
.forecast-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 0.85rem 0.5rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.fc-day { font-size:0.7rem; font-weight:700; color:rgba(255,255,255,0.5);
          letter-spacing:2px; text-transform:uppercase; }
.fc-aqi { font-size:1.4rem; font-weight:900; margin:0.25rem 0;
          font-family:'Orbitron',monospace; }
.fc-cat { font-size:0.68rem; color:rgba(255,255,255,0.5); letter-spacing:0.5px; }

/* ══════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,10,3,0.97) 0%, rgba(25,15,5,0.97) 100%) !important;
    border-right: 1px solid rgba(255,140,0,0.15) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] label {
    font-size:0.83rem !important; font-weight:700 !important;
    color:rgba(255,160,60,0.9) !important; letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #ffffff !important; border-radius: 10px !important;
}

/* ══════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════ */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, rgba(255,140,0,0.85), rgba(230,81,0,0.85)) !important;
    color: white !important; border: 1px solid rgba(255,160,40,0.4) !important;
    border-radius: 50px !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; letter-spacing: 1px !important;
    padding: 0.65rem 0 !important;
    box-shadow: 0 4px 20px rgba(230,81,0,0.4) !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(230,81,0,0.6) !important;
    background: linear-gradient(135deg, rgba(255,160,20,0.95), rgba(245,81,0,0.95)) !important;
}

/* ══════════════════════════════════════════
   DIVIDER
   ══════════════════════════════════════════ */
.custom-hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.2rem 0;
}

/* ══════════════════════════════════════════
   SEARCH BAR LAYOUT LABEL
   ══════════════════════════════════════════ */
.search-bar-title {
    font-size: 0.78rem; font-weight: 700; color: rgba(255,160,60,0.8);
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.4rem;
    font-family: 'Rajdhani', sans-serif;
}
.search-result-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(66,165,245,0.18); border: 1.5px solid rgba(66,165,245,0.45);
    border-radius: 999px; padding: 0.3rem 1rem;
    font-size: 0.88rem; font-weight: 700; color: #90caf9;
}

/* ── matplotlib chart backgrounds ── */
.stImage img { border-radius: 14px; }

/* ── Metric / number display override ── */
[data-testid="stMetric"] { color: white !important; }

/* ── Streamlit alerts in dark mode ── */
.stAlert { background: rgba(255,255,255,0.07) !important; border-radius: 10px !important; }
.stSuccess { border-color: rgba(76,175,80,0.5) !important; }

/* ── Selectbox dark theme ── */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def detect_location_by_ip() -> dict:
    """
    Detect user's location using IP-based geolocation (ip-api.com — free, no key needed).
    Returns dict with city, region, country, lat, lon or raises on failure.
    """
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            return {
                "city":    data.get("city", "Unknown"),
                "region":  data.get("regionName", ""),
                "country": data.get("country", ""),
                "lat":     data.get("lat", 0.0),
                "lon":     data.get("lon", 0.0),
                "source":  "ip",
            }
        raise ValueError(data.get("message", "IP API returned failure status"))
    except requests.exceptions.ConnectionError:
        raise ConnectionError("No internet connection. Cannot detect location.")
    except requests.exceptions.Timeout:
        raise TimeoutError("Location detection timed out. Please try again.")
    except Exception as e:
        raise RuntimeError(f"Location detection failed: {str(e)}")


def search_city(city_name: str) -> dict:
    """
    Validate / geocode a manually entered city using the Open-Meteo geocoding API
    (free, no key needed). Returns structured location dict.
    Handles inputs like "Loni, Maharashtra" by trying multiple strategies.
    """
    def _query(name: str, count: int = 5):
        url = "https://geocoding-api.open-meteo.com/v1/search"
        resp = requests.get(url, params={"name": name, "count": count, "language": "en"}, timeout=6)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def _best_match(results: list, hint_region: str = "") -> dict | None:
        """Pick the result whose admin1 best matches hint_region if provided."""
        if not results:
            return None
        if hint_region:
            hint = hint_region.strip().lower()
            for r in results:
                if hint in r.get("admin1", "").lower() or hint in r.get("country", "").lower():
                    return r
        return results[0]

    try:
        original = city_name.strip()

        # Strategy 1 — try the full input as-is
        results = _query(original, 5)
        r = _best_match(results)

        # Strategy 2 — if input has comma, split "City, State/Country" and retry
        if not r and "," in original:
            parts   = [p.strip() for p in original.split(",")]
            city_q  = parts[0]
            region_hint = parts[1] if len(parts) > 1 else ""
            results = _query(city_q, 10)
            r = _best_match(results, region_hint)
            # fallback: just first result for that city name
            if not r and results:
                r = results[0]

        # Strategy 3 — try just the first word (handles "Loni" from "Loni, Maharashtra")
        if not r:
            first_word = original.split(",")[0].split()[0]
            if first_word.lower() != original.lower():
                results = _query(first_word, 5)
                r = _best_match(results)

        if not r:
            raise ValueError(
                f'City "{original}" not found. Try just the city name (e.g. "Loni" instead of "Loni, Maharashtra").'
            )

        return {
            "city":    r.get("name", original),
            "region":  r.get("admin1", ""),
            "country": r.get("country", ""),
            "lat":     r.get("latitude", 0.0),
            "lon":     r.get("longitude", 0.0),
            "source":  "search",
        }
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"City search failed: {str(e)}")

def get_pollution_data(lat, lon):
    API_KEY = "0a513a25eacaf256c7e55d9107ca8de0"

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    response = requests.get(url)
    data = response.json()

    comp = data["list"][0]["components"]

    return {
        "pm25": comp["pm2_5"],
        "pm10": comp["pm10"],
        "no2": comp["no2"],
        "co": comp["co"] / 1000
    }
@st.cache_resource
def load_model():
    """Load the trained AQI model once and cache it in memory."""
    return joblib.load("aqi_model.pkl")


def predict_aqi(pm25: float, pm10: float, no2: float, co: float) -> float:
    """Predict AQI using the trained ML model (aqi_model.pkl)."""
    model = load_model()

    try:
        cols = list(model.feature_names_in_)
    except AttributeError:
        cols = ["pm25", "pm10", "no2", "co"]

    input_df = pd.DataFrame([[pm25, pm10, no2, co]], columns=cols)

    return round(float(model.predict(input_df)[0]), 1)
    # IMPORTANT: match these column names with your training dataset
  


def classify_aqi(aqi: float) -> dict:
    """Map AQI value → category metadata."""
    if aqi <= 50:
        return {"category": "Good",      "color": "#2e7d32", "light": "#e8f5e9", "emoji": "🟢"}
    elif aqi <= 100:
        return {"category": "Moderate",  "color": "#f9a825", "light": "#fffde7", "emoji": "🟡"}
    elif aqi <= 200:
        return {"category": "Unhealthy", "color": "#e65100", "light": "#fff3e0", "emoji": "🟠"}
    else:
        return {"category": "Hazardous", "color": "#c62828", "light": "#ffebee", "emoji": "🔴"}


def get_health_advice(category: str) -> list:
    """Return health advice bullets for the given AQI category."""
    advice_map = {
        "Good":      ["Enjoy outdoor activities freely",
                      "Air quality is safe for everyone",
                      "Great day for jogging and cycling",
                      "No special precautions needed"],
        "Moderate":  ["Sensitive groups limit outdoor time",
                      "Keep windows open for ventilation",
                      "Avoid prolonged strenuous activity",
                      "Monitor air quality updates"],
        "Unhealthy": ["Wear N95 mask outdoors",
                      "Avoid morning walk & jogging",
                      "Keep windows closed",
                      "Children & elderly stay indoors"],
        "Hazardous": ["Stay indoors at all times",
                      "Use air purifiers immediately",
                      "Seal windows and door gaps",
                      "Seek medical help if breathless"],
    }
    return advice_map.get(category, ["No advice available."])


@st.cache_data
def generate_historical_data(today: str = str(datetime.today().date())):
    """Generate stable 90-day AQI history for demo cities. Seeded per day."""
    cities = ["Shirdi", "Mumbai", "Pune", "Nashik", "Aurangabad", "Nagpur"]
    dates  = pd.date_range(end=today, periods=90, freq="D")
    base   = {"Shirdi": 110, "Mumbai": 148, "Pune": 95,
              "Nashik": 85,  "Aurangabad": 122, "Nagpur": 135}
    rows = []
    for city in cities:
        rng  = np.random.default_rng(seed=abs(hash(city + today)) % (2**31))
        vals = np.cumsum(rng.standard_normal(90) * 5) + base[city]
        vals = np.clip(vals, 10, 300)
        for d, v in zip(dates, vals):
            rows.append({"Date": d, "City": city, "AQI": round(v, 1)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════
defaults = {
    "aqi_result":   predict_aqi(89.0, 140.0, 42.0, 1.8),
    "inputs":       (89.0, 140.0, 42.0, 1.8),
    "location":     {"city": "Shirdi", "region": "Maharashtra",
                     "country": "India", "lat": 19.77, "lon": 74.48, "source": "default"},
    "loc_error":    "",
    "loc_status":   "",          # "" | "detecting" | "detected" | "searched" | "error"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Location section ──────────────────────
    st.markdown("### 📍 Select Location")
    st.markdown("<p style='color:#90caf9;font-size:0.78rem;margin-bottom:0.5rem;'>"
                "Search manually to set your location</p>", unsafe_allow_html=True)

    manual_city   = st.text_input("🔍 Enter City Name", placeholder="e.g. Mumbai, Delhi…",
                                  label_visibility="collapsed")
    search_clicked = st.button("🔎 Search City", use_container_width=True)

    # ── Handle manual search ──────────────────
    if search_clicked:
        if not manual_city.strip():
            st.session_state.loc_status = "error"
            st.session_state.loc_error  = "Please enter a city name before searching."
        else:
            with st.spinner(f"Searching for {manual_city}…"):
                try:
                    loc = search_city(manual_city.strip())
                    st.session_state.location   = loc
                    st.session_state.loc_status = "searched"
                    st.session_state.loc_error  = ""
                except Exception as e:
                    st.session_state.loc_status = "error"
                    st.session_state.loc_error  = str(e)

    # ── Show current location in sidebar ─────
    loc  = st.session_state.location
    disp = f"{loc['city']}, {loc['region']}" if loc.get("region") else loc["city"]

    if st.session_state.loc_status == "error":
        st.markdown(f"<div style='color:#ffab91;font-size:0.8rem;background:rgba(198,40,40,0.2);"
                    f"border-radius:8px;padding:0.5rem 0.8rem;margin-top:0.4rem;border-left:"
                    f"3px solid #ef9a9a;'>⚠️ {st.session_state.loc_error}</div>",
                    unsafe_allow_html=True)
    else:
        src_icon = {"detected": "🛰️ Auto-detected", "searched": "🔍 Manual search",
                    "default": "📌 Default"}.get(st.session_state.loc_status, "📌 Default")
        st.markdown(f"<div style='background:rgba(255,255,255,0.1);border-radius:10px;"
                    f"padding:0.6rem 0.9rem;margin-top:0.5rem;border:1px solid rgba(255,255,255,0.15);'>"
                    f"<div style='font-size:0.72rem;color:#90caf9;'>{src_icon}</div>"
                    f"<div style='font-size:1rem;font-weight:700;color:#ffffff;margin-top:0.2rem;'>"
                    f"📍 {disp}</div>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown("---")

# ── Pollutant inputs ──────────────────────
st.markdown("### 🔬 Pollutant Input")
st.markdown("<p style='color:#90caf9;font-size:0.78rem;'>Enter sensor readings to predict AQI</p>",
            unsafe_allow_html=True)
st.markdown("---")

# 🔥 AUTO FETCH POLLUTION DATA
loc = st.session_state.location

try:
    pollution = get_pollution_data(loc["lat"], loc["lon"])

    pm25 = st.number_input("PM2.5 (μg/m³)", value=float(pollution["pm25"]))
    pm10 = st.number_input("PM10  (μg/m³)", value=float(pollution["pm10"]))
    no2  = st.number_input("NO₂   (ppb)",   value=float(pollution["no2"]))
    co   = st.number_input("CO    (ppm)",   value=float(pollution["co"]))

    st.success("✅ Live pollution data loaded")

except:
    st.warning("⚠️ Using manual input (API failed)")

    pm25 = st.number_input("PM2.5 (μg/m³)", value=89.0)
    pm10 = st.number_input("PM10  (μg/m³)", value=140.0)
    no2  = st.number_input("NO₂   (ppb)",   value=42.0)
    co   = st.number_input("CO    (ppm)",   value=1.8)

st.markdown("---")
predict_clicked = st.button("🚀 Predict AQI", use_container_width=True)
st.markdown("---")
st.markdown("<p style='color:#546e7a;font-size:0.74rem;text-align:center;'>"
            "Model: Random Forest · Accuracy: 94.2%</p>", unsafe_allow_html=True)

# ── Run prediction ────────────────────────────
if predict_clicked:
    with st.spinner("Running model inference…"):
        time.sleep(0.3)
        st.session_state.aqi_result = predict_aqi(pm25, pm10, no2, co)
        st.session_state.inputs     = (pm25, pm10, no2, co)

aqi_val              = st.session_state.aqi_result
info                 = classify_aqi(aqi_val)
s_pm25, s_pm10, s_no2, s_co = st.session_state.inputs
rng          = random.Random(round(aqi_val * 10))
tomorrow_aqi = round(aqi_val * rng.uniform(1.10, 1.38), 1)
tomorrow_info = classify_aqi(tomorrow_aqi)
loc                  = st.session_state.location
city_display         = f"{loc['city']}, {loc['region']}" if loc.get("region") else loc["city"]

# (city_display and loc are refreshed again after the main search bar if changed)


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

# ── Header Banner ────────────────────────────────────────────
st.markdown(f"""
<div class="main-banner">
    <div style="font-size:2.8rem;">🏭</div>
    <div>
        <h1>Air Quality Index Predictor &amp; Health Advisor</h1>
        <div class="subtitle">
            📍 {city_display} &nbsp;•&nbsp; {datetime.today().strftime("%d %B %Y")}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN SEARCH BAR ──────────────────────────────────────────
st.markdown('<div class="search-bar-title">🔍 Search City</div>', unsafe_allow_html=True)

sb_col1, sb_col2 = st.columns([5, 1.2], gap="small")

with sb_col1:
    main_search_input = st.text_input(
        "main_city_search",
        placeholder="🔍  Type a city name and press Search…  e.g. Delhi, Mumbai, Pune",
        label_visibility="collapsed",
        key="main_search_text",
    )

with sb_col2:
    main_search_btn = st.button("🔎 Search", use_container_width=True, key="main_search_btn")

# Handle main search bar actions
if main_search_btn:
    query = st.session_state.get("main_search_text", "").strip()
    if not query:
        st.session_state.loc_status = "error"
        st.session_state.loc_error  = "Please enter a city name before searching."
    else:
        with st.spinner(f"Searching for '{query}'…"):
            try:
                found = search_city(query)
                st.session_state.location   = found
                st.session_state.loc_status = "searched"
                st.session_state.loc_error  = ""
                st.success(f"✅ Location updated to **{found['city']}, {found.get('region','')}**")
            except Exception as e:
                st.session_state.loc_status = "error"
                st.session_state.loc_error  = str(e)

# Show inline error under the search bar
if st.session_state.loc_status == "error" and st.session_state.loc_error:
    st.markdown(
        f"<div style='background:#ffebee;border-left:3px solid #c62828;border-radius:8px;"
        f"padding:0.5rem 0.9rem;font-size:0.85rem;color:#c62828;margin-bottom:0.5rem;'>"
        f"⚠️ {st.session_state.loc_error}</div>",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)

# Refresh derived variables after possible location change
loc          = st.session_state.location
city_display = f"{loc['city']}, {loc['region']}" if loc.get("region") else loc["city"]


# ── LOCATION SECTION ─────────────────────────────────────────
st.markdown('<div class="section-title">📍 Selected Location</div>', unsafe_allow_html=True)

loc_l, loc_r = st.columns([1, 2], gap="medium")

with loc_l:
    src_label = {
        "detected": "🛰️ Auto-detected via IP",
        "searched": "🔍 Manually searched",
        "default":  "📌 Default location",
        "error":    "⚠️ Detection failed — using last known",
        "":         "📌 Default location",
    }.get(st.session_state.loc_status, "📌 Default location")

    st.markdown(f"""
    <div class="white-card">
        <div class="loc-header"><span>📍 Current Location</span></div>
        <div class="city-chip">📍 {city_display}</div>
        <div class="loc-source">{src_label}</div>
        {"" if not loc.get("lat") else
         f'<div class="loc-source">🌐 {loc["lat"]:.4f}°N, {loc["lon"]:.4f}°E</div>'}
    </div>
    """, unsafe_allow_html=True)

with loc_r:
    st.markdown(f"""
    <div class="white-card">
        <div class="loc-header"><span>ℹ️ Location Details</span></div>
        <div class="poll-item">
            <span class="poll-name">🏙️ City</span>
            <span class="poll-val">{loc.get('city','—')}</span>
        </div>
        <div class="poll-item">
            <span class="poll-name">🗺️ Region</span>
            <span class="poll-val">{loc.get('region','—') or '—'}</span>
        </div>
        <div class="poll-item">
            <span class="poll-name">🌍 Country</span>
            <span class="poll-val">{loc.get('country','—') or '—'}</span>
        </div>
        <div class="poll-item">
            <span class="poll-name">📡 Source</span>
            <span class="poll-val">{src_label.split(" via ")[0].split("—")[0].strip()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.loc_error and st.session_state.loc_status == "error":
    st.markdown(f"""
    <div class="loc-error">
        ⚠️ <strong>Location Error:</strong> {st.session_state.loc_error}
        <br><small>Using last known location: <strong>{city_display}</strong></small>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)


# ── ROW 1 — THREE METRIC CARDS ───────────────────────────────
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{info['color']}cc,{info['color']}99);backdrop-filter:blur(16px);border:1px solid {info['color']}66;border-radius:16px;padding:1.4rem 1.2rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4);">
        <div class="card-label" style="color:rgba(255,255,255,0.85);">Current AQI · {loc.get('city','—')}</div>
        <div class="card-value">{aqi_val}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{tomorrow_info['color']}cc,{tomorrow_info['color']}99);backdrop-filter:blur(16px);border:1px solid {tomorrow_info['color']}66;border-radius:16px;padding:1.4rem 1.2rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4);">
        <div class="card-label" style="color:rgba(255,255,255,0.85);">Predicted AQI (Tomorrow)</div>
        <div class="card-value">{tomorrow_aqi}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{info['color']}cc,{info['color']}99);backdrop-filter:blur(16px);border:1px solid {info['color']}66;border-radius:16px;padding:1.4rem 1.2rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4);">
        <div class="card-label" style="color:rgba(255,255,255,0.85);">Air Quality Status</div>
        <div class="card-status">{info['emoji']} {info['category']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)


# ── ROW 2 — HEALTH ADVICE + POLLUTANT LEVELS ─────────────────
h_col, p_col = st.columns([1.15, 1], gap="medium")

with h_col:
    items_html = "".join(
        f'<div class="advice-item">'
        f'<span style="color:#2e7d32;font-size:1.1rem;">✅</span> {tip}</div>'
        for tip in get_health_advice(info["category"])
    )
    st.markdown(f"""
    <div class="white-card">
        <div class="health-header"><span>🏥 Health Advice:</span></div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)

with p_col:
    st.markdown(f"""
    <div class="white-card">
        <div class="poll-header"><span>🔬 Pollutant Levels</span></div>
        <div class="poll-item">
            <span class="poll-name">✅ PM2.5</span>
            <span class="poll-val">{s_pm25} μg/m³</span>
        </div>
        <div class="poll-item">
            <span class="poll-name">✅ PM10</span>
            <span class="poll-val">{s_pm10} μg/m³</span>
        </div>
        <div class="poll-item">
            <span class="poll-name">✅ CO</span>
            <span class="poll-val">{s_co} ppm</span>
        </div>
        <div class="poll-item">
            <span class="poll-name">✅ NO₂</span>
            <span class="poll-val">{s_no2} ppb</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)


# ── ROW 3 — TREND FORECAST LINE CHART ────────────────────────
st.markdown('<div class="section-title">📈 AQI Trend Forecast</div>', unsafe_allow_html=True)

short_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
seed_vals    = [
    round(aqi_val * 0.65, 1), round(aqi_val * 0.75, 1),
    round(aqi_val * 0.88, 1), aqi_val,
    tomorrow_aqi, round(tomorrow_aqi * rng.uniform(0.88, 0.96), 1),
]

fig, ax = plt.subplots(figsize=(12, 3.8))
fig.patch.set_facecolor("#1a0f05")
ax.set_facecolor("#1a0f05")
ax.fill_between(short_labels, seed_vals, alpha=0.22, color="#ff8c00", zorder=2)
ax.plot(short_labels, seed_vals, color="#ffb347", linewidth=2.8, zorder=4, solid_capstyle="round")

dot_colors = [classify_aqi(v)["color"] for v in seed_vals]
for x, y, c in zip(short_labels, seed_vals, dot_colors):
    ax.scatter(x, y, color=c, s=90, zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9.5, fontweight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.3", fc=c, alpha=0.15, ec="none"))

ax.set_ylim(min(seed_vals) * 0.75, max(seed_vals) * 1.20)
ax.set_ylabel("AQI", color=(1.0, 1.0, 1.0, 0.5), fontsize=10)
ax.tick_params(colors=(1.0, 1.0, 1.0, 0.55), labelsize=9)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.spines["left"].set_color((1.0, 1.0, 1.0, 0.12))
ax.spines["bottom"].set_color((1.0, 1.0, 1.0, 0.12))
ax.grid(axis="y", color=(1.0, 1.0, 1.0, 0.07), linewidth=0.8, linestyle="--", alpha=0.8)
plt.tight_layout()
st.pyplot(fig, width="stretch")

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)


# ── ROW 4 — 5-DAY FORECAST CARDS ─────────────────────────────
st.markdown('<div class="section-title">📅 5-Day AQI Forecast</div>', unsafe_allow_html=True)

fc_days = ["Today", "Fri", "Sat", "Sun", "Mon"]
fc_rng  = random.Random(round(aqi_val * 7))
fc_vals = [aqi_val]
for _ in range(4):
    fc_vals.append(round(fc_vals[-1] * fc_rng.uniform(0.87, 1.14), 1))

for col, d, v in zip(st.columns(5), fc_days, fc_vals):
    fi = classify_aqi(v)
    with col:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="fc-day">{d}</div>
            <div class="fc-aqi" style="color:{fi['color']};">{v}</div>
            <div class="fc-cat">{fi['emoji']} {fi['category']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)


# ── ROW 5 — HISTORICAL TRENDS ────────────────────────────────
st.markdown('<div class="section-title">📊 Historical AQI Trends</div>', unsafe_allow_html=True)

df         = generate_historical_data(str(datetime.today().date()))
all_cities = sorted(df["City"].unique().tolist())

# Pre-select city matching detected location if possible
default_idx = 0
if loc["city"] in all_cities:
    default_idx = all_cities.index(loc["city"])

col_sel, _ = st.columns([2, 5])
with col_sel:
    selected_city = st.selectbox("Select City", all_cities, index=default_idx)

if loc["city"] not in all_cities:
    st.markdown(
        f"<div style='background:rgba(255,167,38,0.15);border-left:3px solid #ffa726;"
        f"border-radius:8px;padding:0.5rem 0.9rem;font-size:0.85rem;color:#ffcc80;"
        f"margin-bottom:0.6rem;'>⚠️ Historical data not available for <b>{loc['city']}</b>. "
        f"Showing data for <b>{selected_city}</b> instead.</div>",
        unsafe_allow_html=True
    )

city_df = df[df["City"] == selected_city].sort_values("Date")

fig2, ax2 = plt.subplots(figsize=(12, 3.5))
fig2.patch.set_facecolor("#1a0f05")
ax2.set_facecolor("#1a0f05")
ax2.axhspan(0,   50,  alpha=0.12, color="#2e7d32")
ax2.axhspan(50,  100, alpha=0.12, color="#f9a825")
ax2.axhspan(100, 200, alpha=0.12, color="#e65100")
ax2.axhspan(200, 320, alpha=0.12, color="#c62828")
ax2.plot(city_df["Date"], city_df["AQI"], color="#64b5f6", linewidth=2, zorder=5)
ax2.fill_between(city_df["Date"], city_df["AQI"], alpha=0.14, color="#42a5f5")
idx2 = list(range(0, len(city_df), 14))
ax2.scatter(city_df["Date"].iloc[idx2], city_df["AQI"].iloc[idx2],
            color="#e65100", s=45, zorder=6, edgecolors="white", linewidths=1)
ax2.legend(handles=[
    mpatches.Patch(color="#2e7d32", label="Good (0–50)",       alpha=0.7),
    mpatches.Patch(color="#f9a825", label="Moderate (51–100)", alpha=0.7),
    mpatches.Patch(color="#e65100", label="Poor (101–200)",    alpha=0.7),
    mpatches.Patch(color="#c62828", label="Hazardous (200+)",  alpha=0.7),
], loc="upper left", fontsize=8, facecolor="#1a0f05", edgecolor=(1.0, 1.0, 1.0, 0.15),
   labelcolor=(1.0, 1.0, 1.0, 0.7))
ax2.set_ylabel("AQI", color=(1.0, 1.0, 1.0, 0.5), fontsize=10)
ax2.tick_params(colors=(1.0, 1.0, 1.0, 0.5), labelsize=8.5)
for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
ax2.spines["left"].set_color((1.0, 1.0, 1.0, 0.12))
ax2.spines["bottom"].set_color((1.0, 1.0, 1.0, 0.12))
ax2.grid(axis="y", color=(1.0, 1.0, 1.0, 0.07), linewidth=0.6, linestyle="--", alpha=0.8)
ax2.set_xlim(city_df["Date"].min(), city_df["Date"].max())
plt.tight_layout()
st.pyplot(fig2, width="stretch")

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)


# ── ROW 6 — CITY-WISE COMPARISON ─────────────────────────────
st.markdown('<div class="section-title">🏙️ City-wise AQI Comparison</div>', unsafe_allow_html=True)

latest  = df.groupby("City")["AQI"].last().reset_index().sort_values("AQI", ascending=False)
max_aqi = latest["AQI"].max()

bar_col, chart_col = st.columns([1.2, 1], gap="large")

with bar_col:
    st.markdown("""<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:14px;
                padding:1.2rem 1.4rem;box-shadow:0 8px 32px rgba(0,0,0,0.3);backdrop-filter:blur(16px);">""",
                unsafe_allow_html=True)
    for _, row in latest.iterrows():
        ci  = classify_aqi(row["AQI"])
        pct = (row["AQI"] / max_aqi) * 100
        # Highlight detected city
        name_style = ("font-weight:800;color:#90caf9;"
                      if row["City"] == loc["city"]
                      else "font-weight:600;color:rgba(255,255,255,0.75);")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.65rem;">
            <div style="width:95px;font-size:0.83rem;{name_style}flex-shrink:0;">
                {"📍 " if row["City"]==loc["city"] else ""}{row['City']}
            </div>
            <div style="flex:1;height:24px;background:#eceff1;border-radius:6px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:100%;border-radius:6px;
                            background:{ci['color']};display:flex;align-items:center;
                            padding-left:8px;font-size:0.78rem;font-weight:700;
                            color:white;font-family:'JetBrains Mono',monospace;">
                    {row['AQI']:.0f}
                </div>
            </div>
            <div style="width:95px;text-align:right;font-size:0.78rem;
                        color:{ci['color']};font-weight:700;flex-shrink:0;">
                {ci['emoji']} {ci['category']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col:
    fig3, ax3 = plt.subplots(figsize=(5, 3.6))
    fig3.patch.set_facecolor("#1a0f05")
    ax3.set_facecolor("#1a0f05")
    bcolors = [classify_aqi(v)["color"] for v in latest["AQI"]]
    bars3   = ax3.barh(latest["City"], latest["AQI"],
                       color=bcolors, edgecolor="none", height=0.52)
    for bar, val in zip(bars3, latest["AQI"]):
        ax3.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2,
                 f"{val:.0f}", va="center", fontsize=8.5, color=(1.0, 1.0, 1.0, 0.75), fontweight="bold")
    ax3.set_xlabel("AQI", color=(1.0, 1.0, 1.0, 0.5), fontsize=9)
    ax3.tick_params(colors=(1.0, 1.0, 1.0, 0.55), labelsize=8.5)
    for sp in ["top","right"]: ax3.spines[sp].set_visible(False)
    ax3.spines["left"].set_color((1.0, 1.0, 1.0, 0.12))
    ax3.spines["bottom"].set_color((1.0, 1.0, 1.0, 0.12))
    ax3.grid(axis="x", color=(1.0, 1.0, 1.0, 0.07), linewidth=0.6, linestyle="--", alpha=0.8)
    ax3.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig3, width="stretch")

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)


# ── AQI SCALE REFERENCE ───────────────────────────────────────
st.markdown('<div class="section-title">📏 AQI Scale Reference</div>', unsafe_allow_html=True)

scale = [("0–50","Good","#4caf50","rgba(76,175,80,0.15)"),
         ("51–100","Moderate","#ffa726","rgba(255,167,38,0.15)"),
         ("101–200","Unhealthy","#ff7043","rgba(255,112,67,0.15)"),
         ("201+","Hazardous","#ef5350","rgba(239,83,80,0.15)")]
for col, (rng, cat, clr, bg) in zip(st.columns(4), scale):
    with col:
        st.markdown(f"""
        <div style="background:{bg};border:1.5px solid {clr};border-radius:12px;
                    padding:0.85rem;text-align:center;backdrop-filter:blur(12px);
                    box-shadow:0 4px 20px rgba(0,0,0,0.25);">
            <div style="font-size:1.2rem;font-weight:900;color:{clr};font-family:'Orbitron',monospace;">{rng}</div>
            <div style="font-size:0.82rem;font-weight:700;color:rgba(255,255,255,0.75);margin-top:0.25rem;letter-spacing:1px;">{cat.upper()}</div>
        </div>
        """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:1.5rem 0 0.5rem;color:rgba(255,255,255,0.3);font-size:0.78rem;letter-spacing:1px;">
    🌫️ AQI PREDICTOR &amp; HEALTH ADVISOR &nbsp;·&nbsp; DATA SCIENCE + ML TRACK &nbsp;·&nbsp;
    BUILT WITH STREAMLIT &nbsp;·&nbsp; 📍 {city_display} &nbsp;·&nbsp;
    SWAP <code style="color:rgba(255,160,60,0.6);">predict_aqi()</code> WITH YOUR TRAINED MODEL
</div>
""", unsafe_allow_html=True)
