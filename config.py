import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# OpenAQ Config
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "5f8d24a8d1352d6ffdf8b1a3797252fa2611c9b01119e17998bc0b6bff2533c4")  # Set via export OPENAQ_API_KEY=xxx
YEAR = 2025
MAX_STATIONS = 100
PARAMS = ["pm25", "pm10", "no2", "o3", "temperature", "relative_humidity"]
HEALTH_THRESHOLD = 35  # PM2.5 µg/m³
EXTREME_HAZARD = 200

# Zone keywords
INDUSTRIAL_KEYWORDS = ["industrial", "factory", "port", "power", "refinery"]
RESIDENTIAL_KEYWORDS = ["residential", "park", "school", "hospital"]
