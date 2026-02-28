"""
assign_zones.py  (v3 — hybrid: Nominatim + expert overrides)
-------------------------------------------------------------
Zone assignment pipeline:

  PRIORITY 1 — Expert overrides  (for stations where API is provably wrong)
  PRIORITY 2 — Nominatim cache   (already fetched, use as-is)
  PRIORITY 3 — Nominatim API     (for any uncached stations)
  PRIORITY 4 — Mixed fallback

WHY HYBRID?
  Nominatim resolves coordinates to the nearest OSM feature, which is often
  a residential neighbourhood even when the monitoring station is placed near
  an industrial facility. Expert overrides correct these known mismatches using
  real-world knowledge of the station locations.

  This approach is standard in environmental data science:
  automated geocoding handles the bulk, domain knowledge corrects the edge cases.

USAGE:
  from assign_zones import load_or_fetch_zones, assign_zones_to_df

  zone_map = load_or_fetch_zones()   # instant if cache exists
  df       = assign_zones_to_df(df, zone_map)
"""

import time
import json
import logging
import pandas as pd
import requests
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
CACHE_PATH    = Path("data/raw/zone_cache.json")
RATE_LIMIT    = 1.2
TIMEOUT       = 20
MAX_RETRIES   = 3

HEADERS = {
    "User-Agent": "AirQualityResearch/1.0 (university assignment)"
}

# ── PRIORITY 1: EXPERT OVERRIDES ─────────────────────────────────────────────
# Applied AFTER Nominatim, overriding any API result.
# Each entry documents WHY it overrides the API result.

EXPERT_OVERRIDES = {
    # ── CANADA ──────────────────────────────────────────────────────────────
    # Alberta oil sands corridor — API returns "administrative boundary"
    # because these are sparse rural areas with no OSM land-use polygons,
    # but they're all industrial air quality monitoring posts for oil/gas ops
    297: ("Industrial",  "Alberta oil sands monitoring, near Hinton gas facilities"),
    298: ("Industrial",  "Bruderheim — oil/gas pipeline terminal, Alberta"),
    360: ("Industrial",  "Patricia McInnes — oil sands monitoring station"),
    287: ("Residential", "St. Lina — small agricultural rural community"),
    382: ("Residential", "Elk Island — national park, background monitoring"),
    390: ("Mixed",       "Red Deer Riverside — mixed urban riverside"),

    # ── CHILE ───────────────────────────────────────────────────────────────
    # API returned 'neighbourhood' for all Chilean stations because OSM
    # Chile has poor landuse polygon coverage outside Santiago
    26:  ("Industrial",  "Inpesca — fish processing industrial plant, Coronel"),
    27:  ("Industrial",  "Concón — Chile's main petrochemical/refinery hub"),
    25:  ("Residential", "Parque O'Higgins — urban park, Santiago"),
    364: ("Residential", "Colegio Pedro Vergara Keller — school monitoring"),
    210: ("Residential", "Liceo Polivalente — school/educational zone"),
    69:  ("Residential", "Kingston College — school monitoring station"),
    388: ("Residential", "Lota rural — explicitly rural residential"),
    65:  ("Residential", "Punteras — rural residential, Biobío region"),
    356: ("Residential", "Bocatoma — water intake, rural residential area"),
    68:  ("Residential", "Cerro Merquín — hillside residential/nature"),
    73:  ("Residential", "Coyhaique II — small Patagonian town, residential"),
    72:  ("Residential", "Curicó — mid-sized residential city centre"),
    54:  ("Residential", "Talagante — suburban residential, Santiago metro"),
    270: ("Residential", "Los Ángeles Oriente — residential east side"),
    45:  ("Mixed",       "Puente Alto — large suburban mixed municipality"),

    # ── NETHERLANDS ─────────────────────────────────────────────────────────
    # Dutch stations: API often returns 'administrative' boundary.
    # Netherlands has excellent OSM data but monitoring stations sit on
    # street/boundary nodes so landuse polygons aren't always matched.
    97:  ("Industrial",  "IJmuiden-Kanaaldijk — Tata Steel plant + canal industrial zone"),
    96:  ("Industrial",  "Wijk aan Zee — directly downwind of IJmuiden steel plant"),
    77:  ("Industrial",  "Hoek v. Holland-Berghaven — major Rotterdam port, haven=harbour"),
    33:  ("Industrial",  "Zaanstad-Hemkade — industrial riverbank, Zaan industrial area"),
    82:  ("Industrial",  "Haarlem-Schipholweg — road to Schiphol airport, industrial zone"),
    101: ("Industrial",  "Amsterdam-Hoogtij — western harbour industrial zone"),
    102: ("Industrial",  "Amsterdam-Einsteinweg — industrial park west Amsterdam"),
    80:  ("Industrial",  "Amsterdam-Van Diemenstraat — port/canal industrial corridor"),
    41:  ("Industrial",  "Groningen-Europaweg — industrial ring road"),
    52:  ("Industrial",  "Rotterdam-Oost A13 — motorway/industrial corridor"),
    95:  ("Residential", "Amsterdam-Vondelpark — famous city park, residential"),
    98:  ("Residential", "Amsterdam-Westerpark — park area, residential"),
    100: ("Residential", "Badhoevedorp-Sloterweg — suburban residential"),
    44:  ("Residential", "Heerlen-Jamboreepad — residential neighbourhood"),
    55:  ("Residential", "Utrecht-Kardinaal de Jongweg — residential Utrecht"),
    58:  ("Residential", "Zegveld-Oude Meije — rural polder/farmland"),
    31:  ("Residential", "Wekerom-Riemterdijk — rural Veluwe residential"),
    85:  ("Residential", "Hellendoorn — small town residential"),
    76:  ("Residential", "De Zilk-Vogelaarsdreef — dune/nature residential"),
    63:  ("Residential", "Philippine-Stelleweg — small Zeeland farming town"),
    71:  ("Residential", "Biest Houtakker — rural Noord-Brabant"),
    93:  ("Residential", "Nijmegen-Graafseweg — residential Nijmegen"),
    51:  ("Residential", "Wieringerwerf — rural polder residential"),

    # ── UNITED STATES ───────────────────────────────────────────────────────
    223: ("Industrial",  "Gary-IITRI — Gary IN, iconic US steel/industrial city"),
    282: ("Industrial",  "Pinedale Gaseous — natural gas field, Wyoming"),
    365: ("Industrial",  "Hamshire C64 — petrochemical SE Texas"),
    232: ("Industrial",  "Trona-Athol — mining town, soda ash/trona extraction"),
    314: ("Industrial",  "Indpls I-70 E — highway/industrial corridor Indianapolis"),
    186: ("Industrial",  "Houston Aldine C8 — industrial north Houston"),
    274: ("Mixed",       "Reading Airport — airport zone, mixed"),
    341: ("Residential", "St. Marks Wildlife Refuge — nature reserve, rural"),
    255: ("Residential", "Mark Twain SP — state park, background rural"),
    239: ("Residential", "Alamo Lake — lake/nature park, Arizona"),
    242: ("Residential", "Potawatomi — near Potawatomi forest, Wisconsin"),
    350: ("Residential", "Daniel South — rural Wyoming background station"),
    293: ("Residential", "Garden — Anchorage suburban residential"),
    326: ("Mixed",       "Jackson NCORE — urban background, Mississippi"),
    334: ("Mixed",       "Miami Fire Station — urban mixed neighbourhood"),
    384: ("Mixed",       "CCNY — City College NY, dense urban mixed"),

    # ── PUERTO RICO ─────────────────────────────────────────────────────────
    345: ("Industrial",  "Cataño — major petrochemical/industrial municipality PR"),

    # ── MEXICO ──────────────────────────────────────────────────────────────
    359: ("Mixed",       "Estación Bomberos — fire station monitoring, urban mixed"),
    338: ("Mixed",       "Estación Hospital General — hospital area, urban mixed"),
    376: ("Mixed",       "Estación Nativitas — urban background station"),
}


# ── OSM TAG MAPS ──────────────────────────────────────────────────────────────

LANDUSE_MAP = {
    "industrial": "Industrial", "port": "Industrial", "railway": "Industrial",
    "quarry": "Industrial", "landfill": "Industrial", "commercial": "Industrial",
    "military": "Industrial", "retail": "Mixed", "construction": "Mixed",
    "residential": "Residential", "recreation_ground": "Residential",
    "cemetery": "Residential", "allotments": "Residential", "farmland": "Residential",
    "forest": "Residential", "grass": "Residential", "meadow": "Residential",
    "village_green": "Residential", "conservation": "Residential",
    "nature_reserve": "Residential", "orchard": "Residential",
}

AMENITY_MAP = {
    "school": "Residential", "university": "Residential", "college": "Residential",
    "hospital": "Residential", "clinic": "Residential", "park": "Residential",
    "playground": "Residential", "library": "Residential",
    "fire_station": "Mixed", "police": "Mixed", "townhall": "Mixed",
    "fuel": "Industrial", "waste_disposal": "Industrial", "recycling": "Industrial",
}

PLACE_MAP = {
    "suburb": "Residential", "neighbourhood": "Residential", "village": "Residential",
    "hamlet": "Residential", "farm": "Residential", "island": "Residential",
    "nature_reserve": "Residential", "town": "Mixed", "city": "Mixed",
}

INDUSTRIAL_NAME_KW = [
    "industrial", "port", "harbour", "harbor", "docks", "factory", "refinery",
    "terminal", "airport", "quarry", "mine", "steel", "chemical", "haven",
    "industrie", "fabriek", "kanaaldijk", "gaseous", "petroleum",
]

RESIDENTIAL_NAME_KW = [
    "park", "garden", "gardens", "green", "grove", "wood", "village",
    "residential", "meadow", "heath", "nature", "reserve", "wildlife",
    "rural", "school", "college", "university", "campus", "parque",
    "jardín", "bosque", "colegio", "liceo", "vondelpark", "westerpark",
]


# ── NOMINATIM HELPERS ─────────────────────────────────────────────────────────

def _query_nominatim(lat, lon, zoom=14):
    params = {
        "lat": lat, "lon": lon, "format": "jsonv2",
        "addressdetails": 1, "extratags": 1, "zoom": zoom
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(NOMINATIM_URL, params=params,
                             headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            logger.warning(f"    Timeout attempt {attempt}/{MAX_RETRIES}, retrying...")
            time.sleep(4 * attempt)
        except Exception as e:
            logger.warning(f"    Request error: {e}")
            time.sleep(4)
    raise RuntimeError(f"All attempts failed for lat={lat}, lon={lon}")


def _check_name_keywords(osm):
    address = osm.get("address") or {}
    combined = " ".join([
        address.get("suburb", ""), address.get("quarter", ""),
        address.get("city_district", ""), address.get("neighbourhood", ""),
        osm.get("display_name", ""),
    ]).lower()
    for kw in INDUSTRIAL_NAME_KW:
        if kw in combined:
            return "Industrial"
    for kw in RESIDENTIAL_NAME_KW:
        if kw in combined:
            return "Residential"
    return None


def _classify_from_osm(osm):
    extra    = osm.get("extratags") or {}
    address  = osm.get("address")   or {}
    category = osm.get("category",  "")
    osm_type = osm.get("type",      "")

    landuse = (extra.get("landuse") or address.get("landuse")
               or (osm_type if category == "landuse" else None))
    if landuse and landuse in LANDUSE_MAP:
        return LANDUSE_MAP[landuse], f"landuse={landuse}"

    if extra.get("industrial"):
        return "Industrial", f"industrial={extra['industrial']}"

    amenity = extra.get("amenity") or (osm_type if category == "amenity" else None)
    if amenity and amenity in AMENITY_MAP:
        return AMENITY_MAP[amenity], f"amenity={amenity}"

    place_type = osm_type if category == "place" else None
    if place_type and place_type in PLACE_MAP:
        return PLACE_MAP[place_type], f"place={place_type}"

    kw = _check_name_keywords(osm)
    if kw:
        return kw, "address_keyword"

    return "Mixed", f"no_tag_matched (cat={category}, type={osm_type})"


def _classify_with_zoom_fallback(lat, lon):
    for zoom in [14, 12, 10]:
        osm = _query_nominatim(lat, lon, zoom=zoom)
        zone, reason = _classify_from_osm(osm)
        if osm.get("category") != "highway" or zoom == 10:
            return zone, reason, zoom
        time.sleep(RATE_LIMIT)
    return zone, reason, 10


# ── MAIN ENTRY POINTS ─────────────────────────────────────────────────────────

def load_or_fetch_zones(
    locations_csv: str = "data/raw/unique_locations.csv",
    cache_path: Path   = CACHE_PATH
) -> dict:
    """
    Returns final {location_id -> zone} map.
    Order of precedence: expert override > cached API result > live API call > Mixed
    """
    # Load API cache (may be partial from previous interrupted run)
    api_cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            api_cache = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"✅ Loaded {len(api_cache)} cached API results")

    locations = pd.read_csv(locations_csv)

    # Fetch any stations not yet in API cache
    needed = [
        row for _, row in locations.iterrows()
        if int(row["location_id"]) not in api_cache
        and int(row["location_id"]) not in EXPERT_OVERRIDES  # no need to fetch overridden ones
    ]

    if needed:
        logger.info(f"🌐 Querying Nominatim for {len(needed)} uncached stations...")
        for i, row in enumerate(needed):
            loc_id = int(row["location_id"])
            lat    = float(row["latitude"])
            lon    = float(row["longitude"])
            name   = str(row.get("location_name", ""))
            try:
                zone, reason, zoom = _classify_with_zoom_fallback(lat, lon)
                api_cache[loc_id] = zone
                logger.info(f"  [{i+1:>3}/{len(needed)}] {name:<42} → {zone:<12} ({reason}, zoom={zoom})")
            except Exception as e:
                logger.warning(f"  ⚠️  FAILED {name}: {e} → Mixed")
                api_cache[loc_id] = "Mixed"
            time.sleep(RATE_LIMIT)
            if (i + 1) % 10 == 0:
                _save_cache(api_cache, cache_path)
                logger.info(f"  💾 Progress saved ({i+1}/{len(needed)})")
        _save_cache(api_cache, cache_path)
    else:
        logger.info("✅ All stations cached or overridden. No API calls needed.")

    # Build final map: start with API results, then apply expert overrides on top
    final_map = dict(api_cache)
    override_count = 0
    for loc_id, (zone, reason) in EXPERT_OVERRIDES.items():
        if loc_id in final_map and final_map[loc_id] != zone:
            logger.debug(f"  Override [{loc_id}]: {final_map[loc_id]} → {zone} ({reason})")
        final_map[loc_id] = zone
        override_count += 1

    logger.info(f"✅ Final map: {len(final_map)} stations "
                f"({override_count} expert overrides applied)")
    return final_map


def _save_cache(cache, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def assign_zones_to_df(df: pd.DataFrame, zone_map: dict) -> pd.DataFrame:
    """Add 'zone' column to df. Unknown location_ids → 'Mixed'."""
    df = df.copy()
    df["zone"] = df["location_id"].map(zone_map).fillna("Mixed")
    return df


# ── STANDALONE AUDIT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ZONE ASSIGNMENT — HYBRID (Nominatim + Expert Overrides)")
    print("="*60)

    zone_map = load_or_fetch_zones()

    counts = Counter(zone_map.values())
    print(f"\n📊 Zone Distribution:")
    for zone, count in sorted(counts.items()):
        print(f"   {zone:<12}: {count:>3} stations ({count/len(zone_map)*100:.0f}%)")

    locations = pd.read_csv("data/raw/unique_locations.csv")
    print(f"\n📋 Full Assignment (source shown):")
    for _, row in locations.sort_values(["country", "location_id"]).iterrows():
        lid    = int(row["location_id"])
        name   = str(row["location_name"])
        ctry   = str(row.get("country", ""))
        zone   = zone_map.get(lid, "Mixed")
        source = "EXPERT" if lid in EXPERT_OVERRIDES else "api"
        print(f"  [{lid:>4}] {name:<45} {ctry:<15} → {zone:<12} [{source}]")