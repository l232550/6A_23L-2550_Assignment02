"""
preprocess.py
-------------
Full preprocessing pipeline for OpenAQ v3 data.

Outputs:
  data/processed/task1_pca_dataset.parquet     ← PCA-ready: station-level means, scaled
  data/processed/task1_pca_raw.parquet         ← PCA stations, wide format, unscaled
  data/processed/task2_temporal_dataset.parquet
  data/processed/task3_distribution_dataset.parquet
  data/processed/full_dataset.parquet

PCA preparation flow:
  raw long-format → wide (pivot) → station means → drop incomplete →
  assign zones → StandardScaler → ready for PCA
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

from config import PARAMS, HEALTH_THRESHOLD, EXTREME_HAZARD, PROCESSED_DIR
from processing.assign_zones import load_or_fetch_zones, assign_zones_to_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exact parameter names as they appear in the fetched data
REQUIRED_PARAMS = ['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']


# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

def load_all_stations(raw_dir: str = "data/raw/parquet") -> pd.DataFrame:
    """Load all station parquet files and combine."""
    data_path = Path(raw_dir)
    
    # Support both monthly files (station_X_monthY.parquet) and annual (station_X_2025.parquet)
    parquet_files = list(data_path.glob("station_*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files in {raw_dir}")

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")

    if not dfs:
        raise ValueError("No data could be loaded from parquet files")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Raw combined: {len(combined):,} rows | columns: {list(combined.columns)}")
    return combined


# ── STEP 2: NORMALISE COLUMN NAMES ────────────────────────────────────────────

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent column names regardless of fetcher version.
    Handles: datetimeUtc / timestamp, relative_humidity / humidity
    """
    # Timestamp
    if 'datetimeUtc' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'datetimeUtc': 'timestamp'})

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Humidity naming
    if 'relative_humidity' in df.columns and 'humidity' not in df.columns:
        df = df.rename(columns={'relative_humidity': 'humidity'})

    # Normalise parameter values in the 'parameter' column too
    if 'parameter' in df.columns:
        df['parameter'] = df['parameter'].replace({'relative_humidity': 'humidity'})

    logger.info(f"Columns after normalisation: {list(df.columns)}")
    return df


# ── STEP 3: WIDE FORMAT ───────────────────────────────────────────────────────

def create_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from long format (one row per measurement) to wide format
    (one row per timestamp × station, columns = parameters).
    """
    logger.info("Pivoting to wide format...")

    # Keep only rows for our target parameters
    if 'parameter' in df.columns:
        df = df[df['parameter'].isin(REQUIRED_PARAMS)].copy()

    # Identify available index columns
    index_cols = ['timestamp', 'location_id']
    for optional in ['location_name', 'country', 'latitude', 'longitude']:
        if optional in df.columns:
            index_cols.append(optional)

    try:
        wide = df.pivot_table(
            index=index_cols,
            columns='parameter',
            values='value',
            aggfunc='mean'
        ).reset_index()

        # Flatten column names
        wide.columns.name = None
        logger.info(f"Wide format: {wide.shape} | params present: "
                    f"{[c for c in REQUIRED_PARAMS if c in wide.columns]}")
        return wide

    except Exception as e:
        logger.error(f"Pivot failed: {e}")
        raise


# ── STEP 4: ZONE ASSIGNMENT ───────────────────────────────────────────────────

def assign_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign zone labels using Nominatim reverse geocoding (cached).
    Falls back gracefully if cache/API unavailable.
    """
    logger.info("Assigning zones via Nominatim cache...")
    try:
        zone_map = load_or_fetch_zones()
        df = assign_zones_to_df(df, zone_map)
        counts = df['zone'].value_counts().to_dict()
        logger.info(f"Zone distribution (rows): {counts}")
    except Exception as e:
        logger.warning(f"Zone assignment failed: {e} — defaulting all to 'Mixed'")
        df['zone'] = 'Mixed'
    return df


# ── STEP 5: MISSING VALUE HANDLING ───────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
    """
    Fill missing parameter values using forward/back fill within each station,
    then station mean, then global mean.
    """
    param_cols = [c for c in params if c in df.columns]
    if not param_cols:
        return df

    missing_before = df[param_cols].isnull().sum().sum()
    if missing_before == 0:
        logger.info("No missing values found")
        return df

    logger.info(f"Missing values before fill: {missing_before:,}")

    df = df.sort_values(['location_id', 'timestamp'])

    # Forward fill then backward fill within station
    df[param_cols] = (
        df.groupby('location_id')[param_cols]
        .transform(lambda x: x.ffill().bfill())   # pandas ≥ 2.0 syntax
    )

    # Station mean for any remaining gaps
    df[param_cols] = (
        df.groupby('location_id')[param_cols]
        .transform(lambda x: x.fillna(x.mean()))
    )

    # Global mean as final fallback
    for col in param_cols:
        df[col] = df[col].fillna(df[col].mean())

    missing_after = df[param_cols].isnull().sum().sum()
    logger.info(f"Missing values after fill:  {missing_after:,}")
    return df


# ── STEP 6: TIME FEATURES ─────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, month, season, weekend columns."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['hour']      = df['timestamp'].dt.hour
    df['day']       = df['timestamp'].dt.day
    df['month']     = df['timestamp'].dt.month
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekend']   = df['dayofweek'].isin([5, 6]).astype(int)
    df['season']    = df['month'].map({
        12: 'Winter', 1: 'Winter',  2: 'Winter',
         3: 'Spring', 4: 'Spring',  5: 'Spring',
         6: 'Summer', 7: 'Summer',  8: 'Summer',
         9: 'Fall',  10: 'Fall',   11: 'Fall'
    })
    return df


# ── STEP 7: HEALTH FLAGS ──────────────────────────────────────────────────────

def add_health_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add PM2.5 threshold violation and extreme hazard flags."""
    if 'pm25' not in df.columns:
        return df

    df['date']            = df['timestamp'].dt.date
    df['pm25_daily_avg']  = df.groupby(['location_id', 'date'])['pm25'].transform('mean')
    df['threshold_violation'] = (df['pm25_daily_avg'] > HEALTH_THRESHOLD).astype(int)
    df['extreme_hazard']      = (df['pm25'] > EXTREME_HAZARD).astype(int)

    logger.info(f"Threshold violations: {df['threshold_violation'].mean()*100:.2f}% of days")
    logger.info(f"Extreme hazard events: {df['extreme_hazard'].mean()*100:.2f}% of readings")
    return df


# ── STEP 8: PCA-READY DATASET ─────────────────────────────────────────────────

def build_pca_dataset(wide_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two PCA datasets:

    1. pca_raw   — one row per STATION (mean of all readings across the year).
                   Contains location metadata + zone + unscaled parameter means.
                   Used for plotting and reference.

    2. pca_scaled — same stations, only the 6 scaled parameter columns.
                   Fed directly into sklearn PCA.

    Only stations with ALL 6 parameters are included.
    """
    logger.info("\n" + "="*60)
    logger.info("BUILDING PCA DATASET")
    logger.info("="*60)

    available = [p for p in REQUIRED_PARAMS if p in wide_df.columns]
    logger.info(f"Parameters available: {available}")

    if len(available) < 6:
        logger.warning(f"Only {len(available)}/6 parameters available. "
                       f"Missing: {set(REQUIRED_PARAMS) - set(available)}")

    # ── Aggregate to station level (annual means) ──────────────────────────
    # PCA operates on one vector per station, not per timestamp.
    # Using annual mean captures the typical pollution profile of each location.
    meta_cols = ['location_id', 'location_name', 'country',
                 'latitude', 'longitude', 'zone']
    meta_cols = [c for c in meta_cols if c in wide_df.columns]

    station_means = (
        wide_df.groupby(meta_cols, as_index=False)[available]
        .mean()
    )

    logger.info(f"Station-level aggregation: {len(station_means)} stations")

    # ── Keep only stations with ALL 6 parameters ───────────────────────────
    complete_mask = station_means[available].notna().all(axis=1)
    pca_raw = station_means[complete_mask].copy().reset_index(drop=True)

    dropped = len(station_means) - len(pca_raw)
    logger.info(f"Stations with all {len(available)} params: {len(pca_raw)} "
                f"(dropped {dropped} incomplete)")

    if pca_raw.empty:
        logger.error("No stations have all 6 parameters — cannot build PCA dataset")
        return pd.DataFrame(), pd.DataFrame()

    # ── Log zone distribution ──────────────────────────────────────────────
    if 'zone' in pca_raw.columns:
        zone_dist = pca_raw['zone'].value_counts().to_dict()
        logger.info(f"Zone distribution in PCA set: {zone_dist}")

    # ── Standardise ────────────────────────────────────────────────────────
    # PCA is sensitive to scale. StandardScaler gives each variable
    # zero mean and unit variance so no single variable dominates
    # just because it has larger units (e.g. temperature in °C vs pm25 in µg/m³)
    scaler     = StandardScaler()
    scaled_arr = scaler.fit_transform(pca_raw[available])
    pca_scaled = pd.DataFrame(scaled_arr, columns=available)

    logger.info(f"Standardised {len(available)} parameters across {len(pca_raw)} stations")
    logger.info(f"Mean after scaling (should be ~0): "
                f"{np.abs(pca_scaled.mean().values).mean():.4f}")
    logger.info(f"Std  after scaling (should be ~1): "
                f"{pca_scaled.std().values.mean():.4f}")

    return pca_raw, pca_scaled


# ── STEP 9: TASK DATASETS ─────────────────────────────────────────────────────

def create_task_datasets(
    wide_df: pd.DataFrame,
    pca_raw: pd.DataFrame,
    pca_scaled: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Bundle all task-specific datasets into one dict."""
    datasets = {}

    # Task 1: PCA
    if not pca_raw.empty:
        # Attach scaled columns alongside metadata for convenience
        pca_combined = pca_raw.copy()
        scaled_renamed = pca_scaled.rename(
            columns={p: f"{p}_scaled" for p in REQUIRED_PARAMS if p in pca_scaled.columns}
        )
        pca_combined = pd.concat([pca_combined, scaled_renamed], axis=1)
        datasets['task1_pca'] = pca_combined
        datasets['task1_pca_scaled_only'] = pca_scaled  # clean matrix for PCA input
        logger.info(f"Task 1 PCA dataset: {len(pca_combined)} stations")

    # Task 2: Temporal (full time series, sampled for visualisation)
    sample_size = min(100_000, len(wide_df))
    datasets['task2_temporal'] = wide_df.sample(n=sample_size, random_state=42).copy()
    logger.info(f"Task 2 temporal dataset: {len(datasets['task2_temporal']):,} rows")

    # Task 3: Distribution (industrial zones, full time series)
    if 'zone' in wide_df.columns:
        industrial = wide_df[wide_df['zone'] == 'Industrial'].copy()
        datasets['task3_distribution'] = industrial if not industrial.empty else wide_df.copy()
        logger.info(f"Task 3 distribution dataset: {len(datasets['task3_distribution']):,} rows")
    else:
        datasets['task3_distribution'] = wide_df.copy()

    # Full processed reference
    datasets['full'] = wide_df.copy()

    return datasets


# ── STEP 10: SAVE ─────────────────────────────────────────────────────────────

def save_datasets(datasets: Dict[str, pd.DataFrame]):
    """Save all datasets to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        if df.empty:
            logger.warning(f"Skipping {name} — empty dataframe")
            continue
        out = PROCESSED_DIR / f"{name}_dataset.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Saved {out.name} ({len(df):,} rows)")

        # Small CSV sample for quick inspection
        if name in ('task1_pca', 'task2_temporal'):
            df.head(500).to_csv(PROCESSED_DIR / f"{name}_sample.csv", index=False)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def preprocess_pipeline() -> Dict[str, pd.DataFrame]:
    logger.info("="*60)
    logger.info("PREPROCESSING PIPELINE START")
    logger.info("="*60)

    # 1. Load
    raw_df = load_all_stations()

    # 2. Normalise columns
    raw_df = normalise_columns(raw_df)

    # 3. Wide format
    wide_df = create_wide_format(raw_df)

    # 4. Zones
    wide_df = assign_zones(wide_df)

    # 5. Missing values
    wide_df = handle_missing_values(wide_df, REQUIRED_PARAMS)

    # 6. Time features
    wide_df = add_time_features(wide_df)

    # 7. Health flags
    wide_df = add_health_flags(wide_df)

    # 8. PCA-ready datasets
    pca_raw, pca_scaled = build_pca_dataset(wide_df)

    # 9. All task datasets
    datasets = create_task_datasets(wide_df, pca_raw, pca_scaled)

    # 10. Save
    save_datasets(datasets)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Full dataset:       {wide_df.shape}")
    logger.info(f"Total stations:     {wide_df['location_id'].nunique()}")
    logger.info(f"PCA stations:       {len(pca_raw)}")
    logger.info(f"Date range:         {wide_df['timestamp'].min()} → {wide_df['timestamp'].max()}")
    if 'zone' in wide_df.columns:
        logger.info(f"Zone distribution:  {wide_df.drop_duplicates('location_id')['zone'].value_counts().to_dict()}")

    return datasets


if __name__ == "__main__":
    datasets = preprocess_pipeline()

    print("\n" + "="*60)
    print("DATASETS CREATED")
    print("="*60)
    for name, df in datasets.items():
        if not df.empty:
            print(f"\n{name.upper()}:")
            print(f"  Rows:     {len(df):,}")
            if 'location_id' in df.columns:
                print(f"  Stations: {df['location_id'].nunique()}")
            if 'zone' in df.columns:
                print(f"  Zones:    {df['zone'].value_counts().to_dict()}")
            print(f"  Columns:  {list(df.columns)}")