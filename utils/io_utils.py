import pandas as pd
import pickle
from pathlib import Path
from config import RAW_DIR, PROCESSED_DIR

def save_raw(df_chunk, station_id):
    """Save raw chunk."""
    fname = RAW_DIR / f"station_{station_id}.parquet"
    df_chunk.to_parquet(fname)
    return fname

def load_processed():
    """Load processed data."""
    return pd.read_parquet(PROCESSED_DIR / "main_dataset.parquet")

def save_model(model, name):
    """Save sklearn model."""
    fname = PROCESSED_DIR / f"{name}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(model, f)

def load_model(name):
    """Load model."""
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
