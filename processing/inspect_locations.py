"""
Inspect all unique locations and countries across all parquet files.
Helps decide how to classify zones without re-fetching data.
"""

import pandas as pd
from pathlib import Path

PARQUET_DIR = Path("data/raw/parquet")

def inspect_all_locations(directory=PARQUET_DIR):
    parquet_files = sorted(Path(directory).glob("*.parquet"))

    if not parquet_files:
        print("❌ No parquet files found in", directory)
        return

    print(f"📂 Found {len(parquet_files)} parquet files\n")

    all_dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=[
                col for col in pd.read_parquet(f, engine="pyarrow").columns
                if col in ["location_id", "location_name", "country", "latitude", "longitude"]
            ])
            all_dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ Could not read {f.name}: {e}")

    if not all_dfs:
        print("❌ No data loaded.")
        return

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    # Unique locations
    if "location_name" in combined.columns:
        unique_locations = (
            combined[["location_id", "location_name", "country", "latitude", "longitude"]]
            .drop_duplicates(subset=["location_id"])
            .sort_values("country")
            .reset_index(drop=True)
        )

        print("=" * 70)
        print(f"🌍 UNIQUE LOCATIONS ({len(unique_locations)} total)")
        print("=" * 70)
        print(unique_locations.to_string(index=False))

        print("\n" + "=" * 70)
        print("🗺️  UNIQUE COUNTRIES")
        print("=" * 70)
        country_counts = unique_locations["country"].value_counts()
        for country, count in country_counts.items():
            print(f"  {country}: {count} station(s)")

        print("\n" + "=" * 70)
        print("📋 ALL LOCATION NAMES (raw, for keyword analysis)")
        print("=" * 70)
        for _, row in unique_locations.iterrows():
            print(f"  [{row.get('location_id', '?')}] {row.get('location_name', '?')}  "
                  f"| {row.get('country', '?')}  "
                  f"| lat={row.get('latitude', '?'):.4f}, lon={row.get('longitude', '?'):.4f}")

        # Save to CSV for easy review
        out_csv = Path("data/raw/unique_locations.csv")
        unique_locations.to_csv(out_csv, index=False)
        print(f"\n💾 Saved to {out_csv}")
        print("   → Open this CSV and manually add a 'zone' column if needed.")
    else:
        print("⚠️ 'location_name' column not found in parquet files.")
        print("   Available columns:", combined.columns.tolist())


if __name__ == "__main__":
    inspect_all_locations()