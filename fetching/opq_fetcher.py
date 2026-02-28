"""
OpenAQ Data Fetcher for Urban Environmental Intelligence Challenge
FETCHES EXACTLY WHAT THE MANUAL REQUIRES:
- 100 stations (any IDs)
- Year 2025
- Parameters: pm25, pm10, no2, o3, temperature, humidity
"""

import logging
import pandas as pd
import time
import calendar
from pathlib import Path
import requests
import os
from typing import List, Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MAX_STATIONS = 100
YEAR = 2025
PARAMS = ['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']
OPENAQ_API_KEY = os.getenv('OPENAQ_API_KEY', '5f8d24a8d1352d6ffdf8b1a3797252fa2611c9b01119e17998bc0b6bff2533c4')

# Parameter ID mapping
PARAMETER_IDS = {
    'pm25': 2,
    'pm10': 1,
    'no2': 7,
    'o3': 10,
    'temperature': 75,
    'humidity': 76,
}


class OpenAQFetcher:
    def __init__(self):
        self.base_url = "https://api.openaq.org/v3"
        self.locations_url = f"{self.base_url}/locations"
        self.sensors_url = f"{self.base_url}/sensors"
        
        # Rate limiting
        self.last_request = 0
        self.min_interval = 1.0
        
        # Headers
        self.headers = {
            "X-API-Key": OPENAQ_API_KEY,
            "Accept": "application/json",
        }
        
        # Data directory
        self.raw_dir = Path("data/raw/parquet")
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        
        # Progress tracking
        self.total_requests = 0
        self.start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("OPENAQ DATA FETCHER")
        logger.info("=" * 60)
        logger.info(f"Target: {MAX_STATIONS} stations × {len(PARAMS)} parameters × Year {YEAR}")
        logger.info(f"Parameters: {PARAMS}")
        logger.info("=" * 60)

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
        self.total_requests += 1

    def _get(self, url: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
        self._rate_limit()
        
        for attempt in range(retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    headers=self.headers, 
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Request failed: {e}")
                    return {}
        return {}

    def verify_sensor_has_2025_data(self, sensor_id: int) -> bool:
        """Quickly verify if a sensor has any 2025 data"""
        if not sensor_id:
            return False
        
        # Check just January 2025 to verify data exists
        params = {
            "datetime_from": f"{YEAR}-01-01T00:00:00Z",
            "datetime_to": f"{YEAR}-01-31T23:59:59Z",
            "limit": 1,
        }
        
        data = self._get(
            f"{self.sensors_url}/{sensor_id}/measurements",
            params=params
        )
        
        return bool(data and data.get("results"))

    def find_stations_for_2025(self, needed_stations: int = MAX_STATIONS) -> List[Dict]:
        """
        Find stations that have 2025 data for our parameters
        """
        stations = []
        page = 1
        page_size = 100
        max_pages = 50  # Search through up to 5000 stations
        
        logger.info(f"\n🔍 Searching for {needed_stations} stations with {YEAR} data...")
        
        # Track parameters found to ensure we get all 6 types
        params_found = set()
        
        while len(stations) < needed_stations and page <= max_pages:
            logger.info(f"Scanning page {page}...")
            
            try:
                # Get stations
                params = {
                    "limit": page_size,
                    "page": page,
                }
                
                data = self._get(self.locations_url, params=params)
                if not data or not data.get("results"):
                    break
                
                results = data.get("results", [])
                
                for station_data in results:
                    if len(stations) >= needed_stations:
                        break
                    
                    station_id = station_data.get("id")
                    if not station_id:
                        continue
                    
                    # Get sensors for this station
                    sensors = station_data.get("sensors", [])
                    if not sensors:
                        continue
                    
                    # Check each sensor for our parameters AND 2025 data
                    station_sensors = []
                    
                    for sensor in sensors:
                        param = sensor.get("parameter", {})
                        param_name = param.get("name", "").lower()
                        
                        if param_name in PARAMS:
                            sensor_id = sensor.get("id")
                            
                            # Verify this sensor has 2025 data
                            if self.verify_sensor_has_2025_data(sensor_id):
                                station_sensors.append({
                                    "id": sensor_id,
                                    "parameter": param_name,
                                    "parameter_id": param.get("id"),
                                    "units": param.get("units")
                                })
                                params_found.add(param_name)
                    
                    # Only include stations with at least 2 of our parameters
                    if len(station_sensors) >= 2:
                        stations.append({
                            "id": station_id,
                            "name": station_data.get("name", "Unknown"),
                            "country": station_data.get("country", {}).get("name", "Unknown"),
                            "latitude": station_data.get("coordinates", {}).get("latitude"),
                            "longitude": station_data.get("coordinates", {}).get("longitude"),
                            "sensors": station_sensors
                        })
                        
                        logger.info(f"  ✓ Found station {station_id} in {stations[-1]['country']} "
                                  f"with: {[s['parameter'] for s in station_sensors]}")
                
                page += 1
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                page += 1
        
        logger.info(f"\n✅ Found {len(stations)} stations with {YEAR} data")
        logger.info(f"Parameters covered: {params_found}")
        
        return stations[:needed_stations]

    def fetch_sensor_2025_data(self, sensor_id: int, parameter: str) -> pd.DataFrame:
        """Fetch ALL 2025 data for a single sensor"""
        if not sensor_id:
            return pd.DataFrame()
        
        all_measurements = []
        
        # Fetch month by month to avoid timeouts
        for month in range(1, 13):
            start_date = f"{YEAR}-{month:02d}-01"
            last_day = calendar.monthrange(YEAR, month)[1]
            end_date = f"{YEAR}-{month:02d}-{last_day}"
            
            start_datetime = f"{start_date}T00:00:00Z"
            end_datetime = f"{end_date}T23:59:59Z"
            
            page = 1
            page_size = 1000
            
            while True:
                try:
                    params = {
                        "datetime_from": start_datetime,
                        "datetime_to": end_datetime,
                        "limit": page_size,
                        "page": page,
                    }
                    
                    data = self._get(
                        f"{self.sensors_url}/{sensor_id}/measurements",
                        params=params
                    )
                    
                    if not data:
                        break
                    
                    results = data.get("results", [])
                    if not results:
                        break
                    
                    for result in results:
                        period = result.get("period", {})
                        datetime_from = period.get("datetimeFrom", {})
                        timestamp = datetime_from.get("utc")
                        value = result.get("value")
                        
                        if timestamp and value is not None:
                            all_measurements.append({
                                "timestamp": timestamp,
                                "value": float(value),
                                "parameter": parameter,
                                "sensor_id": sensor_id,
                                "month": month,
                                "year": YEAR
                            })
                    
                    # Check if we need more pages
                    if len(results) < page_size:
                        break
                        
                    page += 1
                    
                except Exception as e:
                    logger.error(f"Error: {e}")
                    break
            
            # Small delay between months
            time.sleep(0.5)
        
        df = pd.DataFrame(all_measurements)
        
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            logger.info(f"    Got {len(df)} records for {parameter}")
        
        return df

    def fetch_station_2025_data(self, station: Dict) -> pd.DataFrame:
        """Fetch ALL 2025 data for one station"""
        logger.info(f"\n📊 Processing station {station['id']} - {station['name']}")
        logger.info(f"   Country: {station['country']}")
        logger.info(f"   Sensors: {[s['parameter'] for s in station['sensors']]}")
        
        station_data = []
        
        for sensor in station['sensors']:
            df = self.fetch_sensor_2025_data(sensor['id'], sensor['parameter'])
            
            if not df.empty:
                df['location_id'] = station['id']
                df['location_name'] = station['name']
                df['latitude'] = station['latitude']
                df['longitude'] = station['longitude']
                df['country'] = station['country']
                station_data.append(df)
        
        if station_data:
            result = pd.concat(station_data, ignore_index=True)
            result = result.sort_values('timestamp')
            logger.info(f"   ✅ Total: {len(result)} records")
            return result
        
        logger.warning(f"   ❌ No data for station {station['id']}")
        return pd.DataFrame()

    def save_station_data(self, df: pd.DataFrame, station_id: int):
        """Save data in the format matching the sample CSV"""
        if df.empty:
            return
        
        # Format columns like the sample CSV
        df_formatted = pd.DataFrame({
            'location_id': df['location_id'],
            'location_name': df['location_name'],
            'parameter': df['parameter'],
            'value': df['value'],
            'unit': 'µg/m³',  # Default unit
            'datetimeUtc': df['timestamp'],
            'latitude': df['latitude'],
            'longitude': df['longitude'],
            'country': df['country']
        })
        
        # Save to parquet
        filename = self.raw_dir / f"station_{station_id}_2025.parquet"
        df_formatted.to_parquet(filename, index=False)
        logger.info(f"   💾 Saved to {filename}")

    def run(self):
        """Main execution - Fetch 100 stations × 2025 data"""
        logger.info("\n" + "=" * 60)
        logger.info("STARTING DATA FETCH")
        logger.info(f"Goal: {MAX_STATIONS} stations × {len(PARAMS)} parameters × Year {YEAR}")
        estimated_points = MAX_STATIONS * len(PARAMS) * 365 * 24
        logger.info(f"Estimated data points: ~{estimated_points:,}")
        logger.info("=" * 60)
        
        # Step 1: Find stations with 2025 data
        stations = self.find_stations_for_2025(MAX_STATIONS)
        
        if not stations:
            logger.error("❌ No stations found with 2025 data!")
            return
        
        logger.info(f"\n🎯 Found {len(stations)} stations to process")
        
        # Step 2: Fetch data for each station
        successful = 0
        total_records = 0
        
        for i, station in enumerate(stations, 1):
            logger.info(f"\n[{i}/{len(stations)}]")
            
            df = self.fetch_station_2025_data(station)
            
            if not df.empty:
                self.save_station_data(df, station['id'])
                successful += 1
                total_records += len(df)
            
            # Pause between stations
            if i < len(stations):
                time.sleep(2)
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info(f"Year: {YEAR}")
        logger.info(f"Stations fetched: {successful}/{MAX_STATIONS}")
        logger.info(f"Total records: {total_records:,}")
        logger.info(f"Parameters: {PARAMS}")
        logger.info(f"API requests: {self.total_requests}")
        logger.info(f"Time: {time.time() - self.start_time:.2f} seconds")
        logger.info("=" * 60)


if __name__ == "__main__":
    fetcher = OpenAQFetcher()
    fetcher.run()