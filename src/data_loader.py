import os
import urllib.request
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
DATA_DIR = "data"
LOCAL_FILE = os.path.join(DATA_DIR, "ibtracs.ALL.list.v04r01.csv")

def download_data():
    """Download the IBTrACS dataset if it doesn't already exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(LOCAL_FILE):
        logger.info(f"Downloading dataset from {URL} (this may take a few minutes)...")
        urllib.request.urlretrieve(URL, LOCAL_FILE)
        logger.info("Download complete.")
    else:
        logger.info("Dataset already exists locally. Skipping download.")

def load_and_filter_data() -> pd.DataFrame:
    """
    Load the IBTrACS dataset and filter it according to the requirements:
    - Target columns: USA_WIND, USA_PRES, LAT, LON, USA_RMW, STORM_SPEED, SID, SEASON
    - N.B. Some agencies use WMO_WIND instead. The experiment is based on WIND and PRES. 
      We'll primarily use USA_WIND/WMO_WIND logic, but let's select the 6 core features.
    - 2004-2023 limit.
    """
    logger.info("Loading dataset into pandas...")
    
    # Read CSV. IBTrACS has a second row with units. We skip row 1 (the units row).
    # low_memory=False to avoid DtypeWarning on mixed types in unused columns.
    df = pd.read_csv(LOCAL_FILE, skiprows=[1], low_memory=False)
    
    logger.info(f"Initial dataset size: {len(df)} rows.")
    
    # Filter by year (2004-2023)
    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    df = df[(df['SEASON'] >= 2004) & (df['SEASON'] <= 2023)]
    
    logger.info(f"Size after filtering for 2004-2023: {len(df)} rows.")

    # The 6 core features required:
    # 1. Maximum sustained wind speed: try WMO_WIND, if NaN use USA_WIND
    # 2. Minimum sea-level pressure: try WMO_PRES, if NaN use USA_PRES
    # 3. Latitude (LAT)
    # 4. Longitude (LON)
    # 5. Radius of maximum winds (USA_RMW)
    # 6. Translational speed (STORM_SPEED)
    
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['USA_RMW'] = pd.to_numeric(df['USA_RMW'], errors='coerce')
    df['STORM_SPEED'] = pd.to_numeric(df['STORM_SPEED'], errors='coerce')
    
    # Wind and Pressure coalescing
    wmo_wind = pd.to_numeric(df['WMO_WIND'], errors='coerce')
    usa_wind = pd.to_numeric(df['USA_WIND'], errors='coerce')
    df['WIND'] = wmo_wind.combine_first(usa_wind)
    
    wmo_pres = pd.to_numeric(df['WMO_PRES'], errors='coerce')
    usa_pres = pd.to_numeric(df['USA_PRES'], errors='coerce')
    df['PRES'] = wmo_pres.combine_first(usa_pres)
    
    # Core columns needed for downstream tasks
    core_cols = ['SID', 'SEASON', 'LAT', 'LON', 'USA_RMW', 'STORM_SPEED', 'WIND', 'PRES']
    
    df_filtered = df[core_cols]
    
    # Drop any row with missing values in the 6 specified features
    features = ['WIND', 'PRES', 'LAT', 'LON', 'USA_RMW', 'STORM_SPEED']
    df_clean = df_filtered.dropna(subset=features).copy()
    
    logger.info(f"Size after dropping missing values in core features: {len(df_clean)} rows.")
    
    return df_clean

if __name__ == "__main__":
    download_data()
    data = load_and_filter_data()
    print(data.head())
