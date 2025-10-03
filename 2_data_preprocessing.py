#!/usr/bin/env python3
"""
NASA Weather Data Preprocessing Pipeline
Convert raw NASA data into training-ready format for WeatherWise ML models

This script processes:
1. NASA GPM IMERG precipitation NetCDF files
2. NASA MERRA-2 meteorological NetCDF files  
3. Creates unified training datasets
4. Performs quality control and validation
5. Generates features for ML training

Usage:
    python 2_data_preprocessing.py --input-dir data/raw --output-dir data/processed
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
import logging
import argparse
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import configuration  
from config import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file']),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NASADataProcessor:
    def __init__(self, input_dir=None, output_dir=None):
        """Initialize NASA data processor"""
        self.input_dir = input_dir or RAW_DATA_DIR
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'training'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'validation'), exist_ok=True)
        
        logger.info(f"📂 Input directory: {self.input_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        # Initialize database connection
        self.db_path = DATABASE_CONFIG['sqlite']['path']
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for processed data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    latitude REAL,
                    longitude REAL,
                    dataset_type TEXT,
                    temperature REAL,
                    precipitation REAL,
                    humidity REAL,
                    pressure REAL,
                    wind_u REAL,
                    wind_v REAL,
                    wind_speed REAL,
                    wind_direction REAL,
                    solar_radiation REAL,
                    data_quality_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    latitude REAL,
                    longitude REAL,
                    region TEXT,
                    temperature REAL,
                    temperature_normalized REAL,
                    precipitation REAL,
                    precipitation_log REAL,
                    humidity REAL,
                    pressure REAL,
                    wind_speed REAL,
                    wind_direction REAL,
                    solar_radiation REAL,
                    cloud_cover REAL,
                    visibility REAL,
                    weather_risk_features TEXT,  -- JSON string
                    event_suitability_features TEXT,  -- JSON string
                    data_quality_flags TEXT,  -- JSON string
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    latitude REAL,
                    longitude REAL,
                    event_type TEXT,
                    features TEXT,  -- JSON feature vector
                    weather_risk_score REAL,
                    event_suitability_score REAL,
                    precipitation_risk REAL,
                    temperature_risk REAL,
                    wind_risk REAL,
                    overall_risk REAL,
                    split_type TEXT,  -- train/validation/test
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            
    def process_imerg_files(self, file_pattern="*.nc4"):
        """Process NASA GPM IMERG precipitation files"""
        logger.info("🌧️  Processing IMERG precipitation data...")
        
        imerg_dir = os.path.join(self.input_dir, 'imerg')
        if not os.path.exists(imerg_dir):
            logger.warning(f"⚠️  IMERG directory not found: {imerg_dir}")
            return []
        
        # Find all IMERG NetCDF files
        imerg_files = []
        for root, dirs, files in os.walk(imerg_dir):
            for file in files:
                if file.endswith('.nc4') or file.endswith('.nc'):
                    imerg_files.append(os.path.join(root, file))
        
        logger.info(f"📊 Found {len(imerg_files)} IMERG files")
        
        processed_data = []
        
        for i, file_path in enumerate(imerg_files[:10]):  # Limit for testing
            try:
                logger.info(f"⚙️  Processing {i+1}/{len(imerg_files[:10])}: {os.path.basename(file_path)}")
                
                # Read NetCDF file
                with xr.open_dataset(file_path) as ds:
                    # Extract precipitation data
                    precip = ds.precipitationCal if 'precipitationCal' in ds else ds.precipitation
                    
                    # Get coordinates
                    lats = ds.lat.values
                    lons = ds.lon.values
                    
                    # Extract date from filename or dataset
                    if 'time' in ds:
                        date = pd.to_datetime(ds.time.values[0])
                    else:
                        # Extract from filename
                        filename = os.path.basename(file_path)
                        date_str = filename.split('.')[3][:8]  # YYYYMMDD
                        date = datetime.strptime(date_str, '%Y%m%d')
                    
                    # Convert to pandas DataFrame for easier processing
                    precip_data = precip.values.flatten()
                    
                    # Create coordinate grids
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    lon_flat = lon_grid.flatten()
                    lat_flat = lat_grid.flatten()
                    
                    # Remove invalid data
                    valid_mask = ~np.isnan(precip_data) & (precip_data >= 0)
                    
                    if np.sum(valid_mask) > 0:
                        df = pd.DataFrame({
                            'date': date,
                            'latitude': lat_flat[valid_mask],
                            'longitude': lon_flat[valid_mask], 
                            'precipitation': precip_data[valid_mask],
                            'dataset_type': 'IMERG'
                        })
                        
                        # Sample data to reduce size (keep every 10th point)
                        df_sampled = df.iloc[::10].copy()
                        processed_data.append(df_sampled)
                
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                continue
        
        if processed_data:
            # Combine all data
            combined_df = pd.concat(processed_data, ignore_index=True)
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, 'imerg_processed.csv')
            combined_df.to_csv(output_path, index=False)
            
            # Save to database
            self.save_to_database(combined_df, 'raw_weather_data')
            
            logger.info(f"✅ Processed IMERG data: {len(combined_df)} records")
            logger.info(f"💾 Saved to: {output_path}")
            
            return combined_df
        else:
            logger.warning("⚠️  No IMERG data processed")
            return pd.DataFrame()
    
    def process_merra2_files(self, collection='M2T1NXSLV'):
        """Process NASA MERRA-2 meteorological files"""
        logger.info(f"🌡️  Processing MERRA-2 {collection} data...")
        
        merra2_dir = os.path.join(self.input_dir, 'merra2', collection)
        if not os.path.exists(merra2_dir):
            logger.warning(f"⚠️  MERRA-2 directory not found: {merra2_dir}")
            return []
        
        # Find all MERRA-2 files
        merra2_files = []
        for root, dirs, files in os.walk(merra2_dir):
            for file in files:
                if file.endswith('.nc4') or file.endswith('.nc'):
                    merra2_files.append(os.path.join(root, file))
        
        logger.info(f"📊 Found {len(merra2_files)} MERRA-2 files")
        
        processed_data = []
        
        for i, file_path in enumerate(merra2_files[:5]):  # Limit for testing
            try:
                logger.info(f"⚙️  Processing {i+1}/{len(merra2_files[:5])}: {os.path.basename(file_path)}")
                
                with xr.open_dataset(file_path) as ds:
                    # Get coordinates
                    lats = ds.lat.values
                    lons = ds.lon.values
                    times = pd.to_datetime(ds.time.values)
                    
                    # Extract variables
                    variables = {}
                    if 'T2M' in ds:
                        variables['temperature'] = ds.T2M - 273.15  # K to C
                    if 'QV2M' in ds:
                        variables['humidity'] = ds.QV2M * 1000  # kg/kg to g/kg
                    if 'PS' in ds:
                        variables['pressure'] = ds.PS / 100  # Pa to hPa
                    if 'U2M' in ds:
                        variables['wind_u'] = ds.U2M
                    if 'V2M' in ds:
                        variables['wind_v'] = ds.V2M
                    if 'PRECTOTCORR' in ds:
                        variables['precipitation'] = ds.PRECTOTCORR * 24  # mm/s to mm/day
                    if 'SWGDN' in ds:
                        variables['solar_radiation'] = ds.SWGDN
                    
                    # Process each time step
                    for t_idx, time in enumerate(times[:24]):  # First 24 hours only
                        data_records = []
                        
                        # Sample coordinates (every 5th point to reduce size)
                        lat_sample = lats[::5]  
                        lon_sample = lons[::5]
                        
                        for lat in lat_sample:
                            for lon in lon_sample:
                                # Get nearest grid point
                                lat_idx = np.argmin(np.abs(lats - lat))
                                lon_idx = np.argmin(np.abs(lons - lon))
                                
                                record = {
                                    'date': time.date(),
                                    'latitude': float(lat),
                                    'longitude': float(lon),
                                    'dataset_type': 'MERRA2'
                                }
                                
                                # Extract variable values
                                for var_name, var_data in variables.items():
                                    try:
                                        value = float(var_data.isel(time=t_idx, lat=lat_idx, lon=lon_idx).values)
                                        if not np.isnan(value):
                                            record[var_name] = value
                                    except:
                                        record[var_name] = None
                                
                                # Calculate derived variables
                                if 'wind_u' in record and 'wind_v' in record:
                                    if record['wind_u'] is not None and record['wind_v'] is not None:
                                        record['wind_speed'] = np.sqrt(record['wind_u']**2 + record['wind_v']**2)
                                        record['wind_direction'] = np.degrees(np.arctan2(record['wind_v'], record['wind_u']))
                                
                                data_records.append(record)
                        
                        if data_records:
                            df_time = pd.DataFrame(data_records)
                            processed_data.append(df_time)
            
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                continue
        
        if processed_data:
            # Combine all data
            combined_df = pd.concat(processed_data, ignore_index=True)
            
            # Clean data
            combined_df = self.clean_weather_data(combined_df)
            
            # Save to CSV  
            output_path = os.path.join(self.output_dir, 'merra2_processed.csv')
            combined_df.to_csv(output_path, index=False)
            
            # Save to database
            self.save_to_database(combined_df, 'raw_weather_data')
            
            logger.info(f"✅ Processed MERRA-2 data: {len(combined_df)} records")
            logger.info(f"💾 Saved to: {output_path}")
            
            return combined_df
        else:
            logger.warning("⚠️  No MERRA-2 data processed")
            return pd.DataFrame()
    
    def clean_weather_data(self, df):
        """Clean and validate weather data"""
        logger.info("🧹 Cleaning weather data...")
        
        initial_count = len(df)
        
        # Remove invalid coordinates
        df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
        df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
        
        # Apply quality checks from config
        for param, (min_val, max_val) in QUALITY_CHECKS.items():
            if param.replace('_range', '') in df.columns:
                col = param.replace('_range', '')
                df = df[(df[col].isna()) | ((df[col] >= min_val) & (df[col] <= max_val))]
        
        # Remove rows with too much missing data
        missing_threshold = QUALITY_CHECKS['missing_data_threshold']
        required_cols = ['temperature', 'precipitation', 'humidity', 'pressure']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if available_cols:
            missing_ratio = df[available_cols].isna().sum(axis=1) / len(available_cols)
            df = df[missing_ratio <= missing_threshold]
        
        logger.info(f"🧹 Cleaned data: {initial_count} → {len(df)} records")
        return df
    
    def create_training_features(self, df):
        """Create ML training features from weather data"""
        logger.info("🔧 Creating training features...")
        
        features_df = df.copy()
        
        # Normalize coordinates
        features_df['lat_normalized'] = (features_df['latitude'] + 90) / 180
        features_df['lon_normalized'] = (features_df['longitude'] + 180) / 360
        
        # Temporal features
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['month'] = features_df['date'].dt.month
        features_df['day_of_year'] = features_df['date'].dt.dayofyear
        features_df['season'] = features_df['month'].apply(self.get_season)
        
        # Weather feature engineering
        if 'temperature' in features_df.columns:
            # Temperature features
            features_df['temperature_normalized'] = (features_df['temperature'] + 50) / 100
            features_df['temperature_extreme'] = (
                (features_df['temperature'] < RISK_THRESHOLDS['temperature']['very_cold']) |
                (features_df['temperature'] > RISK_THRESHOLDS['temperature']['very_hot'])
            ).astype(int)
        
        if 'precipitation' in features_df.columns:
            # Precipitation features  
            features_df['precipitation_log'] = np.log1p(features_df['precipitation'])
            features_df['precipitation_category'] = pd.cut(
                features_df['precipitation'],
                bins=[0, 1, 10, 25, 50, float('inf')],
                labels=['none', 'light', 'moderate', 'heavy', 'extreme']
            )
        
        if 'humidity' in features_df.columns:
            # Humidity comfort index
            features_df['humidity_comfort'] = features_df['humidity'].apply(
                lambda x: 'optimal' if 40 <= x <= 70 else 'uncomfortable'
            )
        
        # Derived weather features
        if all(col in features_df.columns for col in ['temperature', 'humidity']):
            # Heat index approximation
            features_df['heat_index'] = self.calculate_heat_index(
                features_df['temperature'], features_df['humidity']
            )
        
        if 'precipitation' in features_df.columns:
            # Cloud cover estimation from precipitation
            features_df['cloud_cover'] = self.estimate_cloud_cover(features_df['precipitation'])
            
            # Visibility estimation
            features_df['visibility'] = self.estimate_visibility(
                features_df['precipitation'], features_df.get('humidity', 60)
            )
        
        logger.info(f"🔧 Created features: {len(features_df.columns)} columns")
        return features_df
    
    def calculate_weather_risk_scores(self, df):
        """Calculate weather risk scores for different event types"""
        logger.info("📊 Calculating weather risk scores...")
        
        risk_df = df.copy()
        
        # Calculate component risk scores
        if 'precipitation' in risk_df.columns:
            risk_df['precipitation_risk'] = risk_df['precipitation'].apply(
                lambda x: self.calculate_precipitation_risk(x)
            )
        
        if 'temperature' in risk_df.columns:
            risk_df['temperature_risk'] = risk_df['temperature'].apply(
                lambda x: self.calculate_temperature_risk(x)
            )
        
        if 'wind_speed' in risk_df.columns:
            risk_df['wind_risk'] = risk_df['wind_speed'].apply(
                lambda x: self.calculate_wind_risk(x)
            )
        
        if 'humidity' in risk_df.columns:
            risk_df['humidity_risk'] = risk_df['humidity'].apply(
                lambda x: self.calculate_humidity_risk(x)
            )
        
        # Overall weather risk (weighted average)
        risk_components = []
        weights = []
        
        if 'precipitation_risk' in risk_df.columns:
            risk_components.append(risk_df['precipitation_risk'])
            weights.append(0.4)  # Precipitation has highest weight
        
        if 'temperature_risk' in risk_df.columns:
            risk_components.append(risk_df['temperature_risk'])
            weights.append(0.3)
            
        if 'wind_risk' in risk_df.columns:
            risk_components.append(risk_df['wind_risk'])
            weights.append(0.2)
            
        if 'humidity_risk' in risk_df.columns:
            risk_components.append(risk_df['humidity_risk'])
            weights.append(0.1)
        
        if risk_components:
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            risk_df['overall_risk'] = sum(w * comp for w, comp in zip(weights, risk_components))
        
        # Event-specific suitability scores
        for event_type in EVENT_TYPES:
            risk_df[f'{event_type.lower()}_suitability'] = risk_df.apply(
                lambda row: self.calculate_event_suitability(row, event_type), axis=1
            )
        
        return risk_df
    
    def calculate_precipitation_risk(self, precip):
        """Calculate precipitation risk score (0-1)"""
        thresholds = RISK_THRESHOLDS['precipitation']
        if precip <= thresholds['low']:
            return 0.1
        elif precip <= thresholds['medium']:
            return 0.3 + 0.3 * (precip - thresholds['low']) / (thresholds['medium'] - thresholds['low'])
        elif precip <= thresholds['high']:
            return 0.6 + 0.25 * (precip - thresholds['medium']) / (thresholds['high'] - thresholds['medium'])
        else:
            return min(0.95, 0.85 + 0.1 * (precip - thresholds['high']) / thresholds['extreme'])
    
    def calculate_temperature_risk(self, temp):
        """Calculate temperature risk score (0-1)"""
        thresholds = RISK_THRESHOLDS['temperature']
        if temp <= thresholds['very_cold']:
            return 0.9
        elif temp <= thresholds['cold']:
            return 0.4 - 0.3 * (temp - thresholds['very_cold']) / (thresholds['cold'] - thresholds['very_cold'])
        elif temp <= thresholds['optimal']:
            return 0.1
        elif temp <= thresholds['hot']:
            return 0.1 + 0.4 * (temp - thresholds['optimal']) / (thresholds['hot'] - thresholds['optimal'])
        else:
            return min(0.95, 0.5 + 0.45 * (temp - thresholds['hot']) / (thresholds['very_hot'] - thresholds['hot']))
    
    def calculate_wind_risk(self, wind_speed):
        """Calculate wind risk score (0-1)"""
        thresholds = RISK_THRESHOLDS['wind_speed']
        if wind_speed <= thresholds['calm']:
            return 0.05
        elif wind_speed <= thresholds['moderate']:
            return 0.05 + 0.15 * (wind_speed - thresholds['calm']) / (thresholds['moderate'] - thresholds['calm'])
        elif wind_speed <= thresholds['strong']:
            return 0.2 + 0.4 * (wind_speed - thresholds['moderate']) / (thresholds['strong'] - thresholds['moderate'])
        else:
            return min(0.95, 0.6 + 0.35 * (wind_speed - thresholds['strong']) / (thresholds['extreme'] - thresholds['strong']))
    
    def calculate_humidity_risk(self, humidity):
        """Calculate humidity risk score (0-1)"""
        thresholds = RISK_THRESHOLDS['humidity']
        if humidity <= thresholds['very_dry']:
            return 0.3
        elif humidity <= thresholds['optimal']:
            return 0.05
        elif humidity <= thresholds['humid']:
            return 0.05 + 0.15 * (humidity - thresholds['optimal']) / (thresholds['humid'] - thresholds['optimal'])
        else:
            return 0.2 + 0.3 * (humidity - thresholds['humid']) / (thresholds['very_humid'] - thresholds['humid'])
    
    def calculate_event_suitability(self, row, event_type):
        """Calculate event suitability score for specific event type"""
        overall_risk = row.get('overall_risk', 0.5)
        
        # Event-specific adjustments
        if event_type in ['Wedding', 'Art Exhibition', 'Corporate Event']:
            # More sensitive to weather conditions
            return max(0.05, 1.0 - overall_risk * 1.2)
        elif event_type in ['Sports Event', 'Marathon']:
            # Temperature more critical
            temp_risk = row.get('temperature_risk', 0.3)
            return max(0.05, 1.0 - (overall_risk * 0.8 + temp_risk * 0.4))
        else:
            # Standard calculation
            return max(0.05, 1.0 - overall_risk)
    
    def get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def calculate_heat_index(self, temp, humidity):
        """Calculate heat index (simplified)"""
        if isinstance(temp, pd.Series):
            return temp + 0.3 * humidity - 2
        else:
            return temp + 0.3 * humidity - 2
    
    def estimate_cloud_cover(self, precipitation):
        """Estimate cloud cover from precipitation"""
        if isinstance(precipitation, pd.Series):
            return precipitation.apply(lambda x: min(95, 20 + x * 3))
        else:
            return min(95, 20 + precipitation * 3)
    
    def estimate_visibility(self, precipitation, humidity):
        """Estimate visibility from precipitation and humidity"""
        if isinstance(precipitation, pd.Series):
            vis = 20 - precipitation * 0.5 - (humidity - 50) * 0.1
            return vis.clip(1, 20)
        else:
            vis = 20 - precipitation * 0.5 - (humidity - 50) * 0.1
            return max(1, min(20, vis))
    
    def save_to_database(self, df, table_name):
        """Save processed data to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            
            logger.info(f"💾 Saved {len(df)} records to {table_name}")
            
        except Exception as e:
            logger.error(f"❌ Database save failed: {e}")
    
    def create_training_datasets(self):
        """Create final training datasets for ML models"""
        logger.info("📋 Creating training datasets...")
        
        try:
            # Load processed data from database
            conn = sqlite3.connect(self.db_path)
            
            # Query all processed weather data
            query = """
                SELECT * FROM raw_weather_data 
                WHERE data_quality_score IS NULL OR data_quality_score > 0.7
                ORDER BY date, latitude, longitude
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) == 0:
                logger.warning("⚠️  No processed data found in database")
                return
            
            logger.info(f"📊 Loaded {len(df)} weather records")
            
            # Create features
            features_df = self.create_training_features(df)
            
            # Calculate risk scores
            risk_df = self.calculate_weather_risk_scores(features_df)
            
            # Create training data for different models
            self.create_weather_risk_training_data(risk_df)
            self.create_event_suitability_training_data(risk_df)
            
            logger.info("✅ Training datasets created successfully")
            
        except Exception as e:
            logger.error(f"❌ Training dataset creation failed: {e}")
    
    def create_weather_risk_training_data(self, df):
        """Create training data for weather risk prediction model"""
        logger.info("🎯 Creating weather risk training data...")
        
        # Select features for weather risk model
        feature_cols = ML_CONFIG['WEATHER_RISK_MODEL']['features']
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            logger.warning("⚠️  No features available for weather risk model")
            return
        
        # Prepare training data
        X = df[available_features].copy()
        y = df['overall_risk'].copy()
        
        # Remove rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Add metadata
        X['date'] = df['date'][valid_mask]
        X['latitude'] = df['latitude'][valid_mask]  
        X['longitude'] = df['longitude'][valid_mask]
        X['weather_risk_score'] = y
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'training', 'weather_risk_training.csv')
        X.to_csv(output_path, index=False)
        
        logger.info(f"✅ Weather risk training data: {len(X)} samples")
        logger.info(f"💾 Saved to: {output_path}")
    
    def create_event_suitability_training_data(self, df):
        """Create training data for event suitability models"""
        logger.info("🎪 Creating event suitability training data...")
        
        feature_cols = ML_CONFIG['EVENT_SUITABILITY_MODEL']['features']
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            logger.warning("⚠️  No features available for event suitability model")
            return
        
        # Create training data for each event type
        for event_type in EVENT_TYPES:
            suitability_col = f'{event_type.lower()}_suitability'
            
            if suitability_col in df.columns:
                X = df[available_features].copy()
                y = df[suitability_col].copy()
                
                # Remove missing values
                valid_mask = ~y.isna() & ~X.isna().any(axis=1)
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) > 0:
                    # Add metadata
                    X['date'] = df['date'][valid_mask]
                    X['latitude'] = df['latitude'][valid_mask]
                    X['longitude'] = df['longitude'][valid_mask]
                    X['event_type'] = event_type
                    X['suitability_score'] = y
                    
                    # Save to CSV
                    output_path = os.path.join(
                        self.output_dir, 'training', 
                        f'event_suitability_{event_type.lower()}_training.csv'
                    )
                    X.to_csv(output_path, index=False)
                    
                    logger.info(f"✅ {event_type} suitability: {len(X)} samples")

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Process NASA weather data for ML training')
    parser.add_argument('--input-dir', default=RAW_DATA_DIR,
                       help='Input directory with raw NASA data')
    parser.add_argument('--output-dir', default=PROCESSED_DATA_DIR,
                       help='Output directory for processed data')
    parser.add_argument('--datasets', nargs='+', default=['IMERG', 'MERRA2'],
                       choices=['IMERG', 'MERRA2'],
                       help='Datasets to process')
    parser.add_argument('--create-training', action='store_true',
                       help='Create final training datasets')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting NASA Weather Data Preprocessing Pipeline")
    
    # Initialize processor
    processor = NASADataProcessor(args.input_dir, args.output_dir)
    
    try:
        # Process datasets
        if 'IMERG' in args.datasets:
            imerg_data = processor.process_imerg_files()
        
        if 'MERRA2' in args.datasets:
            merra2_data = processor.process_merra2_files()
        
        # Create training datasets
        if args.create_training:
            processor.create_training_datasets()
        
        logger.info("🎉 Preprocessing pipeline complete!")
        
    except KeyboardInterrupt:
        logger.info("⛔ Preprocessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Preprocessing pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()