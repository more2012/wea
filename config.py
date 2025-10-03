# NASA Weather Data Analysis & Training Pipeline - Configuration
# Complete setup for WeatherWise Flutter App

import os
from datetime import datetime, timedelta

# NASA Earthdata Authentication
NASA_TOKEN = "YOUR_NASA_EARTHDATA_TOKEN_HERE"  # Replace with your actual token
NASA_USERNAME = "your_nasa_username"
NASA_PASSWORD = "your_nasa_password"

# NASA API Endpoints (Token Required)
NASA_ENDPOINTS = {
    # GES DISC - Requires Token
    'IMERG_FINAL': 'https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary',
    'IMERG_HALFHOURLY': 'https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary',
    'MERRA2_HOURLY': 'https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary',
    'MERRA2_DAILY': 'https://disc.gsfc.nasa.gov/datasets/M2SDNXSLV_5.12.4/summary',
    
    # Direct Download URLs (Token Required)
    'IMERG_DOWNLOAD': 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07',
    'MERRA2_DOWNLOAD': 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4',
    
    # API Endpoints
    'EARTHDATA_SEARCH': 'https://search.earthdata.nasa.gov/search/granules',
    'GIOVANNI_API': 'https://giovanni.gsfc.nasa.gov/giovanni/daac-bin/service_manager.pl'
}

# Data Parameters for WeatherWise App
WEATHER_PARAMETERS = {
    'IMERG': {
        'precipitation': 'precipitationCal',  # Calibrated precipitation
        'precipitation_error': 'randomError',
        'liquid_probability': 'probabilityLiquidPrecipitation'
    },
    'MERRA2': {
        'temperature': 'T2M',          # 2-meter air temperature
        'humidity': 'QV2M',            # 2-meter specific humidity  
        'pressure': 'PS',              # Surface pressure
        'wind_u': 'U2M',              # 2-meter eastward wind
        'wind_v': 'V2M',              # 2-meter northward wind
        'precipitation': 'PRECTOTCORR', # Bias corrected precipitation
        'solar_radiation': 'SWGDN'     # Surface downward shortwave radiation
    }
}

# Geographic Coverage for Training Data
TRAINING_REGIONS = {
    'GLOBAL': {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'NORTH_AMERICA': {'lat_min': 25, 'lat_max': 70, 'lon_min': -170, 'lon_max': -50},
    'EUROPE': {'lat_min': 35, 'lat_max': 70, 'lon_min': -15, 'lon_max': 50},
    'AFRICA': {'lat_min': -35, 'lat_max': 40, 'lon_min': -20, 'lon_max': 55},
    'ASIA': {'lat_min': -10, 'lat_max': 60, 'lon_min': 60, 'lon_max': 150},
    'MIDDLE_EAST': {'lat_min': 15, 'lat_max': 45, 'lon_min': 25, 'lon_max': 65}
}

# Time Periods for Training Data
DATA_PERIODS = {
    'RECENT': {
        'start_date': '2020-01-01',
        'end_date': '2024-12-31',
        'description': 'Recent 5 years for current patterns'
    },
    'HISTORICAL': {
        'start_date': '2000-01-01', 
        'end_date': '2019-12-31',
        'description': '20 years of historical data for training'
    },
    'VALIDATION': {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31', 
        'description': 'Recent year for model validation'
    }
}

# Machine Learning Configuration
ML_CONFIG = {
    'WEATHER_RISK_MODEL': {
        'algorithm': 'RandomForestRegressor',
        'features': ['temperature', 'precipitation', 'humidity', 'wind_speed', 'pressure'],
        'target': 'weather_risk_score',
        'test_size': 0.2,
        'random_state': 42
    },
    'EVENT_SUITABILITY_MODEL': {
        'algorithm': 'MLPRegressor',
        'features': ['temperature', 'precipitation', 'humidity', 'wind_speed', 'cloud_cover'],
        'target': 'event_suitability_score',
        'hidden_layer_sizes': (100, 50),
        'max_iter': 1000
    },
    'PRECIPITATION_FORECAST': {
        'algorithm': 'LSTM',
        'lookback_days': 7,
        'forecast_days': 3,
        'features': ['precipitation', 'humidity', 'temperature', 'pressure']
    }
}

# Event Types for Risk Assessment (from your WeatherWise app)
EVENT_TYPES = [
    'Wedding', 'Concert', 'Festival', 'Sports Event', 'Conference', 
    'Parade', 'Picnic', 'Corporate Event', 'Art Exhibition', 
    'Food Festival', 'Marathon', 'Other'
]

# Weather Risk Thresholds
RISK_THRESHOLDS = {
    'precipitation': {
        'low': 1.0,      # mm/day
        'medium': 10.0,  # mm/day  
        'high': 25.0,    # mm/day
        'extreme': 50.0  # mm/day
    },
    'temperature': {
        'very_cold': -5,   # °C
        'cold': 10,        # °C
        'optimal': 25,     # °C
        'hot': 35,         # °C
        'very_hot': 40     # °C
    },
    'wind_speed': {
        'calm': 5,         # km/h
        'moderate': 20,    # km/h
        'strong': 40,      # km/h
        'extreme': 60      # km/h
    },
    'humidity': {
        'very_dry': 30,    # %
        'dry': 45,         # %
        'optimal': 65,     # %
        'humid': 85,       # %
        'very_humid': 95   # %
    }
}

# File Paths
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = 'models'
OUTPUT_DIR = 'output'
LOGS_DIR = 'logs'

# Create directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Database Configuration (for processed data storage)
DATABASE_CONFIG = {
    'sqlite': {
        'path': os.path.join(DATA_DIR, 'weather_data.db'),
        'tables': {
            'raw_weather': 'raw_weather_data',
            'processed_weather': 'processed_weather_data',
            'predictions': 'weather_predictions',
            'training_data': 'ml_training_data'
        }
    }
}

# API Configuration for Backend
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 3000,
    'debug': True,
    'rate_limit': '100/hour',
    'cache_timeout': 300,  # 5 minutes
    'max_concurrent_requests': 10
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(LOGS_DIR, f'weather_pipeline_{datetime.now().strftime("%Y%m%d")}.log')
}

# Flutter Integration
FLUTTER_CONFIG = {
    'backend_url': 'http://localhost:3000',
    'api_endpoints': {
        'weather_analysis': '/api/weather-analysis',
        'risk_prediction': '/api/risk-prediction', 
        'event_suitability': '/api/event-suitability',
        'historical_data': '/api/historical-data',
        'bulk_download': '/api/bulk-download'
    }
}

# Data Quality Checks
QUALITY_CHECKS = {
    'temperature_range': (-100, 60),    # °C
    'precipitation_range': (0, 500),    # mm/day
    'humidity_range': (0, 100),         # %
    'pressure_range': (800, 1100),     # hPa
    'wind_speed_range': (0, 200),      # km/h
    'missing_data_threshold': 0.1       # 10% max missing data
}

# Model Performance Metrics
PERFORMANCE_METRICS = [
    'mean_absolute_error',
    'mean_squared_error', 
    'r2_score',
    'accuracy_score',
    'precision_score',
    'recall_score'
]

print("✅ Configuration loaded successfully!")
print(f"📍 Data directory: {DATA_DIR}")
print(f"🤖 Models directory: {MODELS_DIR}")
print(f"📊 Output directory: {OUTPUT_DIR}")
print(f"🔐 NASA Token configured: {'Yes' if NASA_TOKEN != 'YOUR_NASA_EARTHDATA_TOKEN_HERE' else 'No - Please update!'}")