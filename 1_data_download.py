#!/usr/bin/env python3
"""
NASA Weather Data Download Script with Earthdata Token
Complete data acquisition for WeatherWise Flutter App training

This script downloads:
1. NASA GPM IMERG precipitation data (token required)
2. NASA MERRA-2 meteorological data (token required)  
3. NASA MODIS atmospheric data (token required)
4. Historical weather datasets for ML training

Usage:
    python 1_data_download.py --region GLOBAL --period HISTORICAL --parameters all
"""

import os
import sys
import requests
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
from pathlib import Path

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

class NASADataDownloader:
    def __init__(self, token=None):
        """Initialize NASA Data Downloader with Earthdata token"""
        self.token = token or NASA_TOKEN
        self.session = requests.Session()
        
        if self.token == "YOUR_NASA_EARTHDATA_TOKEN_HERE":
            raise ValueError("⚠️  Please update your NASA Earthdata token in config.py!")
        
        # Set authentication headers
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'User-Agent': 'WeatherWise-DataDownloader/1.0',
            'Accept': 'application/json'
        }
        self.session.headers.update(self.headers)
        
        logger.info(f"🛰️  NASA Data Downloader initialized with token: {self.token[:10]}...")
        
    def test_authentication(self):
        """Test NASA Earthdata authentication"""
        try:
            test_url = "https://urs.earthdata.nasa.gov/api/users/current"
            response = self.session.get(test_url, timeout=30)
            
            if response.status_code == 200:
                user_info = response.json()
                logger.info(f"✅ Authentication successful! User: {user_info.get('uid', 'Unknown')}")
                return True
            else:
                logger.error(f"❌ Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {e}")
            return False
    
    def download_imerg_data(self, start_date, end_date, region='GLOBAL', product_type='FINAL'):
        """
        Download NASA GPM IMERG precipitation data
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format  
            region (str): Geographic region from TRAINING_REGIONS
            product_type (str): FINAL, LATE, or EARLY
        """
        logger.info(f"🌧️  Downloading IMERG {product_type} data: {start_date} to {end_date}")
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get region bounds
        if region in TRAINING_REGIONS:
            bounds = TRAINING_REGIONS[region]
        else:
            logger.warning(f"Unknown region {region}, using GLOBAL")
            bounds = TRAINING_REGIONS['GLOBAL']
        
        downloaded_files = []
        current_date = start_dt
        
        while current_date <= end_dt:
            year = current_date.year
            month = current_date.month
            day = current_date.day
            
            # Construct IMERG file URL
            date_str = current_date.strftime('%Y%m%d')
            
            if product_type == 'FINAL':
                base_url = NASA_ENDPOINTS['IMERG_DOWNLOAD']
                filename = f"3B-DAY.MS.MRG.3IMERG.{date_str}-S000000-E235959.V07.nc4"
            else:
                logger.warning(f"Product type {product_type} not fully implemented")
                current_date += timedelta(days=1)
                continue
                
            # Full download URL
            file_url = f"{base_url}/{year:04d}/{month:02d}/{filename}"
            
            # Local file path
            local_dir = os.path.join(RAW_DATA_DIR, 'imerg', str(year), f"{month:02d}")
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            
            # Skip if file exists
            if os.path.exists(local_path):
                logger.info(f"⏭️  File exists, skipping: {filename}")
                downloaded_files.append(local_path)
                current_date += timedelta(days=1)
                continue
            
            # Download file
            try:
                logger.info(f"⬇️  Downloading: {filename}")
                response = self.session.get(file_url, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file size
                file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
                logger.info(f"✅ Downloaded: {filename} ({file_size:.1f} MB)")
                downloaded_files.append(local_path)
                
                # Rate limiting
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Failed to download {filename}: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)  # Remove partial file
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error downloading {filename}: {e}")
                
            current_date += timedelta(days=1)
        
        logger.info(f"🏁 IMERG download complete: {len(downloaded_files)} files")
        return downloaded_files
    
    def download_merra2_data(self, start_date, end_date, region='GLOBAL', collection='M2T1NXSLV'):
        """
        Download NASA MERRA-2 meteorological data
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            region (str): Geographic region 
            collection (str): MERRA-2 collection ID
        """
        logger.info(f"🌡️  Downloading MERRA-2 {collection} data: {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get region bounds
        bounds = TRAINING_REGIONS.get(region, TRAINING_REGIONS['GLOBAL'])
        
        downloaded_files = []
        current_date = start_dt
        
        while current_date <= end_dt:
            year = current_date.year
            month = current_date.month
            day = current_date.day
            
            # MERRA-2 filename format
            date_str = current_date.strftime('%Y%m%d')
            filename = f"MERRA2_400.{collection}.5.12.4.{date_str}.nc4"
            
            # Construct download URL
            base_url = NASA_ENDPOINTS['MERRA2_DOWNLOAD']
            file_url = f"{base_url}/{year:04d}/{month:02d}/{filename}"
            
            # Local path
            local_dir = os.path.join(RAW_DATA_DIR, 'merra2', collection, str(year), f"{month:02d}")
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            
            # Skip if exists
            if os.path.exists(local_path):
                logger.info(f"⏭️  File exists: {filename}")
                downloaded_files.append(local_path)
                current_date += timedelta(days=1)
                continue
            
            # Download
            try:
                logger.info(f"⬇️  Downloading: {filename}")
                response = self.session.get(file_url, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = os.path.getsize(local_path) / (1024 * 1024)
                logger.info(f"✅ Downloaded: {filename} ({file_size:.1f} MB)")
                downloaded_files.append(local_path)
                
                time.sleep(1)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Failed to download {filename}: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                    
            current_date += timedelta(days=1)
        
        logger.info(f"🏁 MERRA-2 download complete: {len(downloaded_files)} files")
        return downloaded_files
    
    def search_earthdata_granules(self, concept_id, start_date, end_date, bbox=None):
        """
        Search for data granules using NASA Earthdata Search API
        
        Args:
            concept_id (str): NASA concept ID for dataset
            start_date (str): Start date
            end_date (str): End date  
            bbox (tuple): Bounding box (west, south, east, north)
        """
        logger.info(f"🔍 Searching Earthdata granules: {concept_id}")
        
        params = {
            'concept_id': concept_id,
            'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
            'page_size': 2000
        }
        
        if bbox:
            params['bounding_box'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        try:
            response = self.session.get(
                NASA_ENDPOINTS['EARTHDATA_SEARCH'],
                params=params,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            granules = data.get('feed', {}).get('entry', [])
            
            logger.info(f"📊 Found {len(granules)} granules")
            return granules
            
        except Exception as e:
            logger.error(f"❌ Earthdata search failed: {e}")
            return []
    
    def download_granule_files(self, granules, max_workers=5):
        """
        Download multiple granule files concurrently
        
        Args:
            granules (list): List of granule metadata
            max_workers (int): Maximum concurrent downloads
        """
        logger.info(f"📦 Starting batch download of {len(granules)} granules")
        
        download_tasks = []
        for granule in granules:
            # Extract download URLs from granule metadata
            if 'links' in granule:
                for link in granule['links']:
                    if 'href' in link and link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                        download_tasks.append({
                            'url': link['href'],
                            'granule_id': granule.get('id', 'unknown'),
                            'title': granule.get('title', 'unknown')
                        })
        
        downloaded_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._download_single_granule, task): task 
                for task in download_tasks[:50]  # Limit to first 50 for testing
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        downloaded_files.append(result)
                except Exception as e:
                    logger.error(f"❌ Failed to download {task['title']}: {e}")
        
        return downloaded_files
    
    def _download_single_granule(self, task):
        """Download a single granule file"""
        try:
            url = task['url']
            granule_id = task['granule_id']
            
            # Determine local path
            filename = url.split('/')[-1]
            local_dir = os.path.join(RAW_DATA_DIR, 'earthdata_granules')
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            
            # Skip if exists
            if os.path.exists(local_path):
                return local_path
            
            # Download
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Downloaded granule: {filename}")
            return local_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download granule {granule_id}: {e}")
            return None
    
    def create_download_inventory(self, output_path=None):
        """Create inventory of all downloaded data"""
        if not output_path:
            output_path = os.path.join(OUTPUT_DIR, 'download_inventory.csv')
        
        inventory = []
        
        # Scan all downloaded files
        for root, dirs, files in os.walk(RAW_DATA_DIR):
            for file in files:
                if file.endswith(('.nc4', '.nc', '.hdf', '.h5')):
                    filepath = os.path.join(root, file)
                    file_size = os.path.getsize(filepath)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    # Determine dataset type
                    if 'imerg' in root.lower():
                        dataset_type = 'IMERG'
                    elif 'merra2' in root.lower():
                        dataset_type = 'MERRA-2'
                    else:
                        dataset_type = 'Other'
                    
                    inventory.append({
                        'filename': file,
                        'filepath': filepath,
                        'dataset_type': dataset_type,
                        'file_size_mb': file_size / (1024 * 1024),
                        'download_date': mod_time.strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # Save inventory
        df = pd.DataFrame(inventory)
        df.to_csv(output_path, index=False)
        
        logger.info(f"📋 Created download inventory: {output_path}")
        logger.info(f"📊 Total files: {len(inventory)}")
        logger.info(f"💾 Total size: {df['file_size_mb'].sum():.1f} MB")
        
        return inventory

def main():
    """Main download function"""
    parser = argparse.ArgumentParser(description='Download NASA weather data with Earthdata token')
    parser.add_argument('--region', default='GLOBAL', choices=list(TRAINING_REGIONS.keys()),
                       help='Geographic region for data download')
    parser.add_argument('--period', default='RECENT', choices=list(DATA_PERIODS.keys()),
                       help='Time period for data download')
    parser.add_argument('--datasets', nargs='+', default=['IMERG', 'MERRA2'],
                       choices=['IMERG', 'MERRA2', 'EARTHDATA'],
                       help='Datasets to download')
    parser.add_argument('--test-auth', action='store_true',
                       help='Test authentication only')
    parser.add_argument('--max-days', type=int, default=30,
                       help='Maximum number of days to download (for testing)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting NASA Weather Data Download Pipeline")
    logger.info(f"📍 Region: {args.region}")
    logger.info(f"📅 Period: {args.period}")
    logger.info(f"📊 Datasets: {args.datasets}")
    
    # Initialize downloader
    downloader = NASADataDownloader()
    
    # Test authentication
    if not downloader.test_authentication():
        logger.error("❌ Authentication failed. Please check your NASA Earthdata token.")
        sys.exit(1)
    
    if args.test_auth:
        logger.info("✅ Authentication test successful!")
        return
    
    # Get date range
    period = DATA_PERIODS[args.period]
    start_date = period['start_date']
    end_date = period['end_date']
    
    # Limit days for testing
    if args.max_days:
        end_dt = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=args.max_days)
        end_date = end_dt.strftime('%Y-%m-%d')
        logger.info(f"⚠️  Limited to {args.max_days} days for testing")
    
    total_files = 0
    
    try:
        # Download IMERG data
        if 'IMERG' in args.datasets:
            imerg_files = downloader.download_imerg_data(start_date, end_date, args.region)
            total_files += len(imerg_files)
        
        # Download MERRA-2 data  
        if 'MERRA2' in args.datasets:
            merra2_files = downloader.download_merra2_data(start_date, end_date, args.region)
            total_files += len(merra2_files)
        
        # Create inventory
        inventory = downloader.create_download_inventory()
        
        logger.info(f"🎉 Download pipeline complete!")
        logger.info(f"📊 Total files downloaded: {total_files}")
        logger.info(f"💾 Data ready for preprocessing and training")
        
    except KeyboardInterrupt:
        logger.info("⛔ Download interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Download pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()