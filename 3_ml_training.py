#!/usr/bin/env python3
"""
Machine Learning Training Pipeline for WeatherWise App
Train weather risk prediction and event suitability models using processed NASA data

This script trains:
1. Weather Risk Prediction Model (Random Forest)
2. Event Suitability Scoring Model (Neural Network)  
3. Precipitation Forecasting Model (LSTM)
4. Temperature Trend Analysis Model (Linear Regression)

Usage:
    python 3_ml_training.py --model weather-risk --data-dir data/processed/training
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. LSTM models will be skipped.")

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

class WeatherMLTrainer:
    def __init__(self, data_dir=None, models_dir=None):
        """Initialize ML trainer for weather models"""
        self.data_dir = data_dir or os.path.join(PROCESSED_DATA_DIR, 'training')
        self.models_dir = models_dir or MODELS_DIR
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, 'weather_risk'), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, 'event_suitability'), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, 'forecasting'), exist_ok=True)
        
        logger.info(f"📂 Data directory: {self.data_dir}")
        logger.info(f"🤖 Models directory: {self.models_dir}")
        
        # Initialize scalers
        self.scalers = {}
        
    def load_training_data(self, dataset_type='weather-risk'):
        """Load training data for specific model type"""
        logger.info(f"📊 Loading {dataset_type} training data...")
        
        if dataset_type == 'weather-risk':
            file_path = os.path.join(self.data_dir, 'weather_risk_training.csv')
        elif dataset_type.startswith('event-suitability'):
            event_type = dataset_type.split('-')[-1] if len(dataset_type.split('-')) > 2 else 'wedding'
            file_path = os.path.join(self.data_dir, f'event_suitability_{event_type}_training.csv')
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded training data: {len(df)} samples, {len(df.columns)} features")
        
        return df
    
    def prepare_features(self, df, model_type='weather-risk'):
        """Prepare features and target variables for training"""
        logger.info(f"🔧 Preparing features for {model_type} model...")
        
        # Select feature columns based on model type
        if model_type == 'weather-risk':
            feature_cols = ML_CONFIG['WEATHER_RISK_MODEL']['features']
            target_col = 'weather_risk_score'
        elif model_type.startswith('event-suitability'):
            feature_cols = ML_CONFIG['EVENT_SUITABILITY_MODEL']['features']
            target_col = 'suitability_score'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            raise ValueError(f"No features available for {model_type} model")
        
        # Prepare feature matrix
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Prepare target variable
        if target_col in df.columns:
            y = df[target_col].copy()
            y = y.fillna(y.median())  # Fill missing targets with median
        else:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Remove samples where target is still NaN
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"🔧 Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"📊 Target range: {y.min():.3f} to {y.max():.3f}")
        
        return X, y
    
    def train_weather_risk_model(self, hyperparameter_tuning=False):
        """Train weather risk prediction model"""
        logger.info("🌦️  Training weather risk prediction model...")
        
        # Load and prepare data
        df = self.load_training_data('weather-risk')
        X, y = self.prepare_features(df, 'weather-risk')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=ML_CONFIG['WEATHER_RISK_MODEL']['test_size'],
            random_state=ML_CONFIG['WEATHER_RISK_MODEL']['random_state']
        )
        
        # Create preprocessing pipeline
        scaler = StandardScaler()
        
        if hyperparameter_tuning:
            # Grid search for best parameters
            logger.info("🔍 Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
            
            # Fit with scaled features
            X_train_scaled = scaler.fit_transform(X_train)
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"✅ Best parameters: {grid_search.best_params_}")
            
        else:
            # Use default parameters
            best_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit model
            X_train_scaled = scaler.fit_transform(X_train)
            best_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        X_test_scaled = scaler.transform(X_test)
        y_pred = best_model.predict(X_test_scaled)
        
        # Evaluate model
        metrics = self.evaluate_model(y_test, y_pred, 'Weather Risk Prediction')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("🔝 Top 5 important features:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save model and scaler
        model_path = os.path.join(self.models_dir, 'weather_risk', 'weather_risk_model.joblib')
        scaler_path = os.path.join(self.models_dir, 'weather_risk', 'weather_risk_scaler.joblib')
        metrics_path = os.path.join(self.models_dir, 'weather_risk', 'weather_risk_metrics.json')
        features_path = os.path.join(self.models_dir, 'weather_risk', 'feature_importance.csv')
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        feature_importance.to_csv(features_path, index=False)
        
        self.scalers['weather_risk'] = scaler
        
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📊 Metrics saved: {metrics_path}")
        
        return best_model, scaler, metrics
    
    def train_event_suitability_model(self, event_type='wedding', hyperparameter_tuning=False):
        """Train event suitability prediction model"""
        logger.info(f"🎪 Training {event_type} suitability model...")
        
        # Load and prepare data
        df = self.load_training_data(f'event-suitability-{event_type}')
        X, y = self.prepare_features(df, 'event-suitability')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create preprocessing pipeline
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if hyperparameter_tuning:
            # Grid search for neural network
            logger.info("🔍 Tuning neural network hyperparameters...")
            
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
            
            mlp = MLPRegressor(max_iter=1000, random_state=42)
            grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"✅ Best parameters: {grid_search.best_params_}")
            
        else:
            # Use default neural network configuration
            best_model = MLPRegressor(
                hidden_layer_sizes=ML_CONFIG['EVENT_SUITABILITY_MODEL']['hidden_layer_sizes'],
                max_iter=ML_CONFIG['EVENT_SUITABILITY_MODEL']['max_iter'],
                alpha=0.001,
                learning_rate_init=0.01,
                random_state=42
            )
            
            best_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        
        # Evaluate model
        metrics = self.evaluate_model(y_test, y_pred, f'{event_type.title()} Suitability Prediction')
        
        # Save model and scaler
        model_dir = os.path.join(self.models_dir, 'event_suitability', event_type)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'{event_type}_suitability_model.joblib')
        scaler_path = os.path.join(model_dir, f'{event_type}_suitability_scaler.joblib')
        metrics_path = os.path.join(model_dir, f'{event_type}_suitability_metrics.json')
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.scalers[f'{event_type}_suitability'] = scaler
        
        logger.info(f"💾 {event_type.title()} model saved: {model_path}")
        
        return best_model, scaler, metrics
    
    def train_precipitation_forecast_model(self):
        """Train LSTM model for precipitation forecasting"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("⚠️  TensorFlow not available. Skipping LSTM training.")
            return None
        
        logger.info("🌧️  Training LSTM precipitation forecasting model...")
        
        try:
            # Load weather risk data (contains historical precipitation)
            df = self.load_training_data('weather-risk')
            
            if 'precipitation' not in df.columns:
                logger.warning("⚠️  Precipitation data not available for LSTM training")
                return None
            
            # Prepare time series data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['latitude', 'longitude', 'date'])
            
            # Group by location and create sequences
            sequences = []
            targets = []
            
            lookback_days = ML_CONFIG['PRECIPITATION_FORECAST']['lookback_days']
            forecast_days = ML_CONFIG['PRECIPITATION_FORECAST']['forecast_days']
            
            for (lat, lon), group in df.groupby(['latitude', 'longitude']):
                if len(group) >= lookback_days + forecast_days:
                    group = group.sort_values('date')
                    precip_values = group['precipitation'].values
                    
                    for i in range(len(precip_values) - lookback_days - forecast_days + 1):
                        sequence = precip_values[i:i+lookback_days]
                        target = precip_values[i+lookback_days:i+lookback_days+forecast_days]
                        
                        sequences.append(sequence)
                        targets.append(target)
            
            if len(sequences) == 0:
                logger.warning("⚠️  No valid sequences found for LSTM training")
                return None
            
            # Convert to arrays
            X = np.array(sequences)
            y = np.array(targets)
            
            # Reshape for LSTM (samples, timesteps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            logger.info(f"🔧 LSTM data prepared: {X.shape} -> {y.shape}")
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Normalize data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback_days, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(forecast_days)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            logger.info("🚂 Training LSTM model...")
            history = model.fit(
                X_train_scaled, y_train_scaled,
                batch_size=32,
                epochs=50,
                validation_split=0.1,
                verbose=0
            )
            
            # Make predictions
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            # Evaluate model
            mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
            
            metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'model_type': 'LSTM',
                'lookback_days': lookback_days,
                'forecast_days': forecast_days,
                'training_samples': len(X_train)
            }
            
            logger.info(f"📊 LSTM Metrics - MAE: {mae:.3f}, RMSE: {np.sqrt(mse):.3f}")
            
            # Save model
            model_dir = os.path.join(self.models_dir, 'forecasting')
            model_path = os.path.join(model_dir, 'precipitation_lstm_model.h5')
            scaler_X_path = os.path.join(model_dir, 'precipitation_scaler_X.joblib')
            scaler_y_path = os.path.join(model_dir, 'precipitation_scaler_y.joblib')
            metrics_path = os.path.join(model_dir, 'precipitation_lstm_metrics.json')
            
            model.save(model_path)
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"💾 LSTM model saved: {model_path}")
            
            return model, (scaler_X, scaler_y), metrics
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            return None
    
    def train_temperature_trend_model(self):
        """Train temperature trend analysis model"""
        logger.info("🌡️  Training temperature trend analysis model...")
        
        # Load data
        df = self.load_training_data('weather-risk')
        
        if 'temperature' not in df.columns:
            logger.warning("⚠️  Temperature data not available")
            return None
        
        # Prepare features for trend analysis
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Group by location and year to get annual averages
        annual_temps = df.groupby(['latitude', 'longitude', 'year']).agg({
            'temperature': 'mean'
        }).reset_index()
        
        # For each location, fit a trend line
        location_trends = []
        
        for (lat, lon), group in annual_temps.groupby(['latitude', 'longitude']):
            if len(group) >= 3:  # Need at least 3 years of data
                X = group['year'].values.reshape(-1, 1)
                y = group['temperature'].values
                
                # Fit linear regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate trend (degrees per year)
                trend = model.coef_[0]
                
                location_trends.append({
                    'latitude': lat,
                    'longitude': lon,
                    'temperature_trend': trend,
                    'years_of_data': len(group),
                    'avg_temperature': group['temperature'].mean()
                })
        
        if location_trends:
            trends_df = pd.DataFrame(location_trends)
            
            # Train a model to predict temperature trends
            feature_cols = ['latitude', 'longitude', 'avg_temperature']
            X = trends_df[feature_cols].copy()
            y = trends_df['temperature_trend'].copy()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, 'Temperature Trend Analysis')
            
            # Save model
            model_dir = os.path.join(self.models_dir, 'forecasting')
            model_path = os.path.join(model_dir, 'temperature_trend_model.joblib')
            scaler_path = os.path.join(model_dir, 'temperature_trend_scaler.joblib')
            metrics_path = os.path.join(model_dir, 'temperature_trend_metrics.json')
            trends_path = os.path.join(model_dir, 'location_temperature_trends.csv')
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            trends_df.to_csv(trends_path, index=False)
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"💾 Temperature trend model saved: {model_path}")
            logger.info(f"📊 Global average trend: {trends_df['temperature_trend'].mean():.4f}°C/year")
            
            return model, scaler, metrics
        else:
            logger.warning("⚠️  Insufficient data for temperature trend analysis")
            return None
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'model_name': model_name,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'samples': len(y_true),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"📊 {model_name} Metrics:")
        logger.info(f"   MAE: {mae:.3f}")
        logger.info(f"   RMSE: {rmse:.3f}")
        logger.info(f"   R²: {r2:.3f}")
        
        return metrics
    
    def train_all_event_suitability_models(self):
        """Train suitability models for all event types"""
        logger.info("🎪 Training all event suitability models...")
        
        trained_models = {}
        
        for event_type in EVENT_TYPES:
            event_name = event_type.lower().replace(' ', '_')
            
            try:
                model, scaler, metrics = self.train_event_suitability_model(event_name)
                trained_models[event_name] = {
                    'model': model,
                    'scaler': scaler,
                    'metrics': metrics
                }
                logger.info(f"✅ {event_type} model trained successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {event_type} model: {e}")
        
        logger.info(f"🎉 Trained {len(trained_models)} event suitability models")
        return trained_models
    
    def create_model_summary(self):
        """Create summary of all trained models"""
        logger.info("📋 Creating model summary...")
        
        summary = {
            'created_at': datetime.now().isoformat(),
            'models_directory': self.models_dir,
            'trained_models': []
        }
        
        # Scan models directory
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith('.joblib') or file.endswith('.h5'):
                    model_path = os.path.join(root, file)
                    rel_path = os.path.relpath(model_path, self.models_dir)
                    
                    # Try to load metrics
                    metrics_file = model_path.replace('.joblib', '_metrics.json').replace('.h5', '_metrics.json')
                    metrics = {}
                    
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                    
                    summary['trained_models'].append({
                        'model_file': rel_path,
                        'model_type': file.split('_')[0],
                        'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                        'metrics': metrics
                    })
        
        # Save summary
        summary_path = os.path.join(self.models_dir, 'models_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Model summary saved: {summary_path}")
        logger.info(f"🤖 Total models: {len(summary['trained_models'])}")
        
        return summary

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ML models for WeatherWise app')
    parser.add_argument('--model', choices=[
        'weather-risk', 'event-suitability', 'precipitation-forecast', 
        'temperature-trend', 'all'
    ], default='all', help='Model type to train')
    parser.add_argument('--data-dir', default=os.path.join(PROCESSED_DATA_DIR, 'training'),
                       help='Training data directory')
    parser.add_argument('--models-dir', default=MODELS_DIR,
                       help='Models output directory')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--event-type', default='wedding',
                       help='Event type for event-suitability model')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting WeatherWise ML Training Pipeline")
    logger.info(f"🎯 Target model: {args.model}")
    
    # Initialize trainer
    trainer = WeatherMLTrainer(args.data_dir, args.models_dir)
    
    try:
        if args.model == 'weather-risk' or args.model == 'all':
            logger.info("=" * 50)
            trainer.train_weather_risk_model(args.hyperparameter_tuning)
        
        if args.model == 'event-suitability':
            logger.info("=" * 50)
            trainer.train_event_suitability_model(args.event_type, args.hyperparameter_tuning)
        elif args.model == 'all':
            logger.info("=" * 50)
            trainer.train_all_event_suitability_models()
        
        if args.model == 'precipitation-forecast' or args.model == 'all':
            logger.info("=" * 50)
            trainer.train_precipitation_forecast_model()
        
        if args.model == 'temperature-trend' or args.model == 'all':
            logger.info("=" * 50) 
            trainer.train_temperature_trend_model()
        
        # Create model summary
        trainer.create_model_summary()
        
        logger.info("🎉 ML training pipeline complete!")
        logger.info("🤖 Models ready for deployment to WeatherWise backend")
        
    except KeyboardInterrupt:
        logger.info("⛔ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ ML training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()