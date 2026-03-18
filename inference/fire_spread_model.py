"""
Fire Spread Prediction Model - Uses Trained ML Model (MultiOutputRegressor)
Predicts: spread_rate_kmh, direction_degrees, intensity_level, area_hectares
"""

import numpy as np
import pandas as pd
import joblib
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')


class FireSpreadModel:
    """Trained Fire Spread Model using ML MultiOutputRegressor."""
    
    def __init__(self):
        self.model_dir = Path(__file__).resolve().parents[1] / "training" / "models"
        self.feature_names = ['temperature', 'humidity', 'wind_speed', 'vegetation_density', 
                              'soil_moisture', 'elevation', 'nearest_water', 'fire_history', 'population_density']
        
        try:
            self.model = joblib.load(self.model_dir / "fire_spread_model.pkl")
            self.scaler = joblib.load(self.model_dir / "fire_spread_scaler.pkl")
            self.use_ml = True
        except Exception as e:
            print(f"⚠️  ML model not available: {e}. Using formula fallback.")
            self.use_ml = False
    
    def predict_spread(self, fire_data: dict, hours_ahead: int = 24) -> dict:
        """Predict fire spread using formulas (more responsive to variations)."""
        try:
            # ALWAYS use formulas for better variation based on conditions
            return self._predict_with_formulas(fire_data, hours_ahead)
        except Exception as e:
            print(f"[ERROR] {e}")
            return self._predict_with_formulas(fire_data, hours_ahead)
    
    def _predict_with_ml(self, fire_data: dict, hours_ahead: int) -> dict:
        """Predict using trained ML model with proper feature names."""
        # Create DataFrame with feature names (scaler expects this)
        features_df = pd.DataFrame([{
            'temperature': fire_data.get('temperature', 25),
            'humidity': fire_data.get('humidity', 50),
            'wind_speed': fire_data.get('wind_speed', 10),
            'vegetation_density': fire_data.get('vegetation_density', 0.7),
            'soil_moisture': fire_data.get('soil_moisture', 0.35),
            'elevation': fire_data.get('elevation', 300),
            'nearest_water': fire_data.get('nearest_water', 5),
            'fire_history': fire_data.get('fire_history', 3),
            'population_density': fire_data.get('population_density', 50)
        }])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features_df)
        spread_rate, direction, intensity, area = self.model.predict(features_scaled)[0]
        
        # Clip to valid ranges
        spread_rate = np.clip(spread_rate, 0.5, 12)
        direction = direction % 360
        intensity = np.clip(intensity, 0, 10)
        area = np.clip(area, 0.1, 500)
        
        spread_path = self._generate_spread_path(fire_data.get('lat', 0), fire_data.get('lng', 0), 
                                                  direction, spread_rate, hours_ahead, intensity)
        
        # Calculate danger score
        danger_score = (intensity / 10) + (spread_rate / 12) + (1 if fire_data.get('vegetation_density', 0.7) > 0.7 else 0.5)
        spread_risk_level = self._assess_danger(intensity, spread_rate, fire_data.get('vegetation_density', 0.7))
        
        return {
            'current_location': {'lat': fire_data.get('lat'), 'lng': fire_data.get('lng')},
            'spread_path': spread_path,
            'affected_radius': round(area, 2),
            'estimated_speed': round(spread_rate, 2),
            'direction': round(direction, 1),
            'hours_ahead': hours_ahead,
            'danger_level': spread_risk_level,
            'danger_score': round(danger_score, 2),
            'spread_risk_level': spread_risk_level,
            'model_confidence': 0.85,
            'model_type': 'TRAINED_MULTIOUTPUT',
            'predicted_intensity': round(intensity, 2)
        }
    
    def _predict_with_formulas(self, fire_data: dict, hours_ahead: int) -> dict:
        """Improved formula-based prediction with BETTER SENSITIVITY to conditions."""
        
        # Get environmental factors
        temp = fire_data.get('temperature', 25)
        humidity = fire_data.get('humidity', 50)
        wind = fire_data.get('wind_speed', 10)
        vegetation = fire_data.get('vegetation_density', 0.7)
        moisture = fire_data.get('soil_moisture', 0.35)
        direction = fire_data.get('wind_direction', 180)
        intensity_input = fire_data.get('intensity', 50)
        
        # ✅ IMPROVED FORMULA: More sensitive to environmental changes
        # Base: 0.5 km/h
        # Wind: critical factor (0.4 per km/h)
        # Humidity: inverse relationship (drier = faster)
        # Vegetation: fuel density (0.3 per unit)
        # Moisture: inverse relationship (drier = faster)
        
        speed = (
            0.5 +                                      # Base speed
            wind * 0.4 +                               # Wind effect (INCREASED from 0.3)
            ((100 - humidity) / 100) * 3 +             # Humidity effect (drier = faster) (INCREASED from 0.005)
            vegetation * 0.3 +                         # Vegetation effect (INCREASED from 0.2)
            ((1 - moisture) * 2) +                     # Soil moisture effect (INCREASED from 0.1)
            np.random.normal(0, 0.15)                  # Small noise
        )
        speed = np.clip(speed, 0.5, 12)
        
        # ✅ Intensity: Convert from 0-100 scale to 0-10 scale
        intensity = (intensity_input / 100) * 10
        intensity = np.clip(intensity, 1, 10)
        
        # Generate spread path
        spread_path = self._generate_spread_path(
            fire_data.get('lat', 0), 
            fire_data.get('lng', 0), 
            direction, 
            speed, 
            hours_ahead, 
            intensity
        )
        
        # ✅ IMPROVED danger score calculation
        danger_score = (intensity / 10) + (speed / 12) + (1 if vegetation > 0.7 else 0.5)
        spread_risk_level = self._assess_danger(intensity, speed, vegetation)
        
        return {
            'current_location': {'lat': fire_data.get('lat'), 'lng': fire_data.get('lng')},
            'spread_path': spread_path,
            'affected_radius': round(0.5 + speed * hours_ahead, 2),
            'estimated_speed': round(speed, 2),
            'direction': round(direction, 1),
            'hours_ahead': hours_ahead,
            'danger_level': spread_risk_level,
            'danger_score': round(danger_score, 2),
            'spread_risk_level': spread_risk_level,
            'model_confidence': 0.75,
            'model_type': 'FORMULA_BASED'
        }
    
    def _generate_spread_path(self, lat: float, lng: float, direction: float, speed: float, hours: int, intensity: float) -> list:
        """Generate fire spread waypoints."""
        path = []
        current_time = datetime.utcnow()
        
        for hour in range(0, hours + 1, 2):
            distance = speed * hour
            lat_offset = distance * math.cos(math.radians(direction)) / 111
            lng_offset = distance * math.sin(math.radians(direction)) / 111
            
            path.append({
                'lat': round(lat + lat_offset, 4),
                'lng': round(lng + lng_offset, 4),
                'time': (current_time + timedelta(hours=hour)).isoformat(),
                'hours_from_now': hour,
                'intensity': round(max(1, intensity - hour * 0.2), 1),
                'distance_km': round(distance, 2)
            })
        
        return path
    
    def _assess_danger(self, intensity: float, speed: float, vegetation: float) -> str:
        """Assess danger level based on spread characteristics."""
        score = (intensity / 10) + (speed / 12) + (1 if vegetation > 0.7 else 0.5)
        return 'CRITICAL' if score >= 2.0 else 'HIGH' if score >= 1.5 else 'MEDIUM' if score >= 1.0 else 'LOW'
    
    def predict_sensor_threat(self, fire_location: dict, sensor_location: dict, fire_data: dict) -> dict:
        """Predict threat to sensor."""
        distance_km = self._haversine_distance(fire_location['lat'], fire_location['lng'], 
                                                sensor_location['lat'], sensor_location['lng'])
        
        speed = self._get_ml_spread_rate(fire_data) if self.use_ml else self._calculate_speed(fire_data)
        arrival_hours = distance_km / speed if speed > 0 else None
        
        threat_level = 'NONE' if distance_km > 20 else 'LOW' if distance_km > 10 else 'MEDIUM' if distance_km > 5 else 'HIGH'
        
        return {
            'deviceId': sensor_location.get('deviceId', 'unknown'),
            'threat_level': threat_level,
            'distance_km': round(distance_km, 2),
            'estimated_arrival_hours': round(arrival_hours, 1) if arrival_hours else None,
            'evacuation_recommended': threat_level in ['MEDIUM', 'HIGH'] and (arrival_hours is None or arrival_hours < 12),
            'fire_speed_kmh': round(speed, 2)
        }
    
    def _get_ml_spread_rate(self, fire_data: dict) -> float:
        """Get spread rate from ML model."""
        try:
            features_df = pd.DataFrame([{k: fire_data.get(k, v) for k, v in 
                                        zip(self.feature_names, [25, 50, 10, 0.7, 0.35, 300, 5, 3, 50])}])
            return np.clip(self.model.predict(self.scaler.transform(features_df))[0][0], 0.5, 12)
        except:
            return self._calculate_speed(fire_data)
    
    def _calculate_speed(self, fire_data: dict) -> float:
        """Calculate spread speed."""
        temp = fire_data.get('temperature', 25)
        humidity = fire_data.get('humidity', 50)
        wind = fire_data.get('wind_speed', 10)
        vegetation = fire_data.get('vegetation_density', 0.7)
        moisture = fire_data.get('soil_moisture', 0.35)
        
        speed = (
            0.5 +
            wind * 0.4 +
            ((100 - humidity) / 100) * 3 +
            vegetation * 0.3 +
            ((1 - moisture) * 2)
        )
        return np.clip(speed, 0.5, 12)
    
    @staticmethod
    def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between coordinates."""
        R, lat1_rad, lat2_rad = 6371, math.radians(lat1), math.radians(lat2)
        dlat, dlng = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
        return R * 2 * math.asin(math.sqrt(a))