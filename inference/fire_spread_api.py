import joblib
import numpy as np
import pandas as pd
import math
from pathlib import Path

# Load trained model and scaler
MODEL_DIR = Path(__file__).resolve().parents[1] / "training" / "models"

fire_spread_model = joblib.load(MODEL_DIR / "fire_spread_model.pkl")
fire_spread_scaler = joblib.load(MODEL_DIR / "fire_spread_scaler.pkl")

# Feature names (must match training)
FEATURE_NAMES = [
    'temperature', 'humidity', 'wind_speed', 'vegetation_density',
    'soil_moisture', 'elevation', 'nearest_water', 'fire_history',
    'population_density'
]

def predict_fire_spread(data: dict, hours: int = 24) -> dict:
    """Predict fire spread using trained ML model."""
    try:
        # Create feature array with proper column names
        features_dict = {
            'temperature': data.get('temperature', 20),
            'humidity': data.get('humidity', 50),
            'wind_speed': data.get('wind_speed', 10),
            'vegetation_density': data.get('vegetation_density', 0.7),
            'soil_moisture': data.get('soil_moisture', 0.3),
            'elevation': data.get('elevation', 300),
            'nearest_water': data.get('nearest_water', 5),
            'fire_history': data.get('fire_history', 3),
            'population_density': data.get('population_density', 20)
        }
        
        # Convert to DataFrame with feature names
        features_df = pd.DataFrame([features_dict])
        
        # Scale features
        features_scaled = fire_spread_scaler.transform(features_df)
        
        # Predict risk score
        risk_score = fire_spread_model.predict(features_scaled)[0]
        risk_score = max(0, min(100, risk_score))
        
        # Convert to spread rate
        spread_rate = 2.0 + (risk_score / 100) * 6.0
        
        # Intensity from risk score
        intensity = max(1, min(10, risk_score / 10))
        
        # Generate timeline
        timeline = []
        start_lat = data.get('latitude', 43.65)
        start_lng = data.get('longitude', -79.38)
        direction = data.get('wind_direction', 180)
        
        for h in range(0, hours + 1, 2):
            distance_km = spread_rate * h
            lat_offset = distance_km * math.cos(math.radians(direction)) / 111
            lng_offset = distance_km * math.sin(math.radians(direction)) / 111
            
            timeline.append({
                'hours': h,
                'area': 1.2 + (h * 0.3),
                'perimeter': [],
                'lat': start_lat + lat_offset,
                'lng': start_lng + lng_offset
            })
        
        # ✅ FIXED: Return keys match backend expectations
        return {
            'spread_rate': round(spread_rate, 2),
            'direction': round(direction, 1),
            'area': round(1.2 + (hours * 0.3), 2),
            'intensity': round(intensity, 1),
            'perimeter_polygon': [],
            'timeline': timeline,
            'model_type': 'ML_TRAINED',
            'risk_score': round(risk_score, 2)
        }
    
    except Exception as e:
        print(f"[ERROR] ML fire spread prediction failed: {e}")
        return {
            'spread_rate': 2.3,
            'direction': data.get('wind_direction', 180),
            'area': 1.2 + (hours * 0.3),
            'intensity': 7,
            'perimeter_polygon': [],
            'timeline': [],
            'model_type': 'FALLBACK',
            'error': str(e)
        }