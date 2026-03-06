"""
ForestShield AI - Feature Engineering Module

Single source of truth for the 11-feature contract.
Used by both training and inference pipelines.

Feature Contract (11 features):
1. temperature (°C)
2. humidity (%)
3. lat (degrees)
4. lng (degrees)
5. nearestFireDistance (km)
6. temp_normalized (0-1)
7. humidity_inverse (0-1)
8. fire_proximity_score (0-1)
9. hour_sin (cyclical time)
10. hour_cos (cyclical time)
11. fire_danger_index (composite risk)
"""

import numpy as np
from datetime import datetime


# Feature names in order (locked contract)
FEATURE_NAMES = [
    'temperature',
    'humidity',
    'lat',
    'lng',
    'nearestFireDistance',
    'temp_normalized',
    'humidity_inverse',
    'fire_proximity_score',
    'hour_sin',
    'hour_cos',
    'fire_danger_index',
]


def build_feature_vector(payload: dict) -> np.ndarray:
    """
    Build 11-feature vector from sensor payload.
    
    Args:
        payload: Dict with keys:
            - temperature (float): Temperature in °C
            - humidity (float): Humidity in %
            - lat (float): Latitude
            - lng (float): Longitude
            - nearestFireDistance (float): Distance to nearest fire in km
            - timestamp (str): ISO 8601 timestamp (e.g. "2024-06-15T14:30:00Z")
    
    Returns:
        numpy array of shape (11,) with features in FEATURE_NAMES order
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Validate required fields
    required = ['temperature', 'humidity', 'lat', 'lng', 'nearestFireDistance', 'timestamp']
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    # Extract raw features
    temperature = float(payload['temperature'])
    humidity = float(payload['humidity'])
    lat = float(payload['lat'])
    lng = float(payload['lng'])
    fire_distance = float(payload['nearestFireDistance'])
    timestamp = payload['timestamp']
    
    # Validate ranges
    if not (10 <= temperature <= 45):
        raise ValueError(f"Temperature {temperature}°C out of valid range [10, 45]")
    if not (15 <= humidity <= 80):
        raise ValueError(f"Humidity {humidity}% out of valid range [15, 80]")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of valid range [-90, 90]")
    if not (-180 <= lng <= 180):
        raise ValueError(f"Longitude {lng} out of valid range [-180, 180]")
    if fire_distance < 0:
        raise ValueError(f"Fire distance {fire_distance} cannot be negative")
    
    # Parse hour from timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        hour = dt.hour
    except:
        # Default to noon if timestamp invalid
        hour = 12
    
    # Derived feature 1: Temperature normalized (0-1)
    temp_normalized = (temperature - 10) / (45 - 10)
    temp_normalized = np.clip(temp_normalized, 0, 1)
    
    # Derived feature 2: Humidity inverse (0-1)
    # Lower humidity = higher risk
    humidity_inverse = 1.0 - ((humidity - 15) / (80 - 15))
    humidity_inverse = np.clip(humidity_inverse, 0, 1)
    
    # Derived feature 3: Fire proximity score (0-1)
    # Exponential decay with distance
    fire_proximity_score = np.exp(-fire_distance / 20.0)
    
    # Derived feature 4-5: Hour of day (cyclical encoding)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    # Derived feature 6: Fire danger index (composite)
    # Combines temperature, humidity, and fire proximity
    fdi = (
        0.4 * temp_normalized +
        0.3 * humidity_inverse +
        0.3 * fire_proximity_score
    )
    
    # Assemble feature vector in order
    features = np.array([
        temperature,
        humidity,
        lat,
        lng,
        fire_distance,
        temp_normalized,
        humidity_inverse,
        fire_proximity_score,
        hour_sin,
        hour_cos,
        fdi,
    ], dtype=np.float64)
    
    return features


def validate_feature_vector(features: np.ndarray) -> bool:
    """
    Validate that feature vector matches contract.
    
    Args:
        features: numpy array to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(features, np.ndarray):
        return False
    
    if features.shape != (11,):
        return False
    
    if not np.all(np.isfinite(features)):
        return False
    
    return True


def get_feature_importance_description() -> dict:
    """
    Get human-readable descriptions of each feature.
    
    Returns:
        Dict mapping feature names to descriptions
    """
    return {
        'temperature': 'Raw temperature reading in °C',
        'humidity': 'Raw humidity reading in %',
        'lat': 'Sensor latitude coordinate',
        'lng': 'Sensor longitude coordinate',
        'nearestFireDistance': 'Distance to nearest known fire in km',
        'temp_normalized': 'Temperature scaled to 0-1 (higher = more dangerous)',
        'humidity_inverse': 'Inverted humidity (lower humidity = higher risk)',
        'fire_proximity_score': 'Exponential decay score based on fire distance',
        'hour_sin': 'Sine component of hour (captures time of day cyclically)',
        'hour_cos': 'Cosine component of hour (captures time of day cyclically)',
        'fire_danger_index': 'Composite risk index (temp + humidity + proximity)',
    }