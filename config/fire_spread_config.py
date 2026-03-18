"""
Configuration for fire spread forecasting (PBI-6).
Contains constants and thresholds for fire prediction.
"""

# Fire spread model constants
FIRE_SPREAD_CONFIG = {
    # Base fire speed (km/h)
    'base_speed': 2.5,
    
    # Environmental factors
    'wind_factor': 0.8,  # Wind influence multiplier
    'slope_factor': 0.6,  # Terrain slope multiplier
    'fuel_factor': 0.4,   # Vegetation/fuel multiplier
    
    # Speed limits
    'min_speed': 0.5,  # km/h
    'max_speed': 15.0,  # km/h
    
    # Fuel types and their modifiers
    'fuel_modifiers': {
        'grass': 1.1,
        'forest': 1.3,
        'mixed': 1.0,
        'urban': 0.7,
        'water': 0.0,
    },
    
    # Danger level thresholds
    'danger_thresholds': {
        'CRITICAL': 2.0,
        'HIGH': 1.5,
        'MEDIUM': 1.0,
        'LOW': 0.0,
    },
}

# Threat assessment thresholds
THREAT_ASSESSMENT = {
    'none_distance': 20,      # km - No threat beyond this
    'low_distance': 10,       # km
    'medium_distance': 5,     # km
    'high_distance': 2,       # km
    
    'evacuation_threshold_hours': 12,  # Evacuate if arrival < 12h
}

# Forecast horizons
FORECAST_HORIZONS = {
    'short_term': 6,   # hours
    'medium_term': 24,  # hours
    'long_term': 72,    # hours
}

# Model confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.85,
    'medium': 0.70,
    'low': 0.50,
}

# Alert criteria
ALERT_CRITERIA = {
    'critical_intensity': 75,
    'high_intensity': 60,
    'medium_intensity': 40,
    
    'critical_speed': 8,      # km/h
    'high_speed': 5,          # km/h
    'medium_speed': 3,        # km/h
}

# Retraining triggers for fire spread model
RETRAINING_TRIGGERS = {
    'prediction_error_threshold': 0.15,  # 15% error
    'samples_before_retraining': 500,
    'model_age_days': 30,
}