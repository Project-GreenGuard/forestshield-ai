"""
Synthetic Data Generation for Ontario Wildfire Risk

This module generates realistic synthetic wildfire sensor observations
for training the ForestShield risk prediction model.

Features:
- Seasonal patterns (fire season peaks in summer)
- Temperature-humidity correlation (real weather patterns)
- Fire proximity clusters (fires have spatial extent)
- Ontario-specific coordinates (lat 42-57°, lng -95 to -74°)
"""

import numpy as np
import pandas as pd


def generate_improved_synthetic_data(n_samples=3000, seed=42):
    """
    Generate improved synthetic Ontario wildfire sensor readings.
    
    Improvements:
    - Seasonal patterns (fire season peaks in summer)
    - Temperature-humidity correlation (real weather patterns)
    - Fire proximity clusters (fires have spatial extent)
    - Better risk score distribution (more realistic)
    
    Args:
        n_samples: Number of synthetic observations to generate
        seed: Random seed for reproducibility
        
    Returns:
        pandas.DataFrame with columns:
            temperature, humidity, lat, lng, nearest_fire_distance,
            hour_of_day, month, temp_normalized, humidity_inverse,
            fire_proximity_score, day_of_week, risk_score
    """
    rng = np.random.RandomState(seed)
    
    data = []
    
    # Ontario bounding box
    lat_min, lat_max = 42.0, 57.0
    lng_min, lng_max = -95.0, -74.0
    
    for _ in range(n_samples):
        # Seasonal variation (fire season biased toward summer/fall)
        month = rng.randint(1, 13)
        season_weight = 1.0 if month in [6, 7, 8, 9] else 0.5  # Higher in fire season
        
        # Temperature correlates with month
        base_temp = 5 + (month - 0.5) / 12 * 30  # 5°C in Jan, 35°C in July
        temperature = np.clip(base_temp + rng.normal(0, 5), -20, 42)
        
        # Humidity inversely correlated with temperature in fire season
        base_humidity = 70 if month < 6 or month > 9 else 50
        humidity = np.clip(base_humidity + rng.normal(0, 15) - (temperature - base_temp) * 0.5, 10, 95)
        
        # Location
        lat = rng.uniform(lat_min, lat_max)
        lng = rng.uniform(lng_min, lng_max)
        
        # Fire proximity (Zipfian: most locations have no nearby fire, some have very close)
        fire_prob = season_weight * (1 - humidity / 100) * (temperature / 40)
        if rng.random() < fire_prob:
            nearest_fire = max(0.5, rng.exponential(15))  # Exponential: mostly close
        else:
            nearest_fire = -1
        
        hour = rng.randint(0, 24)
        
        # Derived features
        temp_normalized = temperature / 50
        humidity_inverse = 1 - humidity / 100
        if nearest_fire > 0:
            fire_proximity_score = max(0, 100 - nearest_fire * 2)
        else:
            fire_proximity_score = 0
        
        day_of_week = rng.randint(0, 7)
        
        # GROUND TRUTH: Risk score based on fire science
        # Core factors: temperature, dryness, fire proximity, season
        base_risk = (
            (temperature / 40) * 0.25 +           # Hot = more risk
            humidity_inverse * 0.25 +              # Dry = more risk
            (season_weight - 0.5) * 0.1 +          # Fire season
            (np.log(hour + 1) / np.log(24)) * 0.05  # Peak danger 2-4pm
        )
        
        # Fire proximity multiplier (nearby fire = significantly more risk)
        if nearest_fire > 0:
            proximity_factor = 1 + (30 - min(nearest_fire, 30)) / 30  # 1-2x multiplier
        else:
            proximity_factor = 1.0
        
        risk_score = np.clip(base_risk * proximity_factor * 100, 0, 100)
        # Add slight noise to avoid perfect patterns
        risk_score += rng.normal(0, 3)
        risk_score = np.clip(risk_score, 0, 100)
        
        data.append({
            'temperature': temperature,
            'humidity': humidity,
            'lat': lat,
            'lng': lng,
            'nearest_fire_distance': nearest_fire,
            'hour_of_day': hour,
            'month': month,
            'temp_normalized': temp_normalized,
            'humidity_inverse': humidity_inverse,
            'fire_proximity_score': fire_proximity_score,
            'day_of_week': day_of_week,
            'risk_score': risk_score,
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Quick test
    print("Generating 100 sample observations...")
    df = generate_improved_synthetic_data(n_samples=100)
    
    print(f"\n✓ Generated {len(df)} observations")
    print(f"  Risk score range: {df['risk_score'].min():.1f} - {df['risk_score'].max():.1f}")
    print(f"  Mean risk: {df['risk_score'].mean():.1f}")
    print(f"  Temperature range: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
    print(f"  Humidity range: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
    print(f"\nSample rows:")
    print(df[['temperature', 'humidity', 'month', 'nearest_fire_distance', 'risk_score']].head(3))
