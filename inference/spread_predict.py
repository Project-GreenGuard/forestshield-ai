"""
Fire Spread Forecasting Inference

Real-time prediction of how wildfire will spread given current conditions.

Usage:
    from inference.spread_predict import predict_spread
    
    forecast = predict_spread(
        temperature=35.0,
        humidity=20.0,
        wind_speed_kmh=25.0,
        vegetation_type="forest",
        topography="hilly"
    )
    
    print(f"Fire will spread at {forecast['spread_rate_kmh']:.1f} km/h")
    print(f"~{forecast['burn_area_24h']:.0f} hectares in 24 hours")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import joblib

from inference.predict import predict_risk, build_feature_vector, FEATURE_COLUMNS

_SPREAD_MODEL = None

VEGETATION_TYPES = ["grassland", "shrubland", "forest", "mixed"]
TOPOGRAPHY_TYPES = ["flat", "rolling", "hilly", "mountainous"]


def _load_spread_model() -> Optional[Any]:
    """Load cached spread rate forecasting model."""
    global _SPREAD_MODEL

    if _SPREAD_MODEL is not None:
        return _SPREAD_MODEL

    model_dir = Path(__file__).parent.parent / "models"

    try:
        spread_path = model_dir / "forestshield_spread_v1.joblib"

        if spread_path.exists():
            _SPREAD_MODEL = joblib.load(spread_path)
            print(f"[ForestShield AI] Loaded spread forecasting model")
            return _SPREAD_MODEL
    except Exception as e:
        print(f"[ForestShield AI] Could not load spread model: {e}")

    return None

def predict_spread(
    temperature: float,
    humidity: float,
    lat: float,
    lng: float,
    nearestFireDistance: float = -1.0,
    timestamp: Optional[str] = None,
    wind_speed_kmh: float = 10.0,
    wind_direction_deg: int = 180,
    fuel_moisture_percent: float = 50.0,
    canopy_density: float = 0.5,
    vegetation_type: str = "forest",
    topography: str = "hilly",
) -> Dict[str, Any]:
    """
    Predict fire spread rate and burn area given weather and terrain conditions.

    Parameters
    ----------
    # Risk prediction features
    temperature : float, °C
    humidity : float, %
    lat : float, decimal degrees (latitude)
    lng : float, decimal degrees (longitude)
    nearestFireDistance : float, km (or -1 if none)
    timestamp : ISO-8601 string (optional, uses now if omitted)

    # Spread-specific features
    wind_speed_kmh : float, km/h (0-50 typical range)
    wind_direction_deg : int, degrees (0-360)
    fuel_moisture_percent : float, % (5-100)
    canopy_density : float, 0-1 (0=open to 1=dense forest)
    vegetation_type : str, one of {grassland, shrubland, forest, mixed}
    topography : str, one of {flat, rolling, hilly, mountainous}

    Returns
    -------
    dict with keys:
    - risk_score : float (0-100)
    - risk_level : str (LOW/MEDIUM/HIGH)
    - spread_rate_kmh : float, fire spread velocity
    - burn_area_24h : float, hectares burned in 24 hours
    - fireline_intensity : float, kW/m (fire intensity)
    - timestamp : ISO-8601 of prediction

    Interpretation:
    - spread_rate_kmh: How fast fire front advances
    - burn_area_24h: Total area consumed in 24 hours
    - fireline_intensity: Heat release (suppression difficulty)
    """
    # 1. Get risk assessment from existing model
    sensor_payload = {
        "temperature": temperature,
        "humidity": humidity,
        "lat": lat,
        "lng": lng,
        "nearestFireDistance": nearestFireDistance,
        "timestamp": timestamp or __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }
    features = build_feature_vector(sensor_payload)
    risk_result = predict_risk(features)

    # 2. Load spread forecasting model
    spread_model = _load_spread_model()

    if spread_model is None:
        # Fallback: simple heuristic
        return _fallback_spread_forecast(
            temperature,
            humidity,
            wind_speed_kmh,
            canopy_density,
            risk_result,
        )

    # 3. Build extended feature vector for spread prediction
    time_features = features.copy()  # risk features

    # Add spread-specific features
    extended_features = [
        time_features.get("temperature", 0),
        time_features.get("humidity", 0),
        time_features.get("lat", 0),
        time_features.get("lng", 0),
        time_features.get("nearest_fire_distance", -1.0),
        time_features.get("hour_of_day", 0),
        time_features.get("month", 0),
        # Spread features
        wind_speed_kmh,
        wind_direction_deg,
        fuel_moisture_percent,
        canopy_density,
        # Categorical (encoded: grassland=0, shrubland=1, forest=2, mixed=3)
        VEGETATION_TYPES.index(vegetation_type) if vegetation_type in VEGETATION_TYPES else 2,
        # Categorical (encoded: flat=0, rolling=1, hilly=2, mountainous=3)
        TOPOGRAPHY_TYPES.index(topography) if topography in TOPOGRAPHY_TYPES else 2,
    ]

    X = np.array([extended_features])

    # 4. Predict spread rate and derive burn area
    spread_rate = float(spread_model.predict(X)[0])
    burn_area_24h = spread_rate * 20  # 20 hectares per km/h spread rate

    # 5. Estimate fireline intensity (kW/m)
    # Simplified: depends on fuel and temperature
    fireline_intensity = (
        spread_rate * 100  # base from spread rate
        + (canopy_density * 50)  # fuel load
        + max((temperature - 20) / 25 * 100, 0)  # temperature effect
    )

    return {
        # Risk assessment
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "model_version": "v1-spread-forecast",
        # Spread forecast
        "spread_rate_kmh": round(spread_rate, 2),
        "burn_area_24h_hectares": round(burn_area_24h, 1),
        "fireline_intensity_kw_m": round(fireline_intensity, 1),
        # Context
        "wind_speed_kmh": wind_speed_kmh,
        "vegetation_type": vegetation_type,
        "topography": topography,
        "timestamp": sensor_payload["timestamp"],
    }

def _fallback_spread_forecast(
    temperature: float,
    humidity: float,
    wind_speed: float,
    canopy_density: float,
    risk_result: dict,
) -> dict:
    """
    Fallback heuristic spread forecast when models unavailable.
    Based on fire science principles.
    """
    # Spread score: hot + dry + windy + dense fuel
    spread_score = (
        (temperature / 50) * 0.3
        + (1 - humidity / 100) * 0.3
        + (wind_speed / 50) * 0.2
        + canopy_density * 0.2
    )
    spread_rate = np.clip(spread_score * 5.0, 0.1, 5.0)
    burn_area = spread_rate * 20  # 20 hectares per km/h spread

    return {
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "model_version": "v1-spread-heuristic",
        "spread_rate_kmh": round(spread_rate, 2),
        "burn_area_24h_hectares": round(burn_area, 1),
        "fireline_intensity_kw_m": round(spread_rate * 100, 1),
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "note": "Using heuristic (models not available)",
    }

# Smoke Test - Test spread forecasting module independently
if __name__ == "__main__":
    print("=" * 80)
    print("ForestShield AI — Fire Spread Forecasting Smoke Test")
    print("=" * 80)

    test_cases = [
        {
            "label": "Mild conditions (low spread)",
            "params": {
                "temperature": 15.0,
                "humidity": 75.0,
                "lat": 45.0,
                "lng": -85.0,
                "wind_speed_kmh": 5.0,
                "vegetation_type": "grassland",
                "topography": "flat",
            },
        },
        {
            "label": "Dangerous conditions (high spread)",
            "params": {
                "temperature": 35.0,
                "humidity": 20.0,
                "lat": 50.0,
                "lng": -87.0,
                "wind_speed_kmh": 30.0,
                "vegetation_type": "forest",
                "topography": "hilly",
            },
        },
        {
            "label": "Extreme conditions (very high spread)",
            "params": {
                "temperature": 40.0,
                "humidity": 10.0,
                "lat": 52.0,
                "lng": -86.0,
                "wind_speed_kmh": 40.0,
                "vegetation_type": "forest",
                "topography": "mountainous",
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        forecast = predict_spread(**case["params"])
        print(f"\nCase {i}: {case['label']}")
        print(f"  Risk Level: {forecast['risk_level']} (score: {forecast['risk_score']:.1f})")
        print(
            f"  Spread Rate: {forecast['spread_rate_kmh']:.2f} km/h"
        )
        print(
            f"  Burn Area (24h): {forecast['burn_area_24h_hectares']:.1f} hectares"
        )
        print(f"  Fire Intensity: {forecast['fireline_intensity_kw_m']:.0f} kW/m")

    print("[OK] Fire Spread Forecasting system operational!")