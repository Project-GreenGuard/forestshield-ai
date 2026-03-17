"""
ForestShield AI - Inference Module

Prototype inference system for wildfire risk prediction.
Uses trained Gradient Boosting model with 11-feature contract.

Reference: forestshield/docs/AI_PREDICTION_AND_TRAINING_SPEC.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.features import build_feature_vector as build_features_array, FEATURE_NAMES


# Global model cache
_MODEL = None
_METADATA = None
_MODEL_PATH = Path(__file__).parent.parent / "models" / "wildfire_risk_model.pkl"
_METADATA_PATH = Path(__file__).parent.parent / "models" / "wildfire_risk_metadata.pkl"


def load_model():
    """Load trained model from disk (cached)."""
    global _MODEL, _METADATA
    if _MODEL is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")
        _MODEL = joblib.load(_MODEL_PATH)
        
        # Load metadata if available (optional)
        if _METADATA_PATH.exists():
            _METADATA = joblib.load(_METADATA_PATH)
        else:
            _METADATA = {}
    
    return _MODEL, _METADATA


def predict_risk(sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run wildfire risk prediction on sensor data.
    
    Args:
        sensor_payload: Dict with keys:
            - temperature (float): Temperature in °C
            - humidity (float): Humidity in %
            - lat (float): Latitude
            - lng (float): Longitude
            - nearestFireDistance (float): Distance to nearest fire in km
            - timestamp (str): ISO 8601 timestamp
    
    Returns:
        Dict with:
            - risk_score (float): 0-100
            - risk_level (str): LOW/MEDIUM/HIGH
            - confidence (float): Model confidence 0-1
            - model_version (str): Model identifier
    """
    # Load model
    model, metadata = load_model()
    
    # Build feature vector using 11-feature contract
    features = build_features_array(sensor_payload)
    features = features.reshape(1, -1)  # Shape for sklearn: (1, 11)
    
    # Predict risk score
    risk_score = model.predict(features)[0]
    risk_score = float(np.clip(risk_score, 0, 100))
    
    # Determine risk level (per spec thresholds)
    if risk_score <= 30:
        risk_level = "LOW"
    elif risk_score <= 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    # Confidence (simple heuristic based on distance to thresholds)
    if risk_level == "LOW":
        confidence = 0.9 if risk_score < 15 else 0.7
    elif risk_level == "MEDIUM":
        confidence = 0.85
    else:
        confidence = 0.95 if risk_score > 80 else 0.85
    
    return {
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'confidence': confidence,
        'model_version': metadata.get('timestamp', 'v1.0'),
    }


def main():
    """Test inference with sample sensor readings."""
    print("\n" + "=" * 70)
    print("    ForestShield AI - Model Inference")
    print("=" * 70 + "\n")
    
    # Load model
    print("Loading model...")
    model, metadata = load_model()
    print(f"[OK] Model version: {metadata.get('timestamp', 'unknown')}")
    print(f"[OK] Training method: {metadata.get('training_method', 'unknown')}")
    print(f"[OK] Performance: RMSE={metadata.get('test_rmse', 0):.2f}, R²={metadata.get('test_r2', 0):.3f}\n")
    
    # Test scenarios
    test_cases = [
        {
            'name': "HIGH RISK: Close to fire, hot & dry",
            'payload': {
                'temperature': 38.0,
                'humidity': 22.0,
                'lat': 49.2827,
                'lng': -123.1207,
                'nearestFireDistance': 2.5,
                'timestamp': '2024-07-15T15:30:00Z',
            }
        },
        {
            'name': "MEDIUM RISK: Moderate distance",
            'payload': {
                'temperature': 28.0,
                'humidity': 45.0,
                'lat': 51.0447,
                'lng': -114.0719,
                'nearestFireDistance': 25.0,
                'timestamp': '2024-06-20T14:00:00Z',
            }
        },
        {
            'name': "LOW RISK: Far from fire, cooler",
            'payload': {
                'temperature': 22.0,
                'humidity': 65.0,
                'lat': 43.6532,
                'lng': -79.3832,
                'nearestFireDistance': 85.0,
                'timestamp': '2024-05-10T10:00:00Z',
            }
        },
    ]
    
    print("="*70)
    print("TEST PREDICTIONS")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Temp: {test['payload']['temperature']}C, "
              f"Humidity: {test['payload']['humidity']}%, "
              f"Fire Distance: {test['payload']['nearestFireDistance']}km")
        
        result = predict_risk(test['payload'])
        
        print(f"   -> Risk Score: {result['risk_score']}/100")
        print(f"   -> Risk Level: {result['risk_level']}")
        print(f"   -> Confidence: {result['confidence']:.0%}")
    
    print("\n" + "="*70)
    print("[COMPLETE] INFERENCE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

