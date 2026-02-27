"""
Inference helpers for the ForestShield wildfire risk model.

Usage
-----
    from inference.predict import build_feature_vector, predict_risk

    features = build_feature_vector(sensor_payload)
    result   = predict_risk(features)
    # result == {"risk_score": 42.7, "risk_level": "MEDIUM", "model_version": "..."}

Model loading
-------------
set the environment variable ``FORESTSHIELD_MODEL_PATH`` to the path of
a joblib-serialised scikit-learn model.  If the variable is unset or the
file is missing the function falls back to the deterministic rule-based
scoring that mirrors the backend Lambda so the system stays functional
without a trained artefact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# When this file is run directly (python inference/predict.py) the repo root
# is not automatically on sys.path, so add it now before any local imports.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from utils import risk_level_from_score, rule_based_risk_score

_MODEL = None          # cached model object (or False if unavailable)
_MODEL_PATH_ENV = "FORESTSHIELD_MODEL_PATH"
_FALLBACK_VERSION = "v1-rule-based"
_CALIBRATOR = None     # cached calibrator for v3+ models


def _load_model() -> Optional[Any]:
    """
    Try to load a joblib model from ``FORESTSHIELD_MODEL_PATH``.
    Falls back to v4 (optimized) → v3 (calibrated) → v2 (original) if env var not set.
    Returns the model object on success, ``None`` otherwise.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL if _MODEL is not False else None

    model_path = os.getenv(_MODEL_PATH_ENV)
    
    # Fallback to best available model (v4 > v3 > v2)
    if not model_path:
        models_dir = Path(__file__).parent.parent / "models"
        for model_name in ["forestshield_v4.joblib", "forestshield_v3.joblib", "forestshield_v2.joblib"]:
            candidate = models_dir / model_name
            if candidate.exists():
                model_path = str(candidate)
                break

    if not model_path:
        _MODEL = False
        return None

    try:
        import joblib  # type: ignore
        _MODEL = joblib.load(model_path)
        print(f"[ForestShield AI] Loaded model from {model_path}")
        return _MODEL
    except Exception as exc:  # noqa: BLE001
        print(f"[ForestShield AI] Could not load model ({exc}); using rule-based fallback.")
        _MODEL = False
        return None

# Ordered list of feature names expected by a trained sklearn model.
# Must match the column order used during training (see training/train.py).
FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "lat",
    "lng",
    "nearest_fire_distance",
    "hour_of_day",
    "month",
]


def build_feature_vector(sensor_payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a raw sensor payload into a flat feature dict.

    Expected keys (aligned with backend / DynamoDB):
    - temperature        : float, °C
    - humidity           : float, %
    - lat                : float, decimal degrees
    - lng                : float, decimal degrees
    - nearestFireDistance: float, km  (optional; defaults to -1 = unknown)
    - timestamp          : ISO-8601 string (optional; used for time features)

    Returns
    -------
    dict with keys matching ``FEATURE_COLUMNS`` plus derived features for v4+.
    """
    temperature = float(sensor_payload.get("temperature", 0.0))
    humidity = float(sensor_payload.get("humidity", 0.0))
    lat = float(sensor_payload.get("lat", 0.0))
    lng = float(sensor_payload.get("lng", 0.0))

    # nearestFireDistance: use -1 to signal "unknown" to the model.
    raw_dist = sensor_payload.get("nearestFireDistance")
    nearest_fire_distance = float(raw_dist) if raw_dist is not None else -1.0

    # Time-based features derived from timestamp (default to UTC now).
    ts_str = sensor_payload.get("timestamp")
    if ts_str:
        try:
            # Handle trailing 'Z' or offset-aware strings.
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    hour_of_day = float(ts.hour)
    month = float(ts.month)
    day_of_week = float(ts.weekday())

    # Derived features (for v4+ models)
    temp_normalized = temperature / 50.0
    humidity_inverse = 1.0 - (humidity / 100.0)
    if nearest_fire_distance > 0:
        fire_proximity_score = max(0.0, 100.0 - nearest_fire_distance * 2.0)
    else:
        fire_proximity_score = 0.0

    return {
        "temperature": temperature,
        "humidity": humidity,
        "lat": lat,
        "lng": lng,
        "nearest_fire_distance": nearest_fire_distance,
        "hour_of_day": hour_of_day,
        "month": month,
        "temp_normalized": temp_normalized,
        "humidity_inverse": humidity_inverse,
        "fire_proximity_score": fire_proximity_score,
        "day_of_week": day_of_week,
    }


def predict_risk(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Run a single prediction and return a structured result.

    Output contract (what the rest of the system expects):
    - risk_score  : float, 0–100
    - risk_level  : "LOW" | "MEDIUM" | "HIGH"
    - model_version: str

    Strategy
    --------
    1. If a trained model is available (``FORESTSHIELD_MODEL_PATH`` is set
       or defaults to v3/v2), use it.
    2. Otherwise fall back to the deterministic rule-based formula so the
       system works end-to-end without a trained artefact.
    """
    model = _load_model()

    if model is not None:
        # Build a 2-D array in the expected column order.
        import numpy as np  # type: ignore

    if model is not None:
        # Build a 2-D array in the expected column order.
        import numpy as np  # type: ignore

        # Use model's feature columns if available (v4), otherwise use standard columns
        feature_columns = getattr(model, 'feature_columns', FEATURE_COLUMNS)
        x = np.array([[features.get(col, 0.0) for col in feature_columns]])
        
        # Apply scaling if available (v4+ models)
        if hasattr(model, 'scaler'):
            x = model.scaler.transform(x)
        
        raw_score = float(model.predict(x)[0])
        
        # Apply calibration if available (v3+ models)
        if hasattr(model, 'calibrator'):
            calibrated_score = float(model.calibrator.predict([raw_score])[0])
            risk_score = round(min(max(calibrated_score, 0.0), 100.0), 2)
        else:
            risk_score = round(min(max(raw_score, 0.0), 100.0), 2)
        
        # Use optimized thresholds if available (v3+ models)
        if hasattr(model, 'thresholds'):
            low_thresh, high_thresh = model.thresholds
            if risk_score <= low_thresh:
                risk_level = "LOW"
            elif risk_score <= high_thresh:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
        else:
            risk_level = risk_level_from_score(risk_score)
        
        model_version = getattr(model, "forestshield_version", "v2-ml")
    else:
        # Rule-based fallback.
        fire_dist = features.get("nearest_fire_distance", -1.0)
        risk_score = rule_based_risk_score(
            temperature=features.get("temperature", 0.0),
            humidity=features.get("humidity", 0.0),
            fire_distance=fire_dist if fire_dist >= 0 else None,
        )
        risk_level = risk_level_from_score(risk_score)
        model_version = _FALLBACK_VERSION

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "model_version": model_version,
    }

# Quick smoke-test  —  test risk prediction module independently

if __name__ == "__main__":
    SAMPLES = [
        {
            "label": "Cool & humid — low risk, no nearby fire",
            "payload": {
                "deviceId": "esp32-demo",
                "temperature": 15.0,
                "humidity": 80.0,
                "lat": 49.25,
                "lng": -123.1,
                "timestamp": "2026-02-25T10:00:00Z",
            },
        },
        {
            "label": "Hot & dry — medium risk, no nearby fire",
            "payload": {
                "deviceId": "esp32-demo",
                "temperature": 35.0,
                "humidity": 25.0,
                "lat": 50.11,
                "lng": -119.5,
                "timestamp": "2026-07-15T14:30:00Z",
            },
        },
        {
            "label": "Hot, dry & fire 5 km away — high risk",
            "payload": {
                "deviceId": "esp32-demo",
                "temperature": 40.0,
                "humidity": 15.0,
                "lat": 51.0,
                "lng": -118.0,
                "nearestFireDistance": 5.0,
                "timestamp": "2026-08-01T16:00:00Z",
            },
        },
    ]

    print("=" * 60)
    print("ForestShield AI — predict.py smoke test")
    print("=" * 60)

    for i, sample in enumerate(SAMPLES, 1):
        features = build_feature_vector(sample["payload"])
        result = predict_risk(features)
        print(f"\nSample {i}: {sample['label']}")
        print(f"  Features : {features}")
        print(f"  Result   : {result}")

    print("\n[OK] All samples processed successfully.")

