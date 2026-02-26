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

# ---------------------------------------------------------------------------
# Optional lazy-loaded model
# ---------------------------------------------------------------------------

_MODEL = None          # cached model object (or False if unavailable)
_MODEL_PATH_ENV = "FORESTSHIELD_MODEL_PATH"
_FALLBACK_VERSION = "v1-rule-based"


def _load_model() -> Optional[Any]:
    """
    Try to load a joblib model from ``FORESTSHIELD_MODEL_PATH``.
    Returns the model object on success, ``None`` otherwise.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL if _MODEL is not False else None

    model_path = os.getenv(_MODEL_PATH_ENV)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    dict with keys matching ``FEATURE_COLUMNS``.
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

    return {
        "temperature": temperature,
        "humidity": humidity,
        "lat": lat,
        "lng": lng,
        "nearest_fire_distance": nearest_fire_distance,
        "hour_of_day": hour_of_day,
        "month": month,
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
       and the file exists), use it.
    2. Otherwise fall back to the deterministic rule-based formula so the
       system works end-to-end without a trained artefact.
    """
    model = _load_model()

    if model is not None:
        # Build a 2-D array in the expected column order.
        import numpy as np  # type: ignore

        x = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
        raw_score = float(model.predict(x)[0])
        risk_score = round(min(max(raw_score, 0.0), 100.0), 2)
        model_version = getattr(model, "forestshield_version", "v2-ml")
    else:
        # Rule-based fallback.
        fire_dist = features.get("nearest_fire_distance", -1.0)
        risk_score = rule_based_risk_score(
            temperature=features.get("temperature", 0.0),
            humidity=features.get("humidity", 0.0),
            fire_distance=fire_dist if fire_dist >= 0 else None,
        )
        model_version = _FALLBACK_VERSION

    return {
        "risk_score": risk_score,
        "risk_level": risk_level_from_score(risk_score),
        "model_version": model_version,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test  —  python inference/predict.py
# ---------------------------------------------------------------------------

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

