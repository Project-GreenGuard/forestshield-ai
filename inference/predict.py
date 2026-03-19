"""
Inference helpers for the ForestShield wildfire risk model.
These are the entry points called by the backend Lambdas / API handler.
Features used (match training exactly)
----------------------------------------
  temperature       float  °C       IoT sensor ambient temperature
  humidity          float  %        IoT sensor relative humidity
  lat               float  degrees  GPS latitude
  lng               float  degrees  GPS longitude
  nearest_fire_dist float  km       Haversine distance to nearest FIRMS fire pixel
                                    (default 100.0 when no fires are detected)
  month             int    1–12     derived from payload timestamp (UTC)
  hour              int    0–23     derived from payload timestamp (UTC)
Output contract (unchanged from AI spec)
-----------------------------------------
  risk_score    float   0–100
  risk_level    str     "LOW" | "MEDIUM" | "HIGH"
  model_version str     e.g. "v2-ontario-gbr"
Quick smoke-test (run from forestshield-ai/ root):
    python inference/predict.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from utils import FEATURE_COLUMNS, compute_risk_level

# ---------------------------------------------------------------------------
# Model loading  (lazy singleton — loaded once on first call)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE.parent / "models" / "risk_model.joblib"
_META_PATH = _HERE.parent / "models" / "model_meta.json"

_model = None
_model_version = "v2-ontario-gbr"


def _load_model():
    global _model, _model_version
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {_MODEL_PATH}.\n"
                "Run:  python training/train.py"
            )
        _model = joblib.load(_MODEL_PATH)
        if _META_PATH.exists():
            meta = json.loads(_META_PATH.read_text())
            _model_version = meta.get("model_version", _model_version)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_vector(sensor_payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a raw backend/Lambda sensor payload into the flat feature dict
    expected by predict_risk().
    Accepted payload keys
    ----------------------
    temperature          float  °C       IoT sensor ambient temperature
    humidity             float  %        IoT sensor relative humidity
    lat                  float  degrees  GPS latitude
    lng                  float  degrees  GPS longitude
    nearestFireDistance  float  km       Haversine distance to nearest FIRMS pixel;
                                         use 100.0 (max) when no fires detected
    timestamp            str    ISO-8601 e.g. "2024-07-15T14:30:00Z"
    """
    ts_raw = sensor_payload.get("timestamp")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    # nearestFireDistance: default to 100 km (no fire nearby) when absent
    fire_dist = sensor_payload.get("nearestFireDistance")
    if fire_dist is None or fire_dist < 0:
        fire_dist = 100.0   # matches backend default behaviour

    return {
        "temperature":       float(sensor_payload.get("temperature", 20.0)),
        "humidity":          float(sensor_payload.get("humidity", 50.0)),
        "lat":               float(sensor_payload.get("lat", 0.0)),
        "lng":               float(sensor_payload.get("lng", 0.0)),
        "nearest_fire_dist": float(fire_dist),
        "month":             float(ts.month),
        "hour":              float(ts.hour),
    }


def predict_risk(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Run a single risk prediction and return a structured result.
    Output contract (same as AI spec):
        risk_score    float  0–100
        risk_level    str    "LOW" | "MEDIUM" | "HIGH"
        model_version str    e.g. "v2-ontario-gbr"
    """
    model = _load_model()

    # Build input array in the exact column order used at training time
    x = np.array([[features[col] for col in FEATURE_COLUMNS]])
    raw_score = float(model.predict(x)[0])
    risk_score = round(min(max(raw_score, 0.0), 100.0), 2)

    return {
        "risk_score":    risk_score,
        "risk_level":    compute_risk_level(risk_score),
        "model_version": _model_version,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # High-risk scenario: very hot, low humidity, fire right next to sensor
    high_risk = {
        "temperature": 42.0,
        "humidity": 15.0,
        "lat": 48.50,
        "lng": -81.33,
        "nearestFireDistance": 2.0,
        "timestamp": "2024-07-20T15:00:00Z",
    }

    # Low-risk scenario: mild temperature, high humidity, no fire nearby
    low_risk = {
        "temperature": 18.0,
        "humidity": 75.0,
        "lat": 43.70,
        "lng": -79.42,
        "nearestFireDistance": 95.0,
        "timestamp": "2024-03-10T08:00:00Z",
    }

    for label, payload in [("HIGH-risk sample", high_risk), ("LOW-risk sample", low_risk)]:
        print(f"\n── {label} ──")
        print("Payload :", json.dumps(payload))
        features = build_feature_vector(payload)
        print("Features:", features)
        result = predict_risk(features)
        print("Result  :", json.dumps(result, indent=2))