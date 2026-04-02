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

Output contract
-----------------------------------------
  risk_score          float   0–100
  risk_level          str     "LOW" | "MEDIUM" | "HIGH"
  model_version       str     e.g. "v3-ontario-gbr-firms"
  risk_factors        list[str]
  recommended_action  str
  explanation         str

Quick smoke-test (run from forestshield-ai/ root):
    python inference/predict.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from utils import FEATURE_COLUMNS, compute_risk_level

# ---------------------------------------------------------------------------
# Model loading (lazy singleton — loaded once on first call)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE.parent / "models" / "risk_model.joblib"
_META_PATH = _HERE.parent / "models" / "model_meta.json"

_model = None
_model_version = "v3-ontario-gbr-firms"


def _load_model():
    global _model, _model_version
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {_MODEL_PATH}.\n"
                "Run: python training/train.py"
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

    fire_dist = sensor_payload.get("nearestFireDistance")
    if fire_dist is None or float(fire_dist) < 0:
        fire_dist = 100.0

    temperature = min(max(float(sensor_payload.get("temperature", 20.0)), 0.0), 50.0)
    humidity = min(max(float(sensor_payload.get("humidity", 50.0)), 20.0), 90.0)

    return {
        "temperature": temperature,
        "humidity": humidity,
        "lat": float(sensor_payload.get("lat", 0.0)),
        "lng": float(sensor_payload.get("lng", 0.0)),
        "nearest_fire_dist": float(fire_dist),
        "month": float(ts.month),
        "hour": float(ts.hour),
    }


def estimate_spread_rate(features: Dict[str, float], risk_score: float) -> float:
    """
    Estimate fire spread rate (km/h) using a simple environmental heuristic.
    Higher temperature, lower humidity, closer fire proximity, and higher
    predicted risk all increase spread rate.
    """
    temp_factor = min(max(features["temperature"], 0.0), 50.0) / 50.0
    humidity_factor = 1.0 - min(max(features["humidity"], 20.0), 90.0) / 100.0
    risk_factor = min(max(risk_score, 0.0), 100.0) / 100.0

    distance = max(features["nearest_fire_dist"], 0.5)
    distance_factor = 1.0 - min(distance, 100.0) / 100.0

    raw = 12.0 * (
        0.30 * temp_factor
        + 0.25 * humidity_factor
        + 0.25 * risk_factor
        + 0.20 * distance_factor
    )

    return round(min(max(raw, 0.5), 12.0), 2)


def generate_ai_insights(
    features: Dict[str, float],
    risk_score: float
) -> Tuple[List[str], str, str]:
    """
    Generate simple AI-style decision-support insights based on model input
    features and predicted risk score.
    """
    reasons: List[str] = []

    if features["temperature"] >= 35:
        reasons.append("high temperature")
    elif features["temperature"] >= 28:
        reasons.append("elevated temperature")

    if features["humidity"] <= 30:
        reasons.append("very low humidity")
    elif features["humidity"] <= 60:
        reasons.append("moderate humidity")
    else:
        reasons.append("higher humidity conditions")

    if features["nearest_fire_dist"] <= 10:
        reasons.append("active fire detected nearby")
    elif features["nearest_fire_dist"] <= 50:
        reasons.append("fire activity within operational range")

    if risk_score >= 75:
        reasons.append("strong model confidence in severe wildfire conditions")
    elif risk_score >= 45 and len(reasons) <= 2:
        reasons.append("moderate environmental risk conditions")

    if risk_score >= 61:
        action = "Dispatch emergency responders, monitor evacuation zones, and issue high-priority alerts."
    elif risk_score >= 31:
        action = "Increase monitoring, prepare response teams, and watch for changing fire conditions."
    else:
        action = "Maintain routine monitoring and continue collecting sensor updates."

    explanation = f"Predicted wildfire risk is driven by {', '.join(reasons)}."

    return reasons, action, explanation


def predict_risk(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Run a single risk prediction and return a structured result.
    """
    model = _load_model()

    x = np.array([[features[col] for col in FEATURE_COLUMNS]])
    raw_score = float(model.predict(x)[0])
    risk_score = round(min(max(raw_score, 0.0), 100.0), 2)

    risk_level = compute_risk_level(risk_score)
    spread_rate = estimate_spread_rate(features, risk_score)
    reasons, action, explanation = generate_ai_insights(features, risk_score)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "spread_rate": spread_rate,
        "model_version": _model_version,
        "risk_factors": reasons,
        "recommended_action": action,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    high_risk = {
        "temperature": 42.0,
        "humidity": 22.0,
        "lat": 48.50,
        "lng": -81.33,
        "nearestFireDistance": 2.0,
        "timestamp": "2024-07-20T15:00:00Z",
    }

    medium_risk = {
        "temperature": 30.0,
        "humidity": 50.0,
        "lat": 45.40,
        "lng": -75.69,
        "nearestFireDistance": 30.0,
        "timestamp": "2024-06-01T12:00:00Z",
    }

    low_risk = {
        "temperature": 18.0,
        "humidity": 75.0,
        "lat": 43.70,
        "lng": -79.42,
        "nearestFireDistance": 95.0,
        "timestamp": "2024-03-10T08:00:00Z",
    }

    for label, payload in [
        ("HIGH-risk sample", high_risk),
        ("MEDIUM-risk sample", medium_risk),
        ("LOW-risk sample", low_risk),
    ]:
        print(f"\n── {label} ──")
        print("Payload :", json.dumps(payload))
        features = build_feature_vector(payload)
        print("Features:", features)
        result = predict_risk(features)
        print("Result  :", json.dumps(result, indent=2))