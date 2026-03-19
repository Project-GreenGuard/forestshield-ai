"""
Shared utilities for the ForestShield AI module.
Contains the canonical feature column list and helper functions used by
both training (train.py) and inference (predict.py) so the two sides
always stay in sync.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Feature columns  —  must match the fields that actually flow through the
#                      ForestShield backend pipeline end-to-end.
#
# Source of truth:  forestshield-backend/lambda-processing/process_sensor_data.py
#                   calculate_risk_score(temperature, humidity, fire_distance)
#
# IoT sensor publishes  →  Lambda enriches  →  DynamoDB stores:
#   temperature        float  °C    DHT11 sensor ambient reading
#   humidity           float  %     DHT11 sensor relative humidity
#   lat                float  °     GPS latitude
#   lng                float  °     GPS longitude
#   nearestFireDistance float km    Haversine distance to nearest FIRMS fire pixel
#   timestamp          str   ISO-8601 → split into month (1-12) and hour (0-23 UTC)
#
# ORDER MATTERS – must match the column order the model was trained on.
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "temperature",        # °C –  IoT sensor / bright_t31−273.15 proxy from MODIS
    "humidity",           # %  –  IoT sensor / 50.0 imputed for MODIS (not measured)
    "lat",                # °  –  GPS latitude
    "lng",                # °  –  GPS longitude
    "nearest_fire_dist",  # km –  Haversine to nearest FIRMS pixel; 1.0 for MODIS rows
                          #       (every MODIS row IS a confirmed fire pixel)
    "month",              # 1–12  derived from timestamp / acq_date
    "hour",               # 0–23 UTC  derived from timestamp / acq_time
]


# ---------------------------------------------------------------------------
# Risk level thresholds  (aligned with frontend DataPanel.js)
# ---------------------------------------------------------------------------
#   LOW    :  0 – 30
#   MEDIUM : 31 – 60    (frontend uses > 30, so we use >= 31 here)
#   HIGH   : 61 – 100   (frontend uses > 60, so we use >= 61 here)
_THRESHOLDS = {"HIGH": 61.0, "MEDIUM": 31.0}


def compute_risk_level(risk_score: float) -> str:
    """Map a numeric risk_score (0–100) to 'LOW', 'MEDIUM', or 'HIGH'.
    Bands mirror the frontend DataPanel.js thresholds:
        HIGH    >= 61
        MEDIUM  >= 31
        LOW      < 31
    """
    if risk_score >= _THRESHOLDS["HIGH"]:
        return "HIGH"
    if risk_score >= _THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def compute_risk_label(row: Any) -> float:
    """
    Compute a risk_score (0–100) label from a preprocessed DataFrame row.
    Mirrors the backend formula in process_sensor_data.calculate_risk_score()
    exactly, so the ML model learns to reproduce (and later extend) it:
        risk_score = 0.4 × temp_score
                   + 0.3 × humidity_score
                   + 0.3 × fire_score
    Column expectations (after preprocessing):
        temperature       float  °C
        humidity          float  %
        nearest_fire_dist float  km  (use 1.0 for MODIS fire pixels)
    """
    # Temperature component (0–50 °C normalisation)
    temp_score = min(max(float(row["temperature"]), 0.0) / 50.0, 1.0) * 100.0

    # Humidity component  (lower humidity → higher risk)
    humidity_score = (1.0 - min(float(row["humidity"]) / 100.0, 1.0)) * 100.0

    # Fire proximity component  (closer fire → higher risk, 0–100 km normalised)
    fire_dist = float(row["nearest_fire_dist"])
    if fire_dist > 0:
        fire_score = max(0.0, (100.0 - min(fire_dist, 100.0)) / 100.0) * 100.0
    else:
        fire_score = 100.0   # distance = 0 → maximum fire risk

    risk_score = 0.4 * temp_score + 0.3 * humidity_score + 0.3 * fire_score
    return round(min(risk_score, 100.0), 2)
