"""
Shared utilities for the ForestShield AI code.
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Risk scoring helpers
# ---------------------------------------------------------------------------

def risk_level_from_score(score: float) -> str:
    """
    Map a numeric risk score (0–100) to a human-readable label.

    Thresholds (from AI spec):
    - LOW    :   0 – 30
    - MEDIUM : > 30 – 60
    - HIGH   : > 60 – 100
    """
    if score > 60:
        return "HIGH"
    if score > 30:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Rule-based baseline (mirrors the backend Lambda so we can generate
# synthetic training labels without a live DynamoDB export).
# ---------------------------------------------------------------------------

def rule_based_risk_score(
    temperature: float,
    humidity: float,
    fire_distance: Optional[float] = None,
) -> float:
    """
    Deterministic risk score that replicates the backend Lambda formula.

    Parameters
    ----------
    temperature : °C (expected range 0–50)
    humidity    : % (expected range 0–100)
    fire_distance : km to nearest active fire; ``None`` or negative means
                    no known fire.

    Returns
    -------
    float in [0, 100]
    """
    w1, w2, w3 = 0.4, 0.3, 0.3

    temp_score = min(temperature / 50.0, 1.0) * 100.0
    humidity_score = (1.0 - min(humidity / 100.0, 1.0)) * 100.0

    if fire_distance is not None and fire_distance > 0:
        fire_score = max(0.0, (100.0 - min(fire_distance, 100.0)) / 100.0) * 100.0
    else:
        fire_score = 0.0

    return round(min(w1 * temp_score + w2 * humidity_score + w3 * fire_score, 100.0), 2)


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = (math.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a)) * 6371.0

