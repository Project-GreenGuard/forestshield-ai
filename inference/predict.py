"""
Inference helpers for the ForestShield wildfire risk model.

These functions define the shape of what the backend / Lambdas
will eventually call, without committing to a specific model
implementation yet.
"""

from __future__ import annotations

from typing import Dict, Any


def build_feature_vector(sensor_payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a raw sensor payload into a flat feature dict.

    Expected keys (aligned with backend / DynamoDB):
    - temperature (°C)
    - humidity (%)
    - lat, lng
    - nearestFireDistance (km, optional)
    - timestamp (ISO8601, optional for time-based features)

    Notes for Samira:
    - Keep this in sync with the shared AI spec in the main docs.
    - You can evolve this over time (e.g. add derived features)
      as long as the contract to the model stays clear.
    """
    # This is intentionally left open-ended for now.
    raise NotImplementedError("Implement feature construction here.")


def predict_risk(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Run a single prediction and return a structured result.

    Output contract (what the rest of the system expects):
    - risk_score: float between 0 and 100
    - risk_level: string in {\"LOW\", \"MEDIUM\", \"HIGH\"}
    - model_version: free-form string, e.g. \"v1-rule-approx\"

    Notes for Samira:
    - Initially, this can just call a local model (e.g. joblib).
    - Later, this function can become a thin wrapper around
      Vertex AI online prediction while keeping the same output
      shape so callers don't have to change.
    """
    raise NotImplementedError("Implement model inference here.")

