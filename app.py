"""
ForestShield AI — Google Cloud Run Deployment
==============================================
Flask API server for Cloud Run. Serves predictions via HTTP POST.
"""

import os
from flask import Flask, request, jsonify

from inference.predict import build_feature_vector, predict_risk

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ForestShield"}), 200


# ---------------------------------------------------------------------------
# Prediction API endpoint
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Predict wildfire risk from sensor payload.
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Empty request body"}), 400

        # Build features
        features = build_feature_vector(payload)

        # Run model
        result = predict_risk(features)

        # Return FULL AI response
        return jsonify({
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "spread_rate": result["spread_rate"],
            "model_version": result["model_version"],
            "risk_factors": result.get("risk_factors", []),
            "recommended_action": result.get("recommended_action"),
            "explanation": result.get("explanation"),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Cloud Run entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)