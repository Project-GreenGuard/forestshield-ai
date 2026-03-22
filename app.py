"""
ForestShield AI — Google Cloud Run Deployment
==============================================
Flask API server for Cloud Run. Serves predictions via HTTP POST.
Also includes optional Gradio UI for local testing.

Run locally:
    python app.py

Deploy to Cloud Run:
    gcloud builds submit --tag gcr.io/PROJECT_ID/forestshield
    gcloud run deploy forestshield --image gcr.io/PROJECT_ID/forestshield --port 8080
"""

import json
import os
from datetime import datetime, timezone

from flask import Flask, request, jsonify

from inference.predict import build_feature_vector, predict_risk
from utils import FEATURE_COLUMNS

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Spread rate heuristic
# ---------------------------------------------------------------------------

def _estimate_spread_rate(temperature: float, humidity: float, risk_score: float) -> float:
    temp_factor     = min(max(temperature, 0.0), 50.0) / 50.0
    humidity_factor = 1.0 - min(max(humidity, 0.0), 100.0) / 100.0
    risk_factor     = min(max(risk_score,  0.0), 100.0) / 100.0
    raw = 12.0 * (0.35 * temp_factor + 0.35 * humidity_factor + 0.30 * risk_factor)
    return round(min(max(raw, 0.5), 12.0), 2)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    """Health check endpoint for Cloud Run."""
    return jsonify({"status": "ok", "service": "ForestShield"}), 200


# ---------------------------------------------------------------------------
# Prediction API endpoint
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Predict wildfire risk from sensor payload.
    
    Expected JSON:
    {
        "temperature": 30.0,
        "humidity": 50.0,
        "lat": 43.5,
        "lng": -79.7,
        "nearestFireDistance": 50.0,
        "timestamp": "2024-07-20T15:00:00Z"
    }
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Empty request body"}), 400

        # Build features from payload
        features = build_feature_vector(payload)

        # Run prediction
        result = predict_risk(features)
        risk_score = result["risk_score"]
        risk_level = result["risk_level"]
        model_version = result["model_version"]
        spread_rate = _estimate_spread_rate(
            payload.get("temperature", 20.0),
            payload.get("humidity", 50.0),
            risk_score
        )

        return jsonify({
            "risk_score": risk_score,
            "risk_level": risk_level,
            "spread_rate": spread_rate,
            "model_version": model_version,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Gradio UI (optional, for local testing only)
# ---------------------------------------------------------------------------

try:
    import gradio as gr

    def predict(temperature, humidity, lat, lng, nearest_fire_dist, month, hour):
        timestamp = f"2024-{int(month):02d}-15T{int(hour):02d}:00:00Z"

        features = build_feature_vector({
            "temperature":        temperature,
            "humidity":           humidity,
            "lat":                lat,
            "lng":                lng,
            "nearestFireDistance": nearest_fire_dist,
            "timestamp":          timestamp,
        })

        result      = predict_risk(features)
        risk_score  = result["risk_score"]
        risk_level  = result["risk_level"]
        model_ver   = result["model_version"]
        spread_rate = _estimate_spread_rate(temperature, humidity, risk_score)

        level_emoji = {"HIGH": "🔴  HIGH", "MEDIUM": "🟠  MEDIUM", "LOW": "🟢  LOW"}[risk_level]

        return (
            f"{risk_score:.1f} / 100",
            level_emoji,
            f"{spread_rate:.1f} km/h",
            model_ver,
        )

    now = datetime.now(timezone.utc)

    with gr.Blocks(
        title="ForestShield — Wildfire Risk Predictor",
        theme=gr.themes.Base(primary_hue="orange", neutral_hue="gray"),
    ) as demo:

        gr.Markdown(
            """
            # 🌲 ForestShield — Wildfire Risk Predictor
            **GradientBoostingRegressor** trained on NASA MODIS fire data · Ontario, Canada 2018–2024

            Adjust the sensor inputs below and click **Predict Risk** to get the risk score,
            risk level, and estimated fire spread rate.
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌡️ Sensor Readings")
                temperature = gr.Slider(-10, 50,  value=30,   step=0.1, label="Temperature (°C)")
                humidity    = gr.Slider(5,  100,  value=40,   step=0.1, label="Humidity (%)")

                gr.Markdown("### 📍 Location (Ontario)")
                lat = gr.Slider(41.9, 56.9,  value=43.5,  step=0.01, label="Latitude")
                lng = gr.Slider(-95.5, -74.0, value=-79.7, step=0.01, label="Longitude")

                gr.Markdown("### 🔥 Fire Proximity")
                nearest_fire_dist = gr.Slider(0.5, 150, value=50, step=0.5,
                                              label="Nearest Fire Distance (km)  —  100 = no fire detected")

                gr.Markdown("### 🕐 Time")
                month = gr.Slider(1, 12, value=now.month, step=1, label="Month")
                hour  = gr.Slider(0, 23, value=now.hour,  step=1, label="Hour (UTC)")

                predict_btn = gr.Button("Predict Risk", variant="primary")

            with gr.Column():
                gr.Markdown("### 📊 Results")
                out_score   = gr.Textbox(label="Risk Score",       interactive=False)
                out_level   = gr.Textbox(label="Risk Level",       interactive=False)
                out_spread  = gr.Textbox(label="Est. Spread Rate", interactive=False)
                out_version = gr.Textbox(label="Model Version",    interactive=False)

        predict_btn.click(
            fn=predict,
            inputs=[temperature, humidity, lat, lng, nearest_fire_dist, month, hour],
            outputs=[out_score, out_level, out_spread, out_version],
            api_name="predict",
        )

        gr.Examples(
            examples=[
                [42, 15, 48.5, -81.3,  2.0, 7, 15],
                [30, 50, 45.4, -75.7, 50.0, 6, 12],
                [18, 75, 43.7, -79.4, 95.0, 3,  8],
            ],
            inputs=[temperature, humidity, lat, lng, nearest_fire_dist, month, hour],
            label="Example Scenarios",
        )

    if os.getenv("GRADIO_ENABLED", "false").lower() == "true":
        demo.launch(server_name="0.0.0.0", server_port=8080, share=False)

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Cloud Run entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)