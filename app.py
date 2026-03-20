"""
ForestShield AI — HuggingFace Gradio Demo
==========================================
Runs as a Gradio Space on HuggingFace using the trained model in models/.
Imports inference helpers directly from the forestshield-ai package.

Run locally:
    cd forestshield-ai
    python app.py
"""

from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np

from inference.predict import build_feature_vector, predict_risk
from utils import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Spread rate heuristic  (mirrors process_sensor_data.py)
# ---------------------------------------------------------------------------

def _estimate_spread_rate(temperature: float, humidity: float, risk_score: float) -> float:
    temp_factor     = min(max(temperature, 0.0), 50.0) / 50.0
    humidity_factor = 1.0 - min(max(humidity, 0.0), 100.0) / 100.0
    risk_factor     = min(max(risk_score,  0.0), 100.0) / 100.0
    raw = 12.0 * (0.35 * temp_factor + 0.35 * humidity_factor + 0.30 * risk_factor)
    return round(min(max(raw, 0.5), 12.0), 2)


# ---------------------------------------------------------------------------
# Prediction function called by Gradio
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

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

            gr.Markdown(
                """
                ---
                #### Risk Level Bands
                | Level      | Score   | Meaning |
                |------------|---------|---------|
                | 🟢 LOW     | 0 – 30  | Normal conditions |
                | 🟠 MEDIUM  | 31 – 60 | Elevated risk — monitor closely |
                | 🔴 HIGH    | 61 – 100| Dangerous — take immediate action |

                #### Spread Rate
                Heuristic estimate based on temperature, humidity, and risk score.
                Range: **0.5 – 12.0 km/h**.
                """
            )

    predict_btn.click(
        fn=predict,
        inputs=[temperature, humidity, lat, lng, nearest_fire_dist, month, hour],
        outputs=[out_score, out_level, out_spread, out_version],
        api_name="predict",
    )

    gr.Examples(
        examples=[
            [42, 15, 48.5, -81.3,  2.0, 7, 15],   # HIGH
            [30, 50, 45.4, -75.7, 50.0, 6, 12],   # MEDIUM
            [18, 75, 43.7, -79.4, 95.0, 3,  8],   # LOW
        ],
        inputs=[temperature, humidity, lat, lng, nearest_fire_dist, month, hour],
        label="Example Scenarios",
    )

    gr.Markdown(
        """
        ---
        **Model:** GradientBoostingRegressor · n_estimators=300 · lr=0.05 · max_depth=5 · subsample=0.8
        **Training data:** NASA MODIS C6.1 NRT (2018–2024) · Ontario · 8× synthetic augmentation
        **Features:** `temperature` · `humidity` · `lat` · `lng` · `nearest_fire_dist` · `month` · `hour`
        """
    )
demo.queue()

if __name__ == "__main__":
    demo.launch()
