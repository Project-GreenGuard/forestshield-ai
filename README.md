# ForestShield AI 🌲🔥

AI-powered wildfire risk prediction system for Ontario using NASA MODIS fire detections and machine learning.

---

## 🚀 Overview

ForestShield AI predicts wildfire risk in real time using environmental sensor data and a trained machine learning model.

The system:
- Uses **NASA MODIS wildfire detections (2018–2024)** as real-world fire data
- Trains a **Gradient Boosting Regressor**
- Outputs:
  - `risk_score` (0–100)
  - `risk_level` (LOW / MEDIUM / HIGH)
  - `spread_rate` (km/h)
  - `risk_factors` (key drivers of risk)
  - `recommended_action` (response guidance)
  - `explanation` (human-readable reasoning)

👉 This is a **data-driven AI model**, not a rule-based system.

---

## 🧠 Model Approach (AI Design)

### Training Strategy

The model is trained using a **hybrid dataset**:

- 🔴 **High-risk samples (REAL DATA)**
  - Derived directly from NASA wildfire detections
- 🟠 **Medium-risk samples (GENERATED)**
  - Simulated environmental conditions
- 🟢 **Low-risk samples (GENERATED)**
  - Stable/no-fire scenarios

This allows the model to:
- Learn from **real fire events**
- Generalize across **different environmental conditions**

📌 Unlike older versions, this model does **not learn a fixed formula** — it learns patterns from data.

---

## 📊 Features Used

| Feature | Description |
|--------|------------|
| `temperature` | Ambient temperature (°C) |
| `humidity` | Relative humidity (%) |
| `lat` / `lng` | GPS location |
| `nearest_fire_dist` | Distance to nearest fire (km) |
| `month` | Time-based feature |
| `hour` | Time-based feature |

---

## 📈 Output (AI Prediction + Insights)

Example output:

```json
{
  "risk_score": 84.2,
  "risk_level": "HIGH",
  "spread_rate": 8.5,
  "model_version": "v3-ontario-gbr-firms",
  "risk_factors": [
    "high temperature",
    "very low humidity",
    "active fire detected nearby"
  ],
  "recommended_action": "Dispatch emergency responders, monitor evacuation zones, and issue high-priority alerts.",
  "explanation": "Predicted wildfire risk is driven by high temperature, very low humidity, and nearby fire activity."
}