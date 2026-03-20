---
title: ForestShield Wildfire Risk Predictor
emoji: 🌲
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: false
---

# ForestShield AI

Machine learning wildfire risk prediction for Ontario using NASA MODIS fire hotspot data (2018–2024).

## Overview

The ForestShield AI module trains a **Gradient Boosting Regressor** on historical MODIS satellite fire data filtered to Ontario, then serves predictions through an inference API consumed by the backend Lambda.

**Risk score** (0–100) and **risk level** (LOW / MEDIUM / HIGH) are computed from three sensor inputs — temperature, humidity, and nearest fire distance — using the same formula as the rule-based backend, so the dashboard requires no changes.

## Risk Levels

| Level  | Score Range | Condition |
|--------|-------------|-----------|
| LOW    | 0 – 30      | Normal conditions |
| MEDIUM | 31 – 60     | Elevated risk |
| HIGH   | 61 – 100    | Dangerous conditions |

## Features

| Feature | Source | Description |
|---|---|---|
| `temperature` | MODIS `bright_t31` − 273.15 | Ambient °C |
| `humidity` | Synthetic (training) / DHT11 (live) | Relative humidity % |
| `lat` / `lng` | MODIS coordinates | GPS location |
| `nearest_fire_dist` | Synthetic (training) / Haversine (live) | Distance to nearest fire km |
| `month` | MODIS `acq_date` | 1–12 |
| `hour` | MODIS `acq_time` | 0–23 UTC |

> `humidity` and `nearest_fire_dist` are synthetically augmented during training (8× per base row) because MODIS does not provide these channels. Real values from the IoT sensor and NASA FIRMS API are used at inference time.

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
cd forestshield-ai
python training/train.py
# Outputs: models/risk_model.joblib + models/model_meta.json
```

### Run inference smoke test
```bash
python inference/predict.py
# Prints HIGH / MEDIUM / LOW predictions for three sample payloads
```

### Use in code
```python
from inference.predict import predict_risk

result = predict_risk({
    "temperature": 35.0,
    "humidity": 25.0,
    "lat": 45.42,
    "lng": -75.70,
    "nearestFireDistance": 5.0,
    "timestamp": "2024-07-15T15:30:00Z"
})
# {"risk_score": 84.5, "risk_level": "HIGH", "model_version": "v2-ontario-gbr"}
```

## Project Structure

```
forestshield-ai/
├── training/
│   └── train.py          # Full training pipeline
├── inference/
│   └── predict.py        # Inference entry point for Lambda
├── utils/
│   └── __init__.py       # Shared feature list, risk thresholds, label formula
├── models/               # Generated — gitignored
│   ├── risk_model.joblib
│   └── model_meta.json
├── data/                 # NASA MODIS CSVs — gitignored
├── requirements.txt
└── .gitignore
```

## Model Details

- **Algorithm**: `GradientBoostingRegressor` (scikit-learn)
- **Hyperparameters**: n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8
- **Region**: Ontario, Canada (lat 41.9–56.9, lon −95.2–−74.3)
- **Data**: NASA MODIS fire hotspot CSVs 2018–2024, augmented 8× per row
- **Label formula**: `0.4 × temp_score + 0.3 × humidity_score + 0.3 × fire_score`
- **Split**: 85% train / 15% test + 5-fold CV
- **Version**: `v2-ontario-gbr`

## Documentation

- **[LABELING_EXPLAINED.md](training/LABELING_EXPLAINED.md)** - Real vs synthetic labels explanation
- **[labeled_training_data.csv](training/labeled_data/labeled_training_data.csv)** - Complete labeled dataset (40,000 samples)

## Integration

### AWS Lambda
```python
import joblib
from utils.features import build_feature_vector

model = joblib.load('models/wildfire_risk_model.pkl')

def lambda_handler(event, context):
    features = build_feature_vector(event).reshape(1, -1)
    risk_score = float(model.predict(features)[0])
    return {
        'risk_score': round(risk_score, 2),
        'risk_level': 'LOW' if risk_score <= 30 else 'MEDIUM' if risk_score <= 60 else 'HIGH'
    }
```

### Vertex AI (Optional)
Model is joblib format, ready for GCS upload and Vertex AI deployment.

## Performance

**Trained with Real Fire Occurrence Labels** (not synthetic distance-based labels)

| Metric | Value | Notes |
|--------|-------|-------|
| Test RMSE | 9.48 | Root mean squared error |
| Test MAE | 7.82 | Mean absolute error |
| Test R² | 0.918 | 91.8% variance explained |
| Training Samples | 29,351 | From 40,000 labeled samples |
| Test Samples | 7,338 | Held-out validation set |
| Label Source | Real Fires | 60,970 Ontario fires (2018-2024) |

**Feature Importance**:
- nearestFireDistance: 53.1%
- fire_proximity_score: 46.3%
- Other features: <1% each

**Key Advantage**: Model learns from actual fire occurrence locations, not synthetic distance formulas.

## Requirements

**Risk Prediction**:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
```

**Fire Spread (additional)**:
```
requests>=2.28.0
```

---

**Model**: GradientBoostingRegressor  
**Training Method**: Real Fire Occurrence Labels  
**Trained**: March 17, 2026  
**Region**: Ontario, Canada  
**Labels**: 15,000 HIGH + 10,000 MEDIUM + 15,000 LOW risk samples  
**Status**: Prototype for real-time wildfire risk assessment and forecasting