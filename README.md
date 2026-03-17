# ForestShield AI

Machine learning wildfire risk prediction + fire spread simulation for Ontario using NASA MODIS fire data.

## Overview

**Risk Prediction**: Gradient Boosting Regressor trained on **real fire occurrence labels** from **60,970 Ontario fires** (2018-2024) to predict wildfire risk scores (0-100).

**Fire Spread**: Real-time fire spread rate and perimeter prediction using empirical models + live weather data from Open-Meteo API.
- 6/12/24 hour forecast horizons
- GeoJSON output for map visualization
- NASA FIRMS validation against actual fires

**Training Method**: Uses actual fire locations as HIGH risk, areas near fires as MEDIUM risk, and areas far from fires as LOW risk (not synthetic distance-based labels).

**Performance**: Test RMSE=9.48, R²=0.918 (91.8%) on held-out validation set

## Quick Start

### Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib
```

### Train Model
```bash
cd forestshield-ai
python training/train.py  # ~60 seconds, outputs: models/wildfire_risk_model.pkl
```

### Test Predictions
```bash
python inference/predict.py
```

### Use in Code
```python
from inference.predict import predict_risk

result = predict_risk({
    'temperature': 35.0,
    'humidity': 25.0,
    'lat': 45.4215,
    'lng': -75.6972,
    'nearestFireDistance': 5.0,
    'timestamp': '2024-07-15T15:30:00Z'
})

print(result)
# {'risk_score': 87.0, 'risk_level': 'HIGH', 'confidence': 0.95, 'model_version': '20260317_101456'}
```

## Fire Spread Prediction

**NEW**: Real-time fire spread rate and perimeter prediction with weather integration.

### Install Additional Dependencies
```bash
pip install requests numpy
```

### Complete Fire Assessment (Risk + Spread)
```python
from fire_spread.integrated_system import IntegratedFireSystem

system = IntegratedFireSystem()

# Assess location with real-time weather
assessment = system.assess_location(
    latitude=43.65,
    longitude=-79.38,
    nearest_fire_distance_km=15.0,
    use_realtime_weather=True,
    fuel_type='mixed'  # 'grass', 'brush', 'forest', 'mixed'
)

print(f"Risk: {assessment['risk_assessment']['risk_level']} ({assessment['risk_assessment']['risk_score']}/100)")
print(f"Spread: {assessment['spread_prediction']['spread_rate_kmh']} km/h")
print(f"Intensity: {assessment['spread_prediction']['fire_intensity']}/10")
```

### Simulate Fire Progression
```python
# Simulate 12 hours of fire spread
simulation = system.simulate_fire_progression(
    ignition_lat=43.65,
    ignition_lng=-79.38,
    hours=12,
    fuel_type='forest',
    use_forecast=True
)

print(f"Final Burned Area: {simulation['final_burned_area_km2']} km²")
print(f"Max Intensity: {simulation['max_intensity']}/10")
```

### Features
- ✅ Real-time weather from Open-Meteo API (free, no API key)
- ✅ Fire spread rate calculation (km/h)
- ✅ Elliptical fire perimeter modeling
- ✅ Fire intensity assessment (0-10 scale)
- ✅ Multi-hour fire progression simulation
- ✅ Integrated with ML risk predictions

**See [fire_spread/README.md](fire_spread/README.md) for detailed documentation.**

## Output Contract

```python
{
    'risk_score': float,      # 0-100
    'risk_level': str,        # 'LOW' (0-30) | 'MEDIUM' (>30-60) | 'HIGH' (>60-100)
    'confidence': float,      # 0-1
    'model_version': str
}
```

## Project Structure

```
forestshield-ai/
├── data/                       # NASA MODIS CSVs (2018-2024)
├── models/                     # Trained models (gitignored)
├── utils/features.py           # 11-feature contract
├── training/
│   ├── data_preparation.py    # Data loading & preprocessing
│   ├── label_generator.py     # Generate real labels from fire occurrences
│   ├── train.py               # Model training with real labels
│   └── labeled_data/          # Generated labeled dataset (40,000 samples)
├── inference/predict.py        # Prediction interface
├── fire_spread/               # NEW: Fire spread prediction module
│   ├── __init__.py
│   ├── weather_client.py      # Open-Meteo API integration
│   ├── spread_predictor.py    # Fire spread rate calculation
│   ├── integrated_system.py   # Combined risk + spread system
│   └── README.md              # Fire spread documentation
└── README.md
```

## Data

**Source**: [NASA MODIS Active Fire Detection](https://firms.modaps.eosdis.nasa.gov/)  
**Coverage**: Ontario, Canada (2018-2024)  
**Fires**: 60,970 detections (8.5% of Canada's 720K fires)

**Training Labels**: Created from real fire occurrences:
- **HIGH risk (80-100)**: Actual fire locations (15,000 samples)
- **MEDIUM risk (40-79)**: 5-20km from fires (10,000 samples)
- **LOW risk (0-39)**: >30km from any fire (15,000 samples)

Place CSV files in `data/` directory: `modis_YYYY_Canada.csv`

## Features

**11-feature contract**: 
- **Raw** (5): temperature, humidity, lat, lng, nearestFireDistance
- **Derived** (6): temp_normalized, humidity_inverse, fire_proximity_score, hour_sin/cos, fire_danger_index

See [utils/features.py](utils/features.py) for implementation.

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