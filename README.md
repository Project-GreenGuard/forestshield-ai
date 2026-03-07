# ForestShield AI

Machine learning wildfire risk prediction for Ontario using NASA MODIS fire data.

## Overview

**Gradient Boosting Regressor** trained on **60,970 Ontario fires** (2018-2024) to predict wildfire risk scores (0-100).

**Performance**: RMSE=5.43, R²=0.952, 93.4% accuracy

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
# {'risk_score': 85.3, 'risk_level': 'HIGH', 'confidence': 0.95, 'model_version': 'v1.0-gradient-boost-nasa'}
```

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
├── data/                       # NASA MODIS CSVs (gitignored)
├── models/                     # Trained models (gitignored)
├── utils/features.py           # 11-feature contract
├── training/
│   ├── data_preparation.py    # Data loading & synthetic samples
│   └── train.py               # Model training & evaluation
├── inference/predict.py        # Prediction interface
└── docs/
    └── MODEL_TRAINING.md      # Complete training guide
```

## Data

**Source**: [NASA MODIS Active Fire Detection](https://firms.modaps.eosdis.nasa.gov/)  
**Coverage**: Ontario, Canada (2018-2024)  
**Fires**: 60,970 detections (8.5% of Canada's 720K fires)

Place CSV files in `data/` directory: `modis_YYYY_Canada.csv`

## Features

**11-feature contract**: 
- **Raw** (5): temperature, humidity, lat, lng, nearestFireDistance
- **Derived** (6): temp_normalized, humidity_inverse, fire_proximity_score, hour_sin/cos, fire_danger_index

See [utils/features.py](utils/features.py) for implementation.

## Documentation

- **[MODEL_TRAINING.md](docs/MODEL_TRAINING.md)** - Complete training guide and methodology
- **[AI_PREDICTION_AND_TRAINING_SPEC.md](../forestshield/docs/AI_PREDICTION_AND_TRAINING_SPEC.md)** - Original specification

## Integration

### AWS Lambda
```python
import joblib
from utils.features import build_feature_vector

model = joblib.load('models/wildfire_risk_model.pkl')['model']

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

| Metric | Value | Status |
|--------|-------|--------|
| RMSE | 5.43 | Excellent |
| MAE | 4.32 | Excellent |
| R² Score | 0.952 | Very Strong |
| Accuracy (±10) | 93.4% | Excellent |

**Feature Importance**:
- Fire proximity: ~99% (dominant factor)
- Temperature/humidity: ~1% (minor adjustments)

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
```

---

**Version**: v1.0-gradient-boost-nasa  
**Trained**: March 6, 2026  
**Region**: Ontario, Canada  
**Status**: Production Ready

For detailed training methodology, see [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md).