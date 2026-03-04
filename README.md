# ForestShield AI

Machine learning for wildfire risk prediction using NASA MODIS fire data (2018-2024).

## Overview

**Gradient Boosting Regressor** trained on **720,259 real fires** to predict wildfire risk scores (0-100).

**Performance**: RMSE=5.44, R²=0.952, 93.2% accuracy

## Quick Start

### Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib
```

### Train Model
```bash
cd forestshield-ai
python training/train.py  # ~90 seconds, outputs: models/wildfire_risk_model.pkl
```

### Test Predictions
```bash
python inference/predict.py
```

### Use in Code
```python
from inference.predict import predict_risk

result = predict_risk({
    'temperature': 38.0,
    'humidity': 22.0,
    'lat': 49.28,
    'lng': -123.12,
    'nearestFireDistance': 2.5,
    'timestamp': '2024-07-15T15:30:00Z'
})

print(result)
# {'risk_score': 94.45, 'risk_level': 'HIGH', 'confidence': 0.95, 'model_version': 'v1.0-gradient-boost-nasa'}
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
    ├── MODEL_TRAINING.md      # Complete training documentation
    └── PHASE1_CHECKLIST.md    # Spec compliance checklist
```

## Data

**Source**: [NASA MODIS Active Fire Detection](https://firms.modaps.eosdis.nasa.gov/)  
**Coverage**: Canada, 2018-2024  
**Fires**: 720,259 detections

Place CSV files in `data/` directory: `modis_YYYY_Canada.csv`

## Features

**11-feature contract**: 5 raw (temp, humidity, lat, lng, distance) + 6 derived (normalized temp, inverted humidity, proximity score, cyclical time, danger index)

See [utils/features.py](utils/features.py) for details.

## Documentation

- **[MODEL_TRAINING.md](docs/MODEL_TRAINING.md)** - Complete training documentation, methodology, performance metrics
- **[PHASE1_CHECKLIST.md](docs/PHASE1_CHECKLIST.md)** - AI spec compliance verification
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

---

**Version**: v1.0-gradient-boost-nasa  
**Trained**: March 3, 2026  
**Status**: Production Ready ✅

For detailed training methodology and performance analysis, see [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md).
