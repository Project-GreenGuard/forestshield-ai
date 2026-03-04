# ForestShield AI - Model Training Documentation

## Overview

Wildfire risk prediction model trained on **720,259 NASA MODIS fire detections** (2018-2024) using Gradient Boosting Regressor. Predicts risk scores 0-100 with 95.2% R² accuracy.

**Reference**: This implementation follows `../../forestshield/docs/AI_PREDICTION_AND_TRAINING_SPEC.md` Phase 1 approach.

## Training Data

**Source**: NASA MODIS Active Fire Detection (Canada, 2018-2024)
- **Total Fires**: 720,259 across 7 years
- **Key Years**: 2023 (324K fires - worst season), 2021 (97K), 2024 (148K)
- **Fire Properties**: Real lat/lng, brightness temp, Fire Radiative Power (FRP), confidence scores

**Data Structure**: 7 CSV files (`modis_YYYY_Canada.csv`) with satellite-measured fire locations, intensity, and timing.

## Training Strategy (Phase 1)

**Approach**: Generate synthetic sensor observations around **real fire locations** with rule-based risk labels.

**Why Synthetic Sensors?**  
NASA MODIS only provides fire coordinates—not ground sensor readings at various distances. We generate "what would a sensor read at 2km, 25km, 85km from this fire?" to create training examples across the full risk spectrum.

**Sensor Generation** (20-30 per fire):
- Distance: Exponential distribution (0.5-100 km, more samples near fire)
- Temperature: Decreases with distance from fire (`base_temp = 20 + (brightness-300)/20 - distance*0.15`)
- Humidity: Increases with distance (`base_humidity = 60 - frp/30 + distance*0.2`)

**Risk Labels** (rule-based):
```python
if distance < 5km: risk=85  | < 15km: risk=65 | < 30km: risk=40 | < 50km: risk=25 | else: risk=15
risk += fire_intensity_adjustment + noise  # Creates realistic gradient
```

**Result**: 100,000 diverse training samples with environmental conditions correlated to real fire patterns.

## Features (11-Feature Contract)

**5 Raw Features**: temperature, humidity, lat, lng, nearestFireDistance  
**6 Derived Features**: temp_normalized, humidity_inverse, fire_proximity_score (exp decay), hour_sin/cos (time), fire_danger_index (composite)

**Feature Importance**: Fire proximity dominates (99% combined: proximity_score 70.5%, distance 28.6%)

## Model: Gradient Boosting Regressor

**Why Gradient Boosting?**
1. **Spec-recommended**: AI prediction spec explicitly suggests GB Trees for Phase 1
2. **Tabular data excellence**: Outperforms neural networks on structured features
3. **Non-linear patterns**: Captures exponential distance decay, temp-humidity interactions
4. **Interpretable**: Feature importance reveals fire proximity drives predictions (99%)
5. **Production-ready**: Fast inference (ms), small size (2-5 MB), stable predictions
6. **Phase 1 fit**: Perfect for approximating rule-based systems

**Hyperparameters**: 100 estimators, 0.1 learning rate, max_depth=5, subsample=0.8  
**Training**: 80K samples (80%), Validation: 20K (20%)

## Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **RMSE** | 5.44 | < 8.0 | ✅ Excellent |
| **MAE** | 4.34 | < 6.0 | ✅ Excellent |
| **R² Score** | 0.952 | > 0.85 | ✅ Very Strong |
| **Accuracy (±10)** | 93.2% | > 80% | ✅ Excellent |

**Interpretation**: Model explains 95.2% of risk variance with average error of 5.44 points (0-100 scale).

## Output Contract

```python
{
    'risk_score': 94.45,           # Float 0-100
    'risk_level': 'HIGH',          # LOW (0-30) | MEDIUM (>30-60) | HIGH (>60-100)
    'confidence': 0.95,            # Float 0-1
    'model_version': 'v1.0-gradient-boost-nasa'
}
```

## Usage

**Train Model**:
```bash
cd forestshield-ai
python training/train.py  # Outputs: models/wildfire_risk_model.pkl (~90 sec)
```

**Test Inference**:
```bash
python inference/predict.py  # Tests HIGH/MEDIUM/LOW scenarios
```

**Python API**:
```python
from inference.predict import predict_risk

result = predict_risk({
    'temperature': 38.0, 'humidity': 22.0, 'lat': 49.28, 'lng': -123.12,
    'nearestFireDistance': 2.5, 'timestamp': '2024-07-15T15:30:00Z'
})
# → {'risk_score': 94.45, 'risk_level': 'HIGH', 'confidence': 0.95}
```

**Example Predictions**:
- **HIGH RISK**: 38°C, 22% humidity, 2.5km from fire → **94.45/100** (95% confidence)
- **MEDIUM RISK**: 28°C, 45% humidity, 25km from fire → **44.78/100** (85% confidence)  
- **LOW RISK**: 22°C, 65% humidity, 85km from fire → **17.92/100** (70% confidence)

## Integration

**AWS Lambda** (load model once, use in handler):
```python
import joblib
from utils.features import build_feature_vector

model = joblib.load('models/wildfire_risk_model.pkl')['model']

def lambda_handler(event, context):
    features = build_feature_vector(event).reshape(1, -1)
    risk_score = float(model.predict(features)[0])
    return {'risk_score': round(risk_score, 2), 
            'risk_level': 'LOW' if risk_score <= 30 else 'MEDIUM' if risk_score <= 60 else 'HIGH'}
```

**Vertex AI** (optional): Model is joblib format ready for GCS upload and Vertex AI endpoint deployment.

## Files

```
forestshield-ai/
├── data/                           # NASA MODIS CSVs (2018-2024, gitignored)
├── models/                         # wildfire_risk_model.pkl (gitignored)
├── utils/features.py               # 11-feature contract
├── training/
│   ├── data_preparation.py        # Data loading & synthetic sample generation
│   └── train.py                   # Model training & evaluation
├── inference/predict.py            # Prediction interface
└── docs/MODEL_TRAINING.md          # This doc
```

**Code Organization**: Training pipeline is split into two modules for better separation of concerns - `data_preparation.py` handles NASA data loading and synthetic sample generation, while `train.py` focuses on model training, evaluation, and saving.

## Future Enhancements

**Phase 2**: Retrain with real fire spread outcomes when available  
**Additional Features**: Weather forecasts, vegetation indices (NDVI), historical patterns  
**Models**: Ensemble methods, XGBoost, online learning

---

**Version**: v1.0-gradient-boost-nasa | **Trained**: March 3, 2026 | **Status**: Production Ready ✅  
**Performance**: RMSE=5.44, R²=0.952 | **Data**: 720K NASA fires (2018-2024) | **References**: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
