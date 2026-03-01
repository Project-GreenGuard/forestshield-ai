# ForestShield Training Guide

## Overview

Trains a wildfire risk scoring model (0–100 scale) using **3000 synthetic Ontario sensor readings** → RandomForestRegressor → calibrated LOW/MED/HIGH classification (**91.2% accuracy**). Synthetic data uses realistic fire physics: seasonal patterns, temperature-humidity correlation, fire proximity clustering.

---

## 5-Step Training Process

### 1. Generate Synthetic Data (3000 samples)
- **Location:** Ontario (lat: 42-57°, lng: -95 to -74°)
- **Weather:** Temperature (-20 to 42°C, seasonal baseline), humidity (10-95%, inversely correlated)
- **Fire distance:** 30% no fire, 70% exponential distribution (0.5-30+ km)
- **Time:** Hour (0-23), month (1-12, summer bias), day of week (0-6)
- **Target:** Risk score (0-100) from temp (25%) + humidity (25%) + fire proximity (multiplier) + season (10%) + time (5%) + noise (±3)

### 2. Train-Test Split
- 80-20 split: 2400 training, 600 test (seed=42 for reproducibility)

### 3. Feature Engineering (11 features)
**User provides:** temperature, humidity, lat, lng, nearest_fire_distance, timestamp  
**Time features (derived from timestamp):** hour_of_day, month, day_of_week  
**Other derived:** temp_normalized, humidity_inverse, fire_proximity_score  
**Scaling:** StandardScaler normalization

**Exact feature vector order used by the model:**
1. temperature
2. humidity
3. lat
4. lng
5. nearest_fire_distance
6. hour_of_day
7. month
8. temp_normalized
9. humidity_inverse
10. fire_proximity_score
11. day_of_week

This order is stored inside the exported model to ensure correct inference.

### 4. Train RandomForest
```python
RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
```

**Why RandomForest (vs other algorithms)?**
- Handles non-linear relationships (temp × humidity × fire proximity interactions)
- Robust to outliers and extreme weather events
- Provides feature importance analysis (identifies fire_proximity as 51.2% most important)
- Ensemble reduces overfitting (200 trees vote on prediction)
- Works well for classification-style outputs (LOW/MEDIUM/HIGH thresholds)
- Although RandomForest does not strictly require scaling, StandardScaler is applied for preprocessing consistency and stable calibration behavior

**Performance:**
- 3-fold cross-validation: R² = 0.9351
- Test performance: R² = 0.9422, MAE = 2.79

### 5. Calibration
Isotonic regression is fit on (raw RF predictions on training data, true risk_score) and learns a monotonic mapping from model outputs → calibrated 0–100 risk scores. Predictions are clipped to the valid range [0,100] before classification into:
- **LOW:** 0-30  
- **MEDIUM:** 31-60  
- **HIGH:** 61-100  
→ **91.2% classification accuracy on the test set**

---

## Key Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **R²** | 0.9422 | Explains 94.2% of variance |
| **MAE** | 2.79 | Avg error ±2.79 on 0-100 scale |
| **Accuracy** | 91.2% | Correct LOW/MEDIUM/HIGH prediction |

---

## Production Use

**To retrain:**
```bash
python training/train_model.py
```

**To make predictions:**
```python
from inference.predict import build_feature_vector, predict_risk

# Provide sensor data with timestamp (hour_of_day & month auto-derived)
sensor_payload = {
    'temperature': 28.5, 'humidity': 35.2, 'lat': 49.5, 'lng': -93.2,
    'nearestFireDistance': 2.3, 'timestamp': '2026-07-15T14:00:00Z'
}

features = build_feature_vector(sensor_payload)
result = predict_risk(features)
# Returns: {'risk_score': 72.8, 'risk_level': 'HIGH', 'model_version': 'v4-optimized'}
```

**For Vertex AI deployment:**
1. Upload `models/forestshield_v4.joblib` to GCS bucket
2. Deploy to Vertex AI endpoint
3. Backend calls endpoint with sensor data → receives predictions

---

## Validation

- ✅ 91.2% accuracy on synthetic test data (600 samples)
- ✅ 62.5% accuracy on 8 historical Ontario wildfires (real-world validation)
- ✅ HIGH risk detection: 100% (2/2 extreme fire cases)
- ✅ R² 0.67 on real fires, burn area correlation 0.74
- ⏳ Production-ready pending integration testing
````
