# Model Training Guide

Training methodology for ForestShield AI wildfire risk prediction model.

## Overview

**Model**: Gradient Boosting Regressor (scikit-learn)  
**Target**: Wildfire risk score (0-100)  
**Geography**: Ontario, Canada  
**Training Duration**: ~60 seconds  
**Performance**: RMSE=5.43, R²=0.952, 93.4% accuracy

---

## Data Preparation

### Source Data

**NASA MODIS Active Fire Detection**
- Provider: NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/)
- Format: CSV files (2018-2024), fields: latitude, longitude, brightness, acq_date, confidence
- Full Canada: 720,259 fires → Ontario filtered: **60,970 fires** (8.5%)

### Ontario Geographic Filtering

**Bounding Box**:
```python
ONTARIO_BOUNDS = {
    'min_lat': 41.913319, 'max_lat': 56.86895,
    'min_lon': -95.154327, 'max_lon': -74.324722
}
```

**Rationale**: Focused region improves prediction accuracy vs. diverse Canadian climates.

**Implementation**: `training/data_preparation.py` → `load_nasa_data(filter_ontario=True)`

### Synthetic Sample Generation

**Challenge**: NASA data has fire locations, not risk scores.

**Solution**: Generate 20 augmented samples per fire with varied risk levels:
- **High-risk** (70-100): Near fires (±0.5km), hot/dry conditions
- **Medium-risk** (35-65): Moderate distance, mixed weather
- **Low-risk** (5-30): Far from fires, cool/humid conditions

**Augmentation**: Position jitter (±0.01°), weather noise (±5°C temp, ±10% humidity), risk randomization (±10 pts)

**Total**: 1.2M possible samples → **100K used** (optimal speed/accuracy balance)

---

## Feature Engineering

### 11-Feature Contract

**Raw Features** (5):
1. `temperature` - °C
2. `humidity` - 0-100%
3. `lat`, `lng` - Coordinates
4. `nearestFireDistance` - km to nearest fire

**Derived Features** (6):
6. `temp_normalized` - Scaled 0-1 (-40 to 50°C range)
7. `humidity_inverse` - Dryness metric
8. `fire_proximity_score` - Exponential decay: `100 * exp(-distance / 10)`
9. `hour_sin`, `hour_cos` - Cyclical time encoding
10. `fire_danger_index` - Composite: temp × humidity_inverse × proximity

**Key Insight**: Fire proximity dominates (99.1% feature importance).

**Implementation**: `utils/features.py` → `build_feature_vector()`

---

## Training Pipeline

### Steps

1. Load Ontario fires: `load_nasa_data()` → 60,970 fires
2. Generate samples: `generate_training_samples()` → 100K
3. Prepare features: `prepare_features()` → X (11 features), y (risk scores)
4. Train: `train_model()` → GradientBoostingRegressor
5. Evaluate: `evaluate_model()` → RMSE, R², accuracy
6. Save: `models/wildfire_risk_model.pkl`

### Train/Test Split
- 80/20 split (80K train, 20K test)
- `random_state=42` for reproducibility

### Run Training

```bash
cd forestshield-ai
python training/train.py
```

**Expected Output**:
```
[OK] Loaded 60970 Ontario fires
[OK] Generated 100000 training samples
[OK] Model trained successfully
Evaluation: RMSE=5.43, MAE=4.32, R²=0.952, Accuracy=93.4%
[OK] Model saved: models/wildfire_risk_model.pkl
```

---

## Model Architecture

### Algorithm: Gradient Boosting Regressor

**Hyperparameters**:
```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

**Why Gradient Boosting?**
- Handles non-linear fire proximity decay
- Captures feature interactions (temp × humidity × proximity)
- Robust to synthetic data noise
- Fast inference, interpretable

### Output

Risk score (0-100) + classification:
- LOW: 0-30
- MEDIUM: 31-60
- HIGH: 61-100

---

## Performance Metrics

### Test Set Results (20,000 samples)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| RMSE | 5.43 | Avg error ~5 points on 0-100 scale |
| MAE | 4.32 | Median error ~4 points |
| R² | 0.952 | Explains 95.2% of variance |
| Accuracy (±10) | 93.4% | 93.4% within ±10 points |

### Feature Importance

```
fire_proximity_score    99.12%
temp_normalized          0.43%
humidity_inverse         0.22%
other                    0.23%
```

### Example Predictions

| Scenario | Input | Prediction | Level |
|----------|-------|-----------|-------|
| Active fire | 42°C, 18% hum, 0.5km | 97.12 | HIGH |
| Moderate | 28°C, 45% hum, 15km | 47.3 | MEDIUM |
| Safe zone | 22°C, 65% hum, 50km | 22.31 | LOW |

---

## Reproduction Steps

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Download NASA Data

1. Visit https://firms.modaps.eosdis.nasa.gov/
2. Download Canada MODIS CSV (2018-2024)
3. Place in `forestshield-ai/data/`:
   ```
   data/modis_2018_Canada.csv
   data/modis_2019_Canada.csv
   ...
   data/modis_2024_Canada.csv
   ```

### 3. Train Model

```bash
python training/train.py
```

Duration: ~60 seconds

### 4. Test Inference

```bash
python inference/predict.py
```

Expected: 3 test cases (HIGH/MEDIUM/LOW) with predictions

### 5. Inspect Model

```python
import joblib
model_data = joblib.load('models/wildfire_risk_model.pkl')
print(model_data['metadata'])
# {'version': 'v1.0-gradient-boost-nasa', 'ontario_fires': 60970, ...}
```

---

## Configuration

### Adjust Training Samples

**In `train.py`**:
```python
samples = samples[:100000]  # Change to 500000 for more data
```

Trade-offs:
- 100K: 60s training, RMSE=5.43
- 500K: 5min training, RMSE~5.2
- 1.2M: 15min training, RMSE~5.0

### Disable Ontario Filter

**In `data_preparation.py`**:
```python
load_nasa_data(filter_ontario=False)  # Use all 720K Canada fires
```

### Tune Hyperparameters

**In `train.py`**:
```python
GradientBoostingRegressor(
    n_estimators=200,     # More trees
    learning_rate=0.05,   # Lower rate
    max_depth=7           # Deeper trees
)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No CSV files found" | Download NASA data to `data/` directory |
| "No Ontario fires found" | Check `ONTARIO_BOUNDS` or disable filter |
| Low R² (<0.90) | Increase `samples_per_fire` or tune hyperparameters |
| All predictions HIGH/LOW | Verify `fire_proximity_score` in `features.py` |

---

## Next Steps

### Deployment
- AWS Lambda: Load model in Lambda function (see main README)
- Vertex AI: Upload `.pkl` to GCS, deploy as Vertex endpoint

### Model Improvements
1. Real-time weather API (OpenWeatherMap/NOAA)
2. Temporal features (day-of-year, drought index)
3. Vegetation index (NDVI)
4. Regional generalization testing (BC, Alberta)

### References
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- scikit-learn: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
- Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"

---

**Model Version**: v1.0-gradient-boost-nasa  
**Training Region**: Ontario, Canada  
**Training Samples**: 100,000 (from 60,970 fires)  
**Last Updated**: March 2026
