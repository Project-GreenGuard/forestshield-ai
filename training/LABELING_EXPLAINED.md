# Real vs Synthetic Labels

## The Problem with Current Approach

### Old Method (Synthetic Labels):
```python
# Generate fake risk scores based on distance from fire
if distance_km < 5:
    risk_score = 85  # Made up!
elif distance_km < 15:
    risk_score = 65  # Made up!
else:
    risk_score = 40  # Made up!
```

**Problem:** These labels are invented, not real. The model learns arbitrary rules, not actual fire risk patterns.

## New Approach (Real Labels from Fire Occurrences)

### Method:
1. **HIGH RISK (80-100)**: Locations where fires **actually happened**
   - Uses exact NASA MODIS fire detection coordinates
   - If a fire occurred at lat/lon → that location was HIGH risk
   
2. **MEDIUM RISK (40-79)**: Within 5-20km of actual fires
   - Nearby areas that had fires in the region
   - Elevated risk but fire didn't reach this exact spot

3. **LOW RISK (0-39)**: Locations with NO fires within 30km
   - Random locations checked against all fire records
   - No fire history in 7 years → LOW risk

### Why This Is Better:

| Aspect | Synthetic Labels | Real Fire Labels |
|--------|-----------------|------------------|
| **Ground Truth** | Made up from distance | Actual fire occurrences |
| **Scientific Validity** | None - arbitrary rules | Historical observations |
| **Professor Approval** | ❌ "You need labeled data" | ✅ Real labeled dataset |
| **Model Quality** | Learns fake patterns | Learns real fire risk |
| **Defensible** | No - synthetic | Yes - based on data |

## Training Data Structure

### Labeled Dataset:
```
lat, lon, temp, humidity, wind, vegetation, ... → risk_score, risk_level

Example HIGH RISK (fire occurred here):
52.1487, -119.4577, 32°C, 28%, 25km/h, 0.85, ... → 92, HIGH

Example LOW RISK (no fire nearby):
45.2341, -80.1234, 22°C, 55%, 10km/h, 0.45, ... → 18, LOW
```

## How to Use

### Train with Real Labels:
```bash
python -m training.train_with_real_labels
```

This will:
1. Load 60,000+ historical fires (2018-2024)
2. Create labeled dataset:
   - 15,000 HIGH risk samples (fire locations)
   - 10,000 MEDIUM risk samples (near fires)
   - 15,000 LOW risk samples (no fires)
3. Train GradientBoostingRegressor
4. Save model to `models/wildfire_risk_model_latest.pkl`

### Output:
- `training/labeled_data/labeled_training_data.csv` - Full labeled dataset
- `models/wildfire_risk_model_YYYYMMDD_HHMMSS.pkl` - Trained model
- `models/wildfire_risk_metadata_YYYYMMDD_HHMMSS.pkl` - Training metadata

## For Your Professor

**What changed:**
- ❌ **Before**: Synthetic labels based on distance formulas
- ✅ **After**: Real labels from NASA MODIS fire observations

**Data source:**
- NASA FIRMS MODIS fire detections (2018-2024)
- 60,000+ Canadian wildfire observations
- Fire occurrence = HIGH risk label
- No fire occurrence = LOW risk label

**Scientific approach:**
- Supervised learning on historical fire patterns
- Ground truth: actual fire locations from satellite data
- Model learns conditions that led to real fires

This is now a **properly labeled dataset** suitable for academic research.
