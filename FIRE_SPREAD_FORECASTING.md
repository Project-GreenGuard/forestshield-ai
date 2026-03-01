# Fire Spread Forecasting

**Status:** Implemented & Smoke Tested (ML model + heuristic fallback)  
**Purpose:** Estimates fire spread rate (km/h) and 24h burn area (hectares) for operational planning

---

## Overview

Extends risk prediction (Will fire happen?) to **spread forecasting** (How fast will it move?).

| Aspect | Risk Prediction | Spread Forecasting |
|--------|-----------------|-------------------|
| Output | LOW/MEDIUM/HIGH | km/h, hectares/24h |
| Users | Citizens (alerts) | First responders |
| Use case | Evacuation timing | Resource allocation, route planning |

---

## Model Architecture

**Algorithm:** GradientBoostingRegressor (captures exponential wind effects, multiplicative interactions)

The spread model is loaded from `models/forestshield_spread_v1.joblib`. If unavailable, the system falls back to a physics-inspired heuristic.

**ML Model Inputs (13 features when artifact is available):**
- Risk factors: temperature, humidity, lat, lng, nearest_fire_distance
- Time features: hour_of_day, month (derived from timestamp)
- Spread factors: wind_speed_kmh, wind_direction_deg, fuel_moisture_percent, canopy_density
- Categorical: vegetation_type (encoded 0-3), topography (encoded 0-3)

**Outputs:**
- `spread_rate_kmh`: Fire velocity (ML prediction or heuristic)
- `burn_area_24h_hectares`: Estimated using simplified operational assumption (~20 hectares per 1 km/h spread rate over 24 hours)
- `fireline_intensity_kw_m`: Heuristic-derived heat release intensity

**Training:** 3000 synthetic observations based on fire physics

**Why GradientBoosting (vs RandomForest for risk)?**
- Captures non-linear interactions (wind 2x → spread 2.5x)
- Models multiplicative effects (temp × dryness × wind)
- Better for threshold-based patterns (fuel moisture <30% = critical)

---

## Example Outputs

| Scenario | Conditions | Spread Rate | Burn Area (24h) | Interpretation |
|----------|-----------|-------------|-----------------|----------------|
| **Mild** | 15°C, 75% humidity, 5 km/h wind, grassland | 1.50 km/h | 28.5 hectares | Manageable by local services |
| **Dangerous** | 35°C, 20% humidity, 30 km/h wind, forest | 3.36 km/h | 68.8 hectares | Requires provincial resources |
| **Extreme** | 40°C, 10% humidity, 40 km/h wind, mountains | 3.76 km/h | 76.0 hectares | Catastrophic - mobilize all resources |

**Operational actions for dangerous/extreme:**
- Evacuate residents within 20+ km radius
- Fire moves 20 km in 6 hours
- Deploy helicopters + ground crews
- Establish fire breaks 5 km ahead of fire front

---

## Usage

```python
from inference.spread_predict import predict_spread

forecast = predict_spread(
    temperature=35.0,
    humidity=20.0,
    lat=49.5,
    lng=-93.2,
    wind_speed_kmh=30.0,
    vegetation_type="forest",
    topography="hilly"
)
# Returns: {'spread_rate_kmh': 3.36, 'burn_area_24h_hectares': 68.8, ...}
```

**Dashboard integration:**
- Map showing fire spread prediction (expanding red circle)
- "Fire will reach your area in X hours"
- Resource requirements, suppression tactics, safe routes

---

## Testing

- ✅ Smoke tested with mild/dangerous/extreme scenarios (module runs end-to-end)
- ✅ Produces stable `spread_rate_kmh` and derived `burn_area_24h_hectares`
- ✅ Falls back to heuristic when ML artifact is unavailable
- ✅ Integrates with risk prediction model (ForestShield v4)

Performance metrics were measured during training (see training logs/validation output if available)

---

## Limitations & Future Work

**Current limitations:**
- Assumes stable weather for 24h (real weather changes hourly)
- Doesn't account for active firefighting suppression
- Simplified terrain effects (no slope angle, vegetation barriers)
- Assumes unlimited fuel (doesn't model fuel depletion)

**Future enhancements:**
- Multi-hour segmented forecasts (6h, 12h, 24h)
- Probabilistic spread (confidence intervals)
- Suppression impact model
- Weather API integration (NOAA)

---

## Files

- **Inference:** `inference/spread_predict.py`
- **Model artifact:** `models/forestshield_spread_v1.joblib` (optional, 1.6 MB if present)
- **Training:** (separate script required; not included in risk training pipeline)

**Integration status:** Inference code operational, ready for backend API integration and real-time dashboard deployment.
