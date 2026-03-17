# ForestShield Predictive Risk Heatmap

**Spatial AI inference system for regional wildfire risk prediction**

## Overview

Applies ForestShield's trained Gradient Boosting model across geographic grids to generate spatial wildfire risk maps. Runs batch AI inference on 100-2,500 locations per request.

**Key Features:**
- Batch AI predictions (400 model inferences per 20×20 grid)
- Uniform and circular grid generation
- Intelligent caching (100-300× speedup, 30-min TTL)
- Risk statistics and distribution analysis
- Flask REST API with 5 endpoints

## Installation

```bash
pip install flask flask-cors numpy scikit-learn joblib pandas scipy
```

Requires trained model at `../models/wildfire_risk_model.pkl`.

## Usage

### Python API

```python
from heatmap import HeatmapPredictor

predictor = HeatmapPredictor(cache_duration_minutes=30)

# Generate 10×10 grid (100 predictions)
heatmap = predictor.generate_heatmap(
    min_lat=43.0, max_lat=44.0, min_lng=-80.0, max_lng=-79.0,
    resolution=10, temperature=32.0, humidity=25.0, fire_distance_km=10.0
)

print(f"Mean Risk: {heatmap['statistics']['mean_risk']}")
print(f"High Risk Zones: {heatmap['statistics']['distribution']['HIGH']}")
```

### REST API

**Start server:**
```bash
python heatmap_api.py  # Runs on http://localhost:5001
```

**Generate heatmap:**
```bash
curl -X POST http://localhost:5001/heatmap/generate \
  -H "Content-Type: application/json" \
  -d '{"min_lat": 43.0, "max_lat": 44.0, "min_lng": -80.0, "max_lng": -79.0,
       "resolution": 15, "temperature": 28.5, "humidity": 35.0, "fire_distance_km": 25.0}'
```

**Circular heatmap:**
```bash
curl -X POST http://localhost:5001/heatmap/circular \
  -H "Content-Type: application/json" \
  -d '{"center_lat": 43.65, "center_lng": -79.38, "radius_km": 50.0,
       "resolution": 20, "temperature": 30.0, "humidity": 40.0, "fire_distance_km": 15.0}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/heatmap/generate` | POST | Rectangular grid heatmap |
| `/heatmap/circular` | POST | Circular area heatmap |
| `/heatmap/cache/stats` | GET | Cache performance metrics |
| `/heatmap/cache/clear` | POST | Clear cached predictions |
| `/heatmap/health` | GET | Health check |

## Response Format

```json
{
  "timestamp": "2026-03-16T21:30:00",
  "grid_info": {"resolution": 15, "total_points": 225},
  "predictions": [
    {"lat": 43.0, "lng": -80.0, "risk_score": 34.2, "risk_level": "MEDIUM"}
  ],
  "statistics": {
    "mean_risk": 48.3, "max_risk": 68.5, "min_risk": 22.1,
    "distribution": {"HIGH": 75, "MEDIUM": 120, "LOW": 30}
  }
}
```

**Risk Levels:** LOW (0-30), MEDIUM (31-60), HIGH (61-100)

## Real-World Integration

```python
# Use actual sensor readings as baseline conditions
sensor_data = {'temperature': 28.5, 'humidity': 38.0, 'nearest_fire_km': 22.3}

heatmap = predictor.generate_circular_heatmap(
    center_lat=45.5, center_lng=-78.5, radius_km=50.0, resolution=15,
    temperature=sensor_data['temperature'],
    humidity=sensor_data['humidity'],
    fire_distance_km=sensor_data['nearest_fire_km']
)
# Result: 145 AI predictions showing regional risk distribution
```

## Testing

```bash
python test_heatmap.py
```

Runs 5 tests: grid generation, batch predictions, caching, risk levels, JSON export.

## Technical Notes

- **Caching**: MD5-keyed with 30-min TTL for 100-300× speedup
- **Resolution**: 2-100 grid points per side (4-10,000 total predictions)
- **Grid Types**: Uniform (bounding box), Circular (Haversine distance)
- **Error Handling**: Graceful failures with UNKNOWN risk level fallback

---

**Part of ForestShield AI** | Trained on 60,970 Ontario wildfires | RMSE=5.43, R²=0.952
