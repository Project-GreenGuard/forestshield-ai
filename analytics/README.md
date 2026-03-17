# ForestShield AI - Analytics & Model Performance Tracking

**Production-ready MLOps system for monitoring wildfire risk prediction model performance in real-time.**

## Overview

This module implements **Machine Learning Operations (MLOps)** practices for the ForestShield AI wildfire prediction system. It provides continuous monitoring, performance analysis, and drift detection capabilities for the deployed Gradient Boosting model.

### Why This Matters

Most machine learning projects stop at model training. However, in production systems, models can degrade over time due to:
- **Data drift** - Input data distributions change
- **Concept drift** - Relationships between features and outcomes evolve
- **Performance degradation** - Model accuracy decreases over time

This analytics system addresses these challenges by tracking predictions, analyzing trends, and detecting anomalies in model behavior.

---

## Features

### 1. **Automated Prediction Logging**
Every prediction made by the AI model is automatically logged with:
- Timestamp and device identifier
- Risk score, level (LOW/MEDIUM/HIGH), and confidence
- Input sensor data (temperature, humidity, location, fire distance)
- Optional ground truth for validation

**Storage**: JSON-based log file with automatic rotation (maintains last 10,000 predictions)

### 2. **Risk Trend Analysis**
Analyzes historical predictions to identify patterns:
- **Time-series trends** - Are risk scores increasing or decreasing?
- **Risk distribution** - Breakdown by LOW/MEDIUM/HIGH categories
- **Hourly/daily averages** - Temporal patterns in predictions
- **Device-level analysis** - Per-sensor performance tracking

**Use Case**: Identify areas with consistently high risk, detect seasonal patterns, validate sensor reliability.

### 3. **Performance Metrics**
Tracks model behavior over time:
- **Confidence distribution** - Model certainty analysis
- **Prediction volume** - Request rate monitoring
- **Risk statistics** - Mean, median, std deviation of risk scores
- **Category breakdown** - LOW/MEDIUM/HIGH proportions

**Use Case**: Ensure model maintains consistent confidence levels, catch anomalous predictions.

### 4. **Model Drift Detection**
Compares recent predictions against historical baselines to detect degradation:
- **Mean drift** - Has average risk score shifted significantly?
- **Statistical significance** - Uses t-tests to validate drift
- **Distribution changes** - Compares risk level proportions
- **Alert generation** - Flags when drift exceeds thresholds

**Use Case**: Detect when model needs retraining, validate deployment success, catch data quality issues.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IoT Sensor Data Stream                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   Lambda Processing        │
         │   + AI Model Prediction    │◄──── NASA Fire Data
         └────────────┬───────────────┘
                      │
                      ├──────────────────┬──────────────────┐
                      ▼                  ▼                  ▼
              ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
              │  DynamoDB    │   │  Analytics  │   │  Dashboard   │
              │  (Storage)   │   │   Tracker   │   │  (Display)   │
              └──────────────┘   └─────────────┘   └──────────────┘
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │ Prediction    │
                                 │ Log (JSON)    │
                                 └───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │  Analytics API (Flask)       │
                          │  - GET /analytics/trends     │
                          │  - GET /analytics/drift      │
                          │  - GET /analytics/performance│
                          └──────────────────────────────┘
```

### Integration Points

**Automatic Logging** (process_sensor_data.py):
```python
# Every AI prediction is automatically logged
if AI_MODEL_AVAILABLE:
    prediction = predict_risk(sensor_payload)
    tracker.log_prediction(device_id, prediction, sensor_data)
```

**No manual intervention required** - Analytics accumulate as the system operates.

---

## Usage

### Running Analytics Tests
```bash
cd forestshield-ai/analytics
python test_analytics.py
```

Generates 50 sample predictions and demonstrates:
- ✓ Risk trend analysis
- ✓ Performance metrics calculation
- ✓ Model drift detection
- ✓ Historical data retrieval

### Starting Analytics API
```bash
cd forestshield-ai/analytics
python analytics_api.py
```

Starts Flask server on port 5001 with REST endpoints for dashboard integration.

### API Endpoints

**1. Get Risk Trends**
```http
GET /analytics/trends?hours=24&device_id=esp32-01
```
Returns: Trend analysis, mean risk scores, HIGH risk count, temporal patterns

**2. Get Performance Metrics**
```http
GET /analytics/performance?hours=168
```
Returns: Confidence stats, prediction volume, risk distribution (7-day default)

**3. Detect Model Drift**
```http
GET /analytics/drift?recent_hours=24&baseline_hours=168
```
Returns: Drift status, statistical tests, mean differences, alert flags

**4. Get Prediction History**
```http
GET /analytics/history?hours=48&limit=100
```
Returns: Raw prediction logs with timestamps, risk scores, sensor data

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `performance_tracker.py` | 343 | Core analytics engine, prediction logging, metrics calculation |
| `analytics_api.py` | 87 | Flask REST API for dashboard integration |
| `test_analytics.py` | 103 | Test script with sample data generation and validation |
| `prediction_log.json` | 1,802 entries | Historical prediction data storage |
| `__init__.py` | 6 | Python package initialization |

---

## Technical Details

### ML/AI Concepts Implemented

1. **Model Monitoring** - Continuous tracking of production model behavior
2. **Drift Detection** - Statistical tests (t-test) for distribution shifts
3. **Confidence Analysis** - Uncertainty quantification for model predictions
4. **Time-series Analysis** - Temporal pattern detection in predictions
5. **Performance Baselines** - Historical comparison for anomaly detection

### Data Structure (Prediction Log Entry)
```json
{
  "timestamp": "2026-03-16T21:40:34.694106Z",
  "device_id": "esp32-01",
  "prediction": {
    "risk_score": 37.0,
    "risk_level": "MEDIUM",
    "confidence": 0.757,
    "model_version": "v1.0-gradient-boost-nasa"
  },
  "sensor_data": {
    "temperature": 31.96,
    "humidity": 36.04,
    "lat": 44.08,
    "lng": -79.60,
    "nearestFireDistance": 159.59
  },
  "actual_outcome": null
}
```

---

## Why This Demonstrates ML Engineering Excellence

✅ **Beyond Training** - Shows understanding that ML deployment requires monitoring  
✅ **Production Mindset** - Implements industry-standard MLOps practices  
✅ **Proactive Maintenance** - Catches issues before they impact users  
✅ **Data-Driven** - Uses statistics to validate model health  
✅ **Scalable Architecture** - JSON log with rotation prevents unbounded growth  

This is the type of work done at companies like Google, Amazon, and Netflix to ensure ML models remain reliable in production.

---

## Future Enhancements

- **Automated Retraining** - Trigger model updates when drift detected
- **Alerting Integration** - Send notifications when anomalies found
- **A/B Testing** - Compare multiple model versions
- **Visualization Dashboard** - Real-time charts and graphs
- **Ground Truth Validation** - Accuracy tracking when outcomes known

---

## References

- **MLOps**: [Google ML Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- **Model Monitoring**: [AWS SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
- **Drift Detection**: Gama et al. (2014), "A Survey on Concept Drift Adaptation"

---

**Author**: ForestShield Team  
**Last Updated**: March 16, 2026  
**Module Version**: 1.0
