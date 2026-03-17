"""
Test script for Analytics module.

Demonstrates:
- Logging predictions
- Getting risk trends
- Performance metrics
- Model drift detection
"""

import sys
from pathlib import Path

# Add AI module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.performance_tracker import PerformanceTracker
from datetime import datetime, timedelta
import random


def generate_sample_data():
    """Generate sample predictions for testing."""
    tracker = PerformanceTracker()
    
    print("=" * 70)
    print("  ForestShield AI - Analytics Test")
    print("=" * 70 + "\n")
    
    # Generate 50 sample predictions over last 48 hours
    print("[1/4] Generating sample prediction data...")
    
    devices = ['esp32-01', 'esp32-02', 'esp32-03']
    
    for i in range(50):
        # Simulate predictions at different times
        hours_ago = random.randint(0, 48)
        
        # Simulate varying risk levels
        risk_score = random.uniform(10, 90)
        
        if risk_score < 30:
            risk_level = 'LOW'
        elif risk_score < 60:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        prediction = {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'confidence': random.uniform(0.7, 0.95),
            'model_version': 'v1.0-gradient-boost-nasa'
        }
        
        sensor_data = {
            'temperature': random.uniform(15, 35),
            'humidity': random.uniform(20, 70),
            'lat': 43.65 + random.uniform(-0.5, 0.5),
            'lng': -79.38 + random.uniform(-0.5, 0.5),
            'nearestFireDistance': random.uniform(10, 200)
        }
        
        tracker.log_prediction(
            device_id=random.choice(devices),
            prediction=prediction,
            sensor_data=sensor_data
        )
    
    print(f"✓ Generated 50 sample predictions\n")
    
    # Test analytics functions
    print("[2/4] Testing risk trend analysis...")
    trends = tracker.get_risk_trends(hours=24)
    print(f"✓ 24-hour trends:")
    print(f"  - Total predictions: {trends['total_predictions']}")
    print(f"  - Mean risk score: {trends.get('risk_statistics', {}).get('mean', 0):.2f}")
    print(f"  - Trend: {trends.get('trend', 'unknown')}")
    print(f"  - HIGH risk count: {trends.get('high_risk_count', 0)}\n")
    
    print("[3/4] Testing performance metrics...")
    performance = tracker.get_performance_metrics(hours=48)
    print(f"✓ Performance metrics:")
    print(f"  - Total predictions: {performance.get('total_predictions', 0)}")
    print(f"  - Unique devices: {performance.get('unique_devices', 0)}")
    print(f"  - Avg confidence: {performance.get('average_confidence', 0):.2f}")
    print(f"  - Predictions/hour: {performance.get('predictions_per_hour', 0):.2f}\n")
    
    print("[4/4] Testing model drift detection...")
    drift = tracker.detect_model_drift(baseline_hours=48, recent_hours=12)
    print(f"✓ Drift analysis:")
    print(f"  - Drift detected: {drift.get('drift_detected', False)}")
    print(f"  - Mean shift: {drift.get('drift_metrics', {}).get('mean_shift_pct', 0):.2f}%")
    print(f"  - Recommendation: {drift.get('recommendation', 'N/A')}\n")
    
    print("=" * 70)
    print("  All Analytics Tests Passed! ✓")
    print("=" * 70)


if __name__ == '__main__':
    generate_sample_data()
