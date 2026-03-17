"""
Combine ML predictions with rule-based risk scoring.
Returns both scores for transparency and debugging.
"""

import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.predict import predict_risk as predict_risk_ml


def calculate_rule_based_risk(sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Old-school rule-based risk scoring based on distance and conditions.
    
    Args:
        sensor_payload: Dict with sensor data
    
    Returns:
        Dict with rule_score, rule_level
    """
    temperature = sensor_payload.get('temperature', 25)
    humidity = sensor_payload.get('humidity', 50)
    fire_distance = sensor_payload.get('nearestFireDistance', 50)
    
    # Distance-based scoring
    if fire_distance < 5:
        distance_score = 85
    elif fire_distance < 15:
        distance_score = 65
    elif fire_distance < 30:
        distance_score = 40
    elif fire_distance < 50:
        distance_score = 25
    else:
        distance_score = 15
    
    # Adjust for temperature (higher temp = higher risk)
    if temperature > 35:
        temp_adj = +10
    elif temperature > 30:
        temp_adj = +5
    else:
        temp_adj = 0
    
    # Adjust for humidity (lower humidity = higher risk)
    if humidity < 30:
        humidity_adj = +10
    elif humidity < 45:
        humidity_adj = +5
    else:
        humidity_adj = 0
    
    rule_score = distance_score + temp_adj + humidity_adj
    rule_score = min(100, max(0, rule_score))  # Clamp 0-100
    
    # Rule-based risk level
    if rule_score <= 30:
        rule_level = "LOW"
    elif rule_score <= 60:
        rule_level = "MEDIUM"
    else:
        rule_level = "HIGH"
    
    return {
        'rule_score': round(rule_score, 2),
        'rule_level': rule_level,
        'distance_component': distance_score,
        'temp_adjustment': temp_adj,
        'humidity_adjustment': humidity_adj,
    }


def predict_risk_hybrid(sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict risk using both ML and rule-based methods.
    
    Returns both scores and a simple weighted average.
    
    Args:
        sensor_payload: Dict with sensor data
    
    Returns:
        Dict with ml_score, rule_score, combined_score, and recommendations
    """
    # Get ML prediction
    ml_result = predict_risk_ml(sensor_payload)
    
    # Get rule-based prediction
    rule_result = calculate_rule_based_risk(sensor_payload)
    
    # Simple average (can adjust weights if needed)
    ml_weight = 0.6  # ML gets more weight (trained on real data)
    rule_weight = 0.4
    
    combined_score = (
        ml_result['risk_score'] * ml_weight +
        rule_result['rule_score'] * rule_weight
    )
    combined_score = round(combined_score, 2)
    
    # Combined risk level
    if combined_score <= 30:
        combined_level = "LOW"
    elif combined_score <= 60:
        combined_level = "MEDIUM"
    else:
        combined_level = "HIGH"
    
    return {
        'ml_score': ml_result['risk_score'],
        'ml_level': ml_result['risk_level'],
        'ml_confidence': ml_result['confidence'],
        'rule_score': rule_result['rule_score'],
        'rule_level': rule_result['rule_level'],
        'combined_score': combined_score,
        'combined_level': combined_level,
        'model_version': ml_result['model_version'],
        'note': f"ML weight: {ml_weight*100:.0f}%, Rule weight: {rule_weight*100:.0f}%"
    }


# Example usage
if __name__ == "__main__":
    test_payload = {
        'temperature': 32.0,
        'humidity': 25.0,
        'lat': 49.2827,
        'lng': -123.1207,
        'nearestFireDistance': 8.0,
        'timestamp': '2024-07-15T15:30:00Z',
    }
    
    result = predict_risk_hybrid(test_payload)
    
    print("\n" + "="*70)
    print("HYBRID RISK PREDICTION")
    print("="*70)
    print(f"\nML-Based Prediction:")
    print(f"  Score: {result['ml_score']}/100")
    print(f"  Level: {result['ml_level']}")
    print(f"  Confidence: {result['ml_confidence']:.0%}")
    
    print(f"\nRule-Based Prediction:")
    print(f"  Score: {result['rule_score']}/100")
    print(f"  Level: {result['rule_level']}")
    
    print(f"\nCombined Prediction:")
    print(f"  Score: {result['combined_score']}/100")
    print(f"  Level: {result['combined_level']}")
    print(f"  ({result['note']})")
    print("\n" + "="*70)