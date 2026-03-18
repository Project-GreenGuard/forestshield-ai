"""
Fire Spread API - Uses trained ML model

⚠️ This is NOT a risk predictor - it predicts fire spread
"""

from .fire_spread_model import FireSpreadModel
from datetime import datetime


fire_model = FireSpreadModel()


def predict_fire_spread(fire_data: dict, hours_ahead: int = 24) -> dict:
    """
    Predict fire spread using TRAINED ML model.
    
    NOT the same as risk prediction - this is spread prediction.
    """
    try:
        result = fire_model.predict_spread(fire_data, hours_ahead)
        result['timestamp'] = datetime.utcnow().isoformat()
        result['hours_ahead'] = hours_ahead
        return result
    except Exception as e:
        print(f"[ERROR] Fire spread prediction failed: {e}")
        return {'error': str(e)}


def predict_sensor_threats(
    fire_location: dict,
    sensors: list,
    fire_data: dict
) -> dict:
    """
    Predict threats to multiple sensors.
    """
    try:
        threats = []
        evacuation_alerts = []
        
        for sensor in sensors:
            threat = fire_model.predict_sensor_threat(
                fire_location,
                sensor,
                fire_data
            )
            threats.append(threat)
            
            if threat['evacuation_recommended']:
                evacuation_alerts.append(sensor.get('deviceId'))
        
        return {
            'fire_location': fire_location,
            'sensor_threats': threats,
            'evacuation_alerts': evacuation_alerts,
            'total_sensors': len(sensors),
            'threatened_sensors': len([t for t in threats if t['threat_level'] != 'NONE']),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Sensor threat prediction failed: {e}")
        return {'error': str(e)}