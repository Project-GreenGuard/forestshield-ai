"""
API endpoints for fire spread forecasting (PBI-6).
"""

from .fire_spread_model import FireSpreadModel
from datetime import datetime


fire_model = FireSpreadModel()


def predict_fire_spread(fire_data: dict, hours_ahead: int = 24) -> dict:
    """
    Predict fire spread for next N hours.
    
    Request:
    {
        "lat": 43.6532,
        "lng": -79.3832,
        "intensity": 65,
        "wind_speed": 15,
        "wind_direction": 45,
        "temperature": 32,
        "humidity": 25,
        "slope": 12,
        "fuel_type": "forest"
    }
    
    Response:
    {
        "current_location": {...},
        "spread_path": [...],
        "affected_radius": 8.5,
        "estimated_speed": 4.2,
        "direction": 45,
        "danger_level": "HIGH"
    }
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
    
    Request:
    {
        "fire_location": {"lat": 43.6532, "lng": -79.3832},
        "sensors": [
            {"deviceId": "SENSOR-001", "lat": 43.7, "lng": -79.4},
            ...
        ],
        "fire_data": {
            "intensity": 65,
            "wind_speed": 15,
            "wind_direction": 45,
            ...
        }
    }
    
    Response:
    {
        "fire_location": {...},
        "sensor_threats": [...],
        "evacuation_alerts": ["SENSOR-001", "SENSOR-003"],
        "timestamp": "2024-07-15T..."
    }
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


def analyze_fire_progression(fire_locations_timeline: list) -> dict:
    """
    Analyze fire progression from multiple readings over time.
    Estimates speed, direction changes, and acceleration.
    
    Request:
    {
        "locations": [
            {"lat": 43.6, "lng": -79.4, "timestamp": "2024-07-15T10:00:00Z", "intensity": 50},
            {"lat": 43.65, "lng": -79.35, "timestamp": "2024-07-15T12:00:00Z", "intensity": 65},
            {"lat": 43.7, "lng": -79.3, "timestamp": "2024-07-15T14:00:00Z", "intensity": 75}
        ]
    }
    
    Response:
    {
        "num_readings": 3,
        "time_span_hours": 4,
        "average_speed_kmh": 5.2,
        "direction_trend": 45,
        "intensity_trend": "INCREASING",
        "acceleration": "ACCELERATING"
    }
    """
    try:
        if len(fire_locations_timeline) < 2:
            return {'error': 'Need at least 2 location readings'}
        
        distances = []
        time_intervals = []
        intensities = []
        
        for i in range(len(fire_locations_timeline) - 1):
            current = fire_locations_timeline[i]
            next_loc = fire_locations_timeline[i + 1]
            
            # Calculate distance
            distance = fire_model._haversine_distance(
                current['lat'], current['lng'],
                next_loc['lat'], next_loc['lng']
            )
            distances.append(distance)
            
            # Calculate time interval
            from datetime import datetime
            time1 = datetime.fromisoformat(current['timestamp'].replace('Z', '+00:00'))
            time2 = datetime.fromisoformat(next_loc['timestamp'].replace('Z', '+00:00'))
            hours = (time2 - time1).total_seconds() / 3600
            time_intervals.append(hours)
            
            # Track intensity
            intensities.append(next_loc.get('intensity', 50))
        
        # Calculate average speed
        total_distance = sum(distances)
        total_time = sum(time_intervals)
        avg_speed = total_distance / total_time if total_time > 0 else 0
        
        # Intensity trend
        intensity_start = fire_locations_timeline[0].get('intensity', 50)
        intensity_end = fire_locations_timeline[-1].get('intensity', 50)
        if intensity_end > intensity_start:
            intensity_trend = 'INCREASING'
        elif intensity_end < intensity_start:
            intensity_trend = 'DECREASING'
        else:
            intensity_trend = 'STABLE'
        
        # Acceleration
        if len(distances) >= 2:
            acceleration = distances[-1] - distances[-2]
            if acceleration > 0.5:
                accel_status = 'ACCELERATING'
            elif acceleration < -0.5:
                accel_status = 'DECELERATING'
            else:
                accel_status = 'STABLE'
        else:
            accel_status = 'UNKNOWN'
        
        return {
            'num_readings': len(fire_locations_timeline),
            'time_span_hours': round(total_time, 1),
            'average_speed_kmh': round(avg_speed, 2),
            'intensity_trend': intensity_trend,
            'acceleration': accel_status,
            'total_distance_km': round(total_distance, 2)
        }
    except Exception as e:
        print(f"[ERROR] Fire progression analysis failed: {e}")
        return {'error': str(e)}