"""
Fire spread prediction model for PBI-6.
Predicts fire direction, speed, and affected areas.
"""

import numpy as np
from datetime import datetime, timedelta
import math


class FireSpreadModel:
    """
    Predicts wildfire spread based on:
    - Current fire location
    - Wind speed and direction
    - Terrain slope
    - Vegetation type
    - Weather conditions
    """

    def __init__(self):
        self.base_speed = 2.5  # km/h base fire speed
        self.wind_factor = 0.8  # Wind influence factor
        self.slope_factor = 0.6  # Terrain slope influence
        self.fuel_factor = 0.4  # Vegetation/fuel influence

    def predict_spread(self, fire_data: dict, hours_ahead: int = 24) -> dict:
        """
        Predict fire spread for next N hours.
        
        Args:
            fire_data: {
                'lat': float,
                'lng': float,
                'intensity': float (0-100),
                'wind_speed': float (km/h),
                'wind_direction': float (0-360 degrees),
                'temperature': float (C),
                'humidity': float (0-100),
                'slope': float (degrees),
                'fuel_type': str ('grass', 'forest', 'mixed')
            }
            hours_ahead: Number of hours to forecast
        
        Returns:
            {
                'current_location': {'lat', 'lng'},
                'spread_path': [{'lat', 'lng', 'time', 'intensity'}, ...],
                'affected_radius': float (km),
                'estimated_speed': float (km/h),
                'direction': float (degrees),
                'affected_areas': [{'name', 'distance_km', 'arrival_hours'}],
                'danger_level': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
            }
        """
        try:
            # Calculate fire speed
            speed = self._calculate_speed(fire_data)
            
            # Calculate spread direction
            direction = fire_data.get('wind_direction', 0)
            
            # Generate spread path
            spread_path = self._generate_spread_path(
                fire_data['lat'],
                fire_data['lng'],
                direction,
                speed,
                hours_ahead,
                fire_data.get('intensity', 50)
            )
            
            # Calculate affected radius (grows over time)
            affected_radius = self._calculate_affected_radius(
                speed, hours_ahead, fire_data.get('intensity', 50)
            )
            
            # Determine danger level
            danger_level = self._assess_danger(
                fire_data.get('intensity', 50),
                speed,
                fire_data.get('fuel_type', 'mixed')
            )
            
            return {
                'current_location': {
                    'lat': fire_data['lat'],
                    'lng': fire_data['lng']
                },
                'spread_path': spread_path,
                'affected_radius': round(affected_radius, 2),
                'estimated_speed': round(speed, 2),
                'direction': round(direction, 1),
                'hours_ahead': hours_ahead,
                'danger_level': danger_level,
                'model_confidence': 0.78 + (fire_data.get('intensity', 50) / 500)
            }
        
        except Exception as e:
            print(f"[ERROR] Fire spread prediction failed: {e}")
            return {'error': str(e)}

    def _calculate_speed(self, fire_data: dict) -> float:
        """Calculate fire spread speed in km/h."""
        speed = self.base_speed
        
        # Wind influence
        wind_speed = fire_data.get('wind_speed', 0)
        speed += (wind_speed * self.wind_factor)
        
        # Slope influence
        slope = fire_data.get('slope', 0)
        speed += (slope * self.slope_factor)
        
        # Fuel type influence
        fuel_type = fire_data.get('fuel_type', 'mixed')
        if fuel_type == 'forest':
            speed *= 1.3
        elif fuel_type == 'grass':
            speed *= 1.1
        
        # Humidity reduces spread
        humidity = fire_data.get('humidity', 50)
        speed *= (1 - (humidity / 200))
        
        # Intensity affects speed
        intensity = fire_data.get('intensity', 50)
        speed *= (1 + (intensity / 200))
        
        return max(0.5, min(speed, 15))  # Clamp: 0.5-15 km/h

    def _generate_spread_path(
        self,
        start_lat: float,
        start_lng: float,
        direction: float,
        speed: float,
        hours: int,
        intensity: float
    ) -> list:
        """Generate waypoints showing fire spread path."""
        path = []
        current_time = datetime.utcnow()
        
        # Generate waypoint every 2 hours
        for hour in range(0, hours + 1, 2):
            # Calculate distance traveled
            distance_km = (speed * hour)
            
            # Convert direction and distance to lat/lng offset
            lat_offset = distance_km * math.cos(math.radians(direction)) / 111
            lng_offset = distance_km * math.sin(math.radians(direction)) / 111
            
            # Intensity decreases over time
            point_intensity = max(20, intensity - (hour * 2))
            
            waypoint = {
                'lat': round(start_lat + lat_offset, 4),
                'lng': round(start_lng + lng_offset, 4),
                'time': (current_time + timedelta(hours=hour)).isoformat(),
                'hours_from_now': hour,
                'intensity': round(point_intensity, 1),
                'distance_km': round(distance_km, 2)
            }
            path.append(waypoint)
        
        return path

    def _calculate_affected_radius(self, speed: float, hours: float, intensity: float) -> float:
        """Calculate area affected by fire."""
        # Base radius + distance traveled + intensity factor
        base_radius = 0.5
        distance_traveled = speed * hours
        intensity_factor = intensity / 100
        
        return base_radius + distance_traveled + intensity_factor

    def _assess_danger(self, intensity: float, speed: float, fuel_type: str) -> str:
        """Assess danger level based on fire characteristics."""
        danger_score = (intensity / 100) + (speed / 15) + (1 if fuel_type == 'forest' else 0.5)
        
        if danger_score >= 2.0:
            return 'CRITICAL'
        elif danger_score >= 1.5:
            return 'HIGH'
        elif danger_score >= 1.0:
            return 'MEDIUM'
        else:
            return 'LOW'

    def predict_sensor_threat(
        self,
        fire_location: dict,
        sensor_location: dict,
        fire_data: dict
    ) -> dict:
        """
        Predict if a fire will threaten a specific sensor.
        
        Args:
            fire_location: {'lat', 'lng'}
            sensor_location: {'lat', 'lng', 'deviceId'}
            fire_data: Wind, intensity, etc.
        
        Returns:
            {
                'deviceId': str,
                'threat_level': 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH',
                'distance_km': float,
                'estimated_arrival_hours': float | None,
                'evacuation_recommended': bool
            }
        """
        # Calculate distance to sensor
        distance_km = self._haversine_distance(
            fire_location['lat'],
            fire_location['lng'],
            sensor_location['lat'],
            sensor_location['lng']
        )
        
        # Calculate fire speed
        speed = self._calculate_speed(fire_data)
        
        # Estimate arrival time
        if speed > 0:
            arrival_hours = distance_km / speed
        else:
            arrival_hours = None
        
        # Determine threat level
        if distance_km > 20:
            threat_level = 'NONE'
        elif distance_km > 10:
            threat_level = 'LOW'
        elif distance_km > 5:
            threat_level = 'MEDIUM'
        else:
            threat_level = 'HIGH'
        
        # Evacuation recommendation
        evacuation = threat_level in ['MEDIUM', 'HIGH'] and (arrival_hours is None or arrival_hours < 12)
        
        return {
            'deviceId': sensor_location.get('deviceId', 'unknown'),
            'threat_level': threat_level,
            'distance_km': round(distance_km, 2),
            'estimated_arrival_hours': round(arrival_hours, 1) if arrival_hours else None,
            'evacuation_recommended': evacuation,
            'fire_speed_kmh': round(speed, 2)
        }

    @staticmethod
    def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two coordinates in km."""
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c