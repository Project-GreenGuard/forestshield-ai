"""
Fetch weather data from Open-Meteo API (free, no API key required).
Extends feature vector with real weather conditions.

Note: Optimized for Ontario, Canada region.
"""

import requests
from typing import Tuple


# Free Open-Meteo API (no API key required)
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# Ontario bounding box for validation
ONTARIO_BOUNDS = {
    'min_lat': 41.913319,
    'max_lat': 56.86895,
    'min_lon': -95.154327,
    'max_lon': -74.324722,
}


def validate_ontario_location(lat: float, lng: float) -> bool:
    """
    Check if coordinates are within Ontario bounds.
    
    Args:
        lat: Latitude
        lng: Longitude
    
    Returns:
        True if within Ontario, False otherwise
    """
    in_bounds = (
        ONTARIO_BOUNDS['min_lat'] <= lat <= ONTARIO_BOUNDS['max_lat'] and
        ONTARIO_BOUNDS['min_lon'] <= lng <= ONTARIO_BOUNDS['max_lon']
    )
    return in_bounds


def fetch_weather_data(lat: float, lng: float) -> Tuple[float, float, float, float]:
    """
    Fetch real weather data from Open-Meteo API (free, no key required).
    
    Optimized for Ontario, Canada coordinates.
    Falls back to defaults if location is outside Ontario or API fails.
    
    Args:
        lat: Latitude
        lng: Longitude
    
    Returns:
        (temperature, humidity, wind_speed, wind_direction) tuple
        Falls back to (25.0, 50.0, 12.0, 180) if API fails or location invalid
    """
    # Validate Ontario location
    if not validate_ontario_location(lat, lng):
        print(f"[WARN] Location ({lat}, {lng}) outside Ontario bounds, using defaults")
        return 25.0, 50.0, 12.0, 180.0
    
    try:
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh"
        }
        
        response = requests.get(WEATHER_API_URL, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            temp = current.get('temperature_2m', 25.0)
            humidity = current.get('relative_humidity_2m', 50.0)
            wind_speed = current.get('wind_speed_10m', 12.0)
            wind_direction = current.get('wind_direction_10m', 180.0)
            
            # Validate reasonable ranges for Ontario
            temp = max(-40, min(50, temp))  # Ontario: -40°C to +50°C
            humidity = max(0, min(100, humidity))  # 0-100%
            wind_speed = max(0, min(100, wind_speed))  # 0-100 km/h
            wind_direction = wind_direction % 360  # 0-360°
            
            print(f"✅ Weather for ({lat:.4f}, {lng:.4f}): {temp}°C, {humidity}% RH, {wind_speed} km/h @ {wind_direction}°")
            
            return float(temp), float(humidity), float(wind_speed), float(wind_direction)
        else:
            print(f"[WARN] Weather API returned {response.status_code}")
            return 25.0, 50.0, 12.0, 180.0
    
    except Exception as e:
        print(f"[WARN] Weather fetch failed ({e}), using defaults")
        return 25.0, 50.0, 12.0, 180.0


def add_weather_features(payload: dict, use_api: bool = False) -> dict:
    """
    Optionally fetch and add real weather data to payload.
    
    For Ontario locations only. Blends API data with sensor readings.
    
    Args:
        payload: Original sensor payload with lat/lng
        use_api: If True, fetch from Open-Meteo API; if False, use provided values
    
    Returns:
        Payload with potentially updated weather data
    """
    if use_api and 'lat' in payload and 'lng' in payload:
        lat = payload['lat']
        lng = payload['lng']
        
        # Only fetch if in Ontario
        if not validate_ontario_location(lat, lng):
            print(f"[WARN] Skipping weather fetch - location outside Ontario")
            return payload
        
        api_temp, api_humidity, api_wind_speed, api_wind_direction = fetch_weather_data(lat, lng)
        
        # Blend: 70% API data, 30% provided sensor data
        if 'temperature' in payload:
            payload['temperature'] = (api_temp * 0.7 + payload['temperature'] * 0.3)
        else:
            payload['temperature'] = api_temp
        
        if 'humidity' in payload:
            payload['humidity'] = (api_humidity * 0.7 + payload['humidity'] * 0.3)
        else:
            payload['humidity'] = api_humidity
        
        # Update wind data
        payload['wind_speed'] = api_wind_speed
        payload['wind_direction'] = api_wind_direction
    
    return payload


# Example usage
if __name__ == "__main__":
    test_locations = [
        ("Toronto, ON", 43.6532, -79.3832),
        ("Ottawa, ON", 45.4215, -75.6972),
        ("London, ON", 42.9849, -81.2453),
    ]
    
    print("\n" + "="*70)
    print("WEATHER FEATURES - ONTARIO REGION TEST")
    print("="*70)
    print("Using Open-Meteo API (free, no API key required)\n")
    
    for name, lat, lng in test_locations:
        print(f"\n{name} ({lat}, {lng})")
        if validate_ontario_location(lat, lng):
            temp, humidity, wind_speed, wind_direction = fetch_weather_data(lat, lng)
            print(f"  Temp: {temp}°C | Humidity: {humidity}% | Wind: {wind_speed} km/h @ {wind_direction}°")
        else:
            print(f"  Location outside Ontario")