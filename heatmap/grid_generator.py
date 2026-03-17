"""
Grid Generator for Spatial Risk Analysis
Generates geographic grids for heatmap visualization
"""

import numpy as np
from typing import List, Dict, Tuple
import math


class GridGenerator:
    """
    Generates uniform grids of geographic coordinates for spatial analysis
    """
    
    def __init__(self):
        """Initialize the grid generator"""
        self.earth_radius_km = 6371.0
    
    def generate_grid(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lng: float, 
        max_lng: float,
        resolution: int = 20
    ) -> List[Dict[str, float]]:
        """
        Generate a uniform grid of points within bounding box
        
        Args:
            min_lat: Minimum latitude (south bound)
            max_lat: Maximum latitude (north bound)
            min_lng: Minimum longitude (west bound)
            max_lng: Maximum longitude (east bound)
            resolution: Number of points per side (N x N grid)
        
        Returns:
            List of dictionaries with 'lat' and 'lng' keys
        """
        if resolution < 2:
            raise ValueError("Resolution must be at least 2")
        
        if min_lat >= max_lat or min_lng >= max_lng:
            raise ValueError("Invalid bounding box coordinates")
        
        # Validate coordinate ranges
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= min_lng <= 180 and -180 <= max_lng <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        
        # Generate uniform grid
        latitudes = np.linspace(min_lat, max_lat, resolution)
        longitudes = np.linspace(min_lng, max_lng, resolution)
        
        grid_points = []
        for lat in latitudes:
            for lng in longitudes:
                grid_points.append({
                    'lat': round(float(lat), 6),
                    'lng': round(float(lng), 6)
                })
        
        return grid_points
    
    def generate_adaptive_grid(
        self,
        center_lat: float,
        center_lng: float,
        radius_km: float,
        resolution: int = 20
    ) -> List[Dict[str, float]]:
        """
        Generate a circular grid around a center point
        
        Args:
            center_lat: Center latitude
            center_lng: Center longitude
            radius_km: Radius in kilometers
            resolution: Number of points across diameter
        
        Returns:
            List of grid points within circular boundary
        """
        # Calculate bounding box from center and radius
        lat_delta = radius_km / 111.0  # 1 degree lat ≈ 111 km
        lng_delta = radius_km / (111.0 * math.cos(math.radians(center_lat)))
        
        min_lat = center_lat - lat_delta
        max_lat = center_lat + lat_delta
        min_lng = center_lng - lng_delta
        max_lng = center_lng + lng_delta
        
        # Generate rectangular grid
        all_points = self.generate_grid(min_lat, max_lat, min_lng, max_lng, resolution)
        
        # Filter to circular boundary
        circular_points = []
        for point in all_points:
            distance = self._haversine_distance(
                center_lat, center_lng, 
                point['lat'], point['lng']
            )
            if distance <= radius_km:
                circular_points.append(point)
        
        return circular_points
    
    def _haversine_distance(
        self, 
        lat1: float, 
        lng1: float, 
        lat2: float, 
        lng2: float
    ) -> float:
        """
        Calculate great circle distance between two points (Haversine formula)
        
        Args:
            lat1, lng1: First point coordinates
            lat2, lng2: Second point coordinates
        
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return self.earth_radius_km * c
    
    def calculate_grid_statistics(
        self,
        min_lat: float,
        max_lat: float,
        min_lng: float,
        max_lng: float,
        resolution: int
    ) -> Dict[str, float]:
        """
        Calculate statistics about grid coverage
        
        Args:
            min_lat, max_lat, min_lng, max_lng: Bounding box
            resolution: Grid resolution
        
        Returns:
            Dictionary with area, point count, and spacing info
        """
        # Calculate approximate area (assuming small region)
        lat_km = (max_lat - min_lat) * 111.0
        lng_km = (max_lng - min_lng) * 111.0 * math.cos(
            math.radians((min_lat + max_lat) / 2)
        )
        area_km2 = lat_km * lng_km
        
        # Grid stats
        total_points = resolution * resolution
        spacing_km = math.sqrt(area_km2 / total_points)
        
        return {
            'area_km2': round(area_km2, 2),
            'total_points': total_points,
            'spacing_km': round(spacing_km, 2),
            'coverage_lat_km': round(lat_km, 2),
            'coverage_lng_km': round(lng_km, 2)
        }
