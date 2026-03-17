"""Heatmap Risk Predictor - Batch prediction engine"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predict import predict_risk
from heatmap.grid_generator import GridGenerator


class HeatmapPredictor:
    """Generates wildfire risk heatmaps using AI model predictions"""
    
    def __init__(self, cache_duration_minutes: int = 30):
        self.grid_generator = GridGenerator()
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cache_file = Path(__file__).parent / 'heatmap_cache.json'
        self._load_cache()
    
    def generate_heatmap(self, min_lat: float, max_lat: float, min_lng: float, max_lng: float,
                        resolution: int = 20, temperature: float = 25.0, humidity: float = 45.0,
                        fire_distance_km: float = 50.0, use_cache: bool = True) -> Dict[str, Any]:
        """Generate complete risk heatmap for a geographic region"""
        cache_key = self._generate_cache_key(min_lat, max_lat, min_lng, max_lng, resolution,
                                             temperature, humidity, fire_distance_km)
        
        # Check cache
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now() - datetime.fromisoformat(cached['timestamp']) < self.cache_duration:
                cached['from_cache'] = True
                return cached
        
        # Generate grid  
        grid_points = self.grid_generator.generate_grid(min_lat, max_lat, min_lng, max_lng, resolution)
        grid_stats = self.grid_generator.calculate_grid_statistics(min_lat, max_lat, min_lng, max_lng, resolution)
        
        # Run predictions
        predictions, risk_scores = self._batch_predict(grid_points, temperature, humidity, fire_distance_km)
        
        # Build response
        heatmap_data = {
            'timestamp': datetime.now().isoformat(),
            'bounding_box': {'min_lat': min_lat, 'max_lat': max_lat, 'min_lng': min_lng, 'max_lng': max_lng},
            'conditions': {'temperature': temperature, 'humidity': humidity, 'fire_distance_km': fire_distance_km},
            'grid_info': {'resolution': resolution, **grid_stats},
            'predictions': predictions,
            'statistics': self._calc_stats(risk_scores, predictions),
            'from_cache': False
        }
        
        if use_cache:
            self.cache[cache_key] = heatmap_data
            self._save_cache()
        
        return heatmap_data
    
    def generate_circular_heatmap(self, center_lat: float, center_lng: float, radius_km: float,
                                  resolution: int = 20, temperature: float = 25.0, humidity: float = 45.0,
                                  fire_distance_km: float = 50.0) -> Dict[str, Any]:
        """Generate circular heatmap around a point"""
        grid_points = self.grid_generator.generate_adaptive_grid(center_lat, center_lng, radius_km, resolution)
        predictions, risk_scores = self._batch_predict(grid_points, temperature, humidity, fire_distance_km)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'center': {'lat': center_lat, 'lng': center_lng},
            'radius_km': radius_km,
            'conditions': {'temperature': temperature, 'humidity': humidity, 'fire_distance_km': fire_distance_km},
            'grid_info': {'resolution': resolution, 'total_points': len(predictions)},
            'predictions': predictions,
            'statistics': self._calc_stats(risk_scores, predictions)
        }
    
    def _batch_predict(self, grid_points: List[Dict], temp: float, humid: float, 
                      fire_dist: float) -> tuple[List[Dict], List[float]]:
        """Run batch predictions on grid points"""
        predictions, risk_scores = [], []
        
        for pt in grid_points:
            try:
                payload = {
                    'temperature': temp, 'humidity': humid,
                    'lat': pt['lat'], 'lng': pt['lng'],
                    'nearestFireDistance': fire_dist,
                    'timestamp': datetime.now().isoformat()
                }
                result = predict_risk(payload)
                predictions.append({
                    'lat': pt['lat'], 'lng': pt['lng'],
                    'risk_score': result['risk_score'],
                    'risk_level': result['risk_level'],
                    'confidence': result.get('confidence', 0.0)
                })
                risk_scores.append(result['risk_score'])
            except Exception as e:
                predictions.append({
                    'lat': pt['lat'], 'lng': pt['lng'],
                    'risk_score': 0.0, 'risk_level': 'UNKNOWN', 
                    'confidence': 0.0, 'error': str(e)
                })
        
        return predictions, risk_scores
    
    def _calc_stats(self, risk_scores: List[float], predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics"""
        if not risk_scores:
            return {'mean_risk': 0.0, 'max_risk': 0.0, 'min_risk': 0.0, 'median_risk': 0.0}
        
        stats = {
            'mean_risk': round(sum(risk_scores) / len(risk_scores), 2),
            'max_risk': round(max(risk_scores), 2),
            'min_risk': round(min(risk_scores), 2),
            'median_risk': round(sorted(risk_scores)[len(risk_scores) // 2], 2)
        }
        
        # Add distribution
        dist = {}
        for p in predictions:
            level = p['risk_level']
            dist[level] = dist.get(level, 0) + 1
        stats['distribution'] = dist
        
        return stats
    
    def _generate_cache_key(self, *args) -> str:
        """Generate MD5 cache key from parameters"""
        return hashlib.md5('_'.join(str(a) for a in args).encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except:
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk with cleanup"""
        try:
            now = datetime.now()
            self.cache = {k: v for k, v in self.cache.items()
                         if now - datetime.fromisoformat(v['timestamp']) < self.cache_duration}
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'cache_file': str(self.cache_file),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60
        }
