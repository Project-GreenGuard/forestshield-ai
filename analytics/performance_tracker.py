"""
ForestShield AI - Performance Tracker

Tracks model predictions over time and analyzes performance metrics,
trends, and model drift detection.

Features:
- Historical prediction tracking
- Risk trend analysis
- Model drift detection
- Performance metrics aggregation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class PerformanceTracker:
    """
    Track and analyze model performance over time.
    
    Stores predictions in a simple JSON log file and provides
    analytics capabilities for monitoring model behavior.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize performance tracker.
        
        Args:
            log_path: Path to JSON log file. Defaults to analytics/prediction_log.json
        """
        if log_path is None:
            log_path = Path(__file__).parent / "prediction_log.json"
        
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not self.log_path.exists():
            self._write_log([])
    
    def log_prediction(self, 
                      device_id: str,
                      prediction: Dict[str, Any],
                      sensor_data: Dict[str, Any],
                      actual_outcome: Optional[bool] = None) -> None:
        """
        Log a prediction for later analysis.
        
        Args:
            device_id: Sensor device identifier
            prediction: Model prediction dict (risk_score, risk_level, confidence)
            sensor_data: Raw sensor data used for prediction
            actual_outcome: Optional ground truth (True if fire occurred, False otherwise)
        """
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'device_id': device_id,
            'prediction': {
                'risk_score': prediction['risk_score'],
                'risk_level': prediction['risk_level'],
                'confidence': prediction['confidence'],
                'model_version': prediction.get('model_version', 'unknown')
            },
            'sensor_data': {
                'temperature': sensor_data.get('temperature'),
                'humidity': sensor_data.get('humidity'),
                'lat': sensor_data.get('lat'),
                'lng': sensor_data.get('lng'),
                'nearestFireDistance': sensor_data.get('nearestFireDistance')
            },
            'actual_outcome': actual_outcome
        }
        
        # Append to log
        log_data = self._read_log()
        log_data.append(entry)
        
        # Keep only last 10,000 entries to prevent unbounded growth
        if len(log_data) > 10000:
            log_data = log_data[-10000:]
        
        self._write_log(log_data)
    
    def get_risk_trends(self, 
                       hours: int = 24,
                       device_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze risk trends over time.
        
        Args:
            hours: Number of hours to look back
            device_id: Optional filter for specific device
        
        Returns:
            Dict with trend statistics and time series data
        """
        log_data = self._read_log()
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter by time and device
        filtered = [
            entry for entry in log_data
            if datetime.fromisoformat(entry['timestamp'].replace('Z', '')) > cutoff
            and (device_id is None or entry['device_id'] == device_id)
        ]
        
        if not filtered:
            return {
                'period_hours': hours,
                'total_predictions': 0,
                'message': 'No predictions in time window'
            }
        
        # Calculate statistics
        risk_scores = [e['prediction']['risk_score'] for e in filtered]
        risk_levels = [e['prediction']['risk_level'] for e in filtered]
        
        level_counts = defaultdict(int)
        for level in risk_levels:
            level_counts[level] += 1
        
        # Detect trends (increasing/decreasing risk)
        recent_half = risk_scores[len(risk_scores)//2:]
        older_half = risk_scores[:len(risk_scores)//2]
        
        trend = 'stable'
        if len(recent_half) > 0 and len(older_half) > 0:
            recent_avg = np.mean(recent_half)
            older_avg = np.mean(older_half)
            change_pct = ((recent_avg - older_avg) / older_avg) * 100
            
            if change_pct > 10:
                trend = 'increasing'
            elif change_pct < -10:
                trend = 'decreasing'
        
        return {
            'period_hours': hours,
            'total_predictions': len(filtered),
            'risk_statistics': {
                'mean': float(np.mean(risk_scores)),
                'median': float(np.median(risk_scores)),
                'std_dev': float(np.std(risk_scores)),
                'min': float(np.min(risk_scores)),
                'max': float(np.max(risk_scores))
            },
            'risk_level_distribution': dict(level_counts),
            'trend': trend,
            'trend_change_pct': change_pct if len(recent_half) > 0 else 0,
            'high_risk_count': level_counts.get('HIGH', 0),
            'time_series': [
                {
                    'timestamp': e['timestamp'],
                    'risk_score': e['prediction']['risk_score'],
                    'risk_level': e['prediction']['risk_level']
                }
                for e in filtered[-100:]  # Last 100 points for plotting
            ]
        }
    
    def get_performance_metrics(self, hours: int = 168) -> Dict[str, Any]:
        """
        Calculate model performance metrics.
        
        Args:
            hours: Hours to look back (default 1 week)
        
        Returns:
            Dict with performance statistics
        """
        log_data = self._read_log()
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        filtered = [
            entry for entry in log_data
            if datetime.fromisoformat(entry['timestamp'].replace('Z', '')) > cutoff
        ]
        
        if not filtered:
            return {'message': 'No data available'}
        
        # Extract metrics
        predictions = len(filtered)
        unique_devices = len(set(e['device_id'] for e in filtered))
        
        # Confidence analysis
        confidences = [e['prediction']['confidence'] for e in filtered]
        avg_confidence = np.mean(confidences)
        
        # Model version tracking
        versions = defaultdict(int)
        for entry in filtered:
            versions[entry['prediction']['model_version']] += 1
        
        # Accuracy (if actual outcomes available)
        with_outcomes = [e for e in filtered if e.get('actual_outcome') is not None]
        accuracy = None
        
        if with_outcomes:
            correct = sum(
                1 for e in with_outcomes
                if (e['actual_outcome'] and e['prediction']['risk_level'] == 'HIGH') or
                   (not e['actual_outcome'] and e['prediction']['risk_level'] != 'HIGH')
            )
            accuracy = (correct / len(with_outcomes)) * 100
        
        return {
            'period_hours': hours,
            'total_predictions': predictions,
            'unique_devices': unique_devices,
            'predictions_per_hour': predictions / hours,
            'average_confidence': float(avg_confidence),
            'model_versions': dict(versions),
            'accuracy_pct': float(accuracy) if accuracy else None,
            'validated_predictions': len(with_outcomes)
        }
    
    def detect_model_drift(self, 
                          baseline_hours: int = 168,
                          recent_hours: int = 24) -> Dict[str, Any]:
        """
        Detect if model behavior has changed significantly.
        
        Compares recent predictions against historical baseline to
        identify potential model drift.
        
        Args:
            baseline_hours: Hours for baseline period (default 1 week)
            recent_hours: Hours for recent period (default 24h)
        
        Returns:
            Dict with drift analysis
        """
        log_data = self._read_log()
        
        baseline_cutoff = datetime.utcnow() - timedelta(hours=baseline_hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=recent_hours)
        
        baseline_data = [
            e for e in log_data
            if baseline_cutoff < datetime.fromisoformat(e['timestamp'].replace('Z', '')) < recent_cutoff
        ]
        
        recent_data = [
            e for e in log_data
            if datetime.fromisoformat(e['timestamp'].replace('Z', '')) > recent_cutoff
        ]
        
        if not baseline_data or not recent_data:
            return {
                'drift_detected': False,
                'message': 'Insufficient data for drift detection'
            }
        
        # Compare distributions
        baseline_scores = [e['prediction']['risk_score'] for e in baseline_data]
        recent_scores = [e['prediction']['risk_score'] for e in recent_data]
        
        baseline_mean = np.mean(baseline_scores)
        recent_mean = np.mean(recent_scores)
        
        baseline_std = np.std(baseline_scores)
        recent_std = np.std(recent_scores)
        
        # Calculate drift metrics
        mean_shift = abs(recent_mean - baseline_mean)
        mean_shift_pct = (mean_shift / baseline_mean) * 100
        std_shift = abs(recent_std - baseline_std)
        
        # Thresholds for drift detection
        drift_detected = mean_shift_pct > 15 or std_shift > 5
        
        return {
            'drift_detected': drift_detected,
            'baseline_period_hours': baseline_hours,
            'recent_period_hours': recent_hours,
            'baseline_stats': {
                'mean': float(baseline_mean),
                'std_dev': float(baseline_std),
                'samples': len(baseline_scores)
            },
            'recent_stats': {
                'mean': float(recent_mean),
                'std_dev': float(recent_std),
                'samples': len(recent_scores)
            },
            'drift_metrics': {
                'mean_shift': float(mean_shift),
                'mean_shift_pct': float(mean_shift_pct),
                'std_shift': float(std_shift)
            },
            'recommendation': 'Consider model retraining' if drift_detected else 'Model stable'
        }
    
    def _read_log(self) -> List[Dict[str, Any]]:
        """Read prediction log from disk."""
        try:
            with open(self.log_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _write_log(self, data: List[Dict[str, Any]]) -> None:
        """Write prediction log to disk."""
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2)


def calculate_model_drift(baseline_predictions: List[float],
                         recent_predictions: List[float]) -> Dict[str, Any]:
    """
    Standalone function to calculate drift between two prediction sets.
    
    Args:
        baseline_predictions: Historical prediction scores
        recent_predictions: Recent prediction scores
    
    Returns:
        Dict with drift statistics
    """
    if not baseline_predictions or not recent_predictions:
        return {'error': 'Empty prediction sets'}
    
    baseline_mean = np.mean(baseline_predictions)
    recent_mean = np.mean(recent_predictions)
    
    drift_pct = ((recent_mean - baseline_mean) / baseline_mean) * 100
    
    return {
        'baseline_mean': float(baseline_mean),
        'recent_mean': float(recent_mean),
        'drift_percentage': float(drift_pct),
        'drift_detected': abs(drift_pct) > 15
    }
