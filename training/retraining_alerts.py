"""
Simple alerting system for model degradation.
Sends alerts when retraining is needed.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List


class Alert:
    """Simple alert object."""
    
    def __init__(self, severity: str, message: str):
        self.severity = severity  # CRITICAL, WARNING, INFO
        self.message = message
        self.timestamp = datetime.now().isoformat()
    
    def __str__(self):
        return f"[{self.severity}] {self.message}"


class AlertSystem:
    """Manages alerts for model monitoring."""
    
    def __init__(self, alert_log_path: Path = Path("training/logs/alerts.log")):
        self.alert_log_path = alert_log_path
        self.alerts: List[Alert] = []
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def add_alert(self, severity: str, message: str):
        """Add a new alert."""
        alert = Alert(severity, message)
        self.alerts.append(alert)
        self._log_alert(alert)
    
    def _log_alert(self, alert: Alert):
        """Write alert to log file."""
        with open(self.alert_log_path, 'a') as f:
            f.write(f"{alert.timestamp} | {alert.severity} | {alert.message}\n")
    
    def check_retraining_needed(self, drift_result: Dict) -> bool:
        """
        Determine if model should be retrained based on drift.
        
        Args:
            drift_result: Output from drift_detection.detect_drift()
        
        Returns:
            True if retraining recommended
        """
        if not drift_result['has_drift']:
            return False
        
        # CRITICAL: Retrain immediately
        if drift_result['rmse_increase_pct'] > 25:
            self.add_alert(
                'CRITICAL',
                f"RMSE increased {drift_result['rmse_increase_pct']:.1f}% - RETRAIN IMMEDIATELY"
            )
            return True
        
        # WARNING: Consider retraining
        if drift_result['accuracy_decrease_pct'] > 15:
            self.add_alert(
                'WARNING',
                f"Accuracy decreased {drift_result['accuracy_decrease_pct']:.1f}% - Consider retraining"
            )
            return True
        
        return False
    
    def print_alerts(self):
        """Print all recent alerts."""
        if not self.alerts:
            print("✅ No active alerts")
            return
        
        print("\n" + "="*70)
        print("ACTIVE ALERTS")
        print("="*70)
        
        for alert in self.alerts:
            color_code = {
                'CRITICAL': '🔴',
                'WARNING': '🟡',
                'INFO': '🟢'
            }.get(alert.severity, '⚪')
            
            print(f"{color_code} {alert}")
        
        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    alert_system = AlertSystem()
    
    # Simulate drift detection results
    drift_result = {
        'has_drift': True,
        'rmse_increase_pct': 28.5,
        'accuracy_decrease_pct': 12.0,
        'drift_reasons': ['RMSE increased 28.5%']
    }
    
    # Check if retraining needed
    needs_retrain = alert_system.check_retraining_needed(drift_result)
    
    print(f"Retraining needed: {needs_retrain}")
    alert_system.print_alerts()