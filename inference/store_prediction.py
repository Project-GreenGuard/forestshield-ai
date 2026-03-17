"""
Store predictions to CSV for monitoring.
Connects inference output to performance monitoring.
"""

from pathlib import Path
import csv
from datetime import datetime
from typing import Optional


PREDICTIONS_CSV = Path("training/logs/predictions.csv")


def store_prediction(
    risk_score: float,
    risk_level: str,
    actual_risk: Optional[float] = None,
    timestamp: Optional[str] = None
) -> bool:
    """
    Store prediction to CSV for later monitoring and analysis.
    
    Args:
        risk_score: Predicted risk (0-100)
        risk_level: Predicted level (LOW/MEDIUM/HIGH)
        actual_risk: Ground truth risk (optional, added later)
        timestamp: When prediction was made (default: now)
    
    Returns:
        True if stored successfully
    """
    try:
        PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV if doesn't exist
        if not PREDICTIONS_CSV.exists():
            with open(PREDICTIONS_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'predicted_risk', 'actual_risk', 'risk_level'])
        
        # Append new prediction
        with open(PREDICTIONS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp or datetime.now().isoformat(),
                round(risk_score, 2),
                actual_risk or '',
                risk_level
            ])
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Failed to store prediction: {e}")
        return False


# Example usage
if __name__ == "__main__":
    print("\nTesting prediction storage...\n")
    
    # Simulate several predictions
    test_predictions = [
        (75.5, "HIGH", None),
        (45.2, "MEDIUM", None),
        (15.8, "LOW", None),
    ]
    
    for risk_score, risk_level, actual in test_predictions:
        success = store_prediction(risk_score, risk_level, actual)
        if success:
            print(f"✓ Stored: {risk_score}/100 ({risk_level})")
    
    print(f"\n✓ Predictions saved to {PREDICTIONS_CSV}")