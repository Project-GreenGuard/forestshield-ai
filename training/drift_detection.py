"""
Detect model drift by comparing current performance to baseline.
Alerts when model performance degrades significantly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def calculate_baseline_metrics(baseline_csv: Path) -> Dict[str, float]:
    """
    Calculate baseline metrics from initial training data.
    
    Args:
        baseline_csv: Path to baseline predictions CSV
    
    Returns:
        Dict with baseline RMSE, accuracy, etc.
    """
    if not baseline_csv.exists():
        print(f"[WARN] Baseline file not found: {baseline_csv}")
        return {
            'rmse': 8.0,      # Default from training
            'accuracy': 0.929,
            'mae': 5.49
        }
    
    df = pd.read_csv(baseline_csv)
    
    y_true = df['actual_risk'].values
    y_pred = df['predicted_risk'].values
    
    baseline_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    baseline_mae = np.mean(np.abs(y_true - y_pred))
    
    def score_to_level(score):
        if score < 40:
            return 0
        elif score < 80:
            return 1
        else:
            return 2
    
    y_true_class = np.array([score_to_level(s) for s in y_true])
    y_pred_class = np.array([score_to_level(s) for s in y_pred])
    
    accuracy = np.mean(y_true_class == y_pred_class)
    
    return {
        'rmse': baseline_rmse,
        'mae': baseline_mae,
        'accuracy': accuracy
    }


def detect_drift(
    current_csv: Path,
    baseline_metrics: Dict[str, float],
    rmse_threshold: float = 0.15,
    accuracy_threshold: float = 0.10
) -> Dict[str, any]:
    """
    Detect if current model performance drifted from baseline.
    
    Args:
        current_csv: Path to recent predictions CSV
        baseline_metrics: Baseline performance metrics
        rmse_threshold: Alert if RMSE increase > this % (0.15 = 15%)
        accuracy_threshold: Alert if accuracy decrease > this % (0.10 = 10%)
    
    Returns:
        Dict with drift detection results
    """
    if not current_csv.exists():
        return {
            'has_drift': False,
            'reason': 'No predictions to analyze yet'
        }
    
    df = pd.read_csv(current_csv)
    
    if len(df) == 0:
        return {
            'has_drift': False,
            'reason': 'No predictions to analyze'
        }
    
    y_true = df['actual_risk'].values
    y_pred = df['predicted_risk'].values
    
    # Calculate current metrics
    current_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    current_mae = np.mean(np.abs(y_true - y_pred))
    
    def score_to_level(score):
        if score < 40:
            return 0
        elif score < 80:
            return 1
        else:
            return 2
    
    y_true_class = np.array([score_to_level(s) for s in y_true])
    y_pred_class = np.array([score_to_level(s) for s in y_pred])
    current_accuracy = np.mean(y_true_class == y_pred_class)
    
    # Compare to baseline
    baseline_rmse = baseline_metrics['rmse']
    baseline_accuracy = baseline_metrics['accuracy']
    
    rmse_increase = (current_rmse - baseline_rmse) / baseline_rmse
    accuracy_decrease = (baseline_accuracy - current_accuracy) / baseline_accuracy
    
    # Detect drift
    has_drift = False
    drift_reasons = []
    
    if rmse_increase > rmse_threshold:
        has_drift = True
        drift_reasons.append(
            f"RMSE degraded: {baseline_rmse:.2f} → {current_rmse:.2f} "
            f"(+{rmse_increase*100:.1f}%)"
        )
    
    if accuracy_decrease > accuracy_threshold:
        has_drift = True
        drift_reasons.append(
            f"Accuracy degraded: {baseline_accuracy:.2%} → {current_accuracy:.2%} "
            f"(-{accuracy_decrease*100:.1f}%)"
        )
    
    return {
        'has_drift': has_drift,
        'baseline_rmse': baseline_rmse,
        'current_rmse': current_rmse,
        'rmse_increase_pct': rmse_increase * 100,
        'baseline_accuracy': baseline_accuracy,
        'current_accuracy': current_accuracy,
        'accuracy_decrease_pct': accuracy_decrease * 100,
        'drift_reasons': drift_reasons,
        'num_predictions': len(df),
    }


def print_drift_report(drift_result: Dict):
    """Pretty print drift detection results."""
    print("\n" + "="*70)
    print("MODEL DRIFT DETECTION REPORT")
    print("="*70)
    
    print(f"\nAnalyzed {drift_result['num_predictions']} predictions")
    
    if drift_result['has_drift']:
        print("\n⚠️  DRIFT DETECTED!")
        for reason in drift_result['drift_reasons']:
            print(f"  • {reason}")
        print("\n🔧 ACTION REQUIRED: Consider retraining the model")
    else:
        print("\n✅ No drift detected - Model is stable")
        print(f"  RMSE: {drift_result['current_rmse']:.2f} (baseline: {drift_result['baseline_rmse']:.2f})")
        print(f"  Accuracy: {drift_result['current_accuracy']:.2%} (baseline: {drift_result['baseline_accuracy']:.2%})")
    
    print("\n" + "="*70 + "\n")


# Example usage
if __name__ == "__main__":
    baseline_path = Path("training/logs/baseline_predictions.csv")
    current_path = Path("training/logs/predictions.csv")
    
    # Get baseline
    baseline = calculate_baseline_metrics(baseline_path)
    print(f"Baseline RMSE: {baseline['rmse']:.2f}")
    print(f"Baseline Accuracy: {baseline['accuracy']:.2%}\n")
    
    # Detect drift
    drift_result = detect_drift(current_path, baseline)
    print_drift_report(drift_result)