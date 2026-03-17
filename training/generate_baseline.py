"""
Generate baseline metrics from current predictions.
Use this once after initial model training to establish baseline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import pandas as pd
from training.monitor_performance import calculate_metrics


def generate_baseline():
    """
    Create baseline_predictions.csv from current predictions.csv
    This represents the 'good' model performance to compare against.
    """
    predictions_csv = Path("training/logs/predictions.csv")
    baseline_csv = Path("training/logs/baseline_predictions.csv")
    
    # Check if predictions exist
    if not predictions_csv.exists():
        print(f"[ERROR] {predictions_csv} not found")
        print("Run monitor_performance.py first to generate predictions.csv")
        return False
    
    # Copy predictions to baseline
    shutil.copy(predictions_csv, baseline_csv)
    print(f"✓ Baseline created: {baseline_csv}")
    
    # Show baseline metrics
    df = pd.read_csv(baseline_csv)
    metrics = calculate_metrics(df)
    
    print(f"\nBaseline Metrics:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Samples: {metrics['samples']}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING BASELINE METRICS")
    print("="*70 + "\n")
    
    success = generate_baseline()
    
    if success:
        print("\n✅ Baseline ready for drift detection")
    else:
        print("\n❌ Failed to generate baseline")