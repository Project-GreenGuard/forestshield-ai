"""
Complete monitoring pipeline.
Combines performance tracking, drift detection, and alerting.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.monitor_performance import (
    load_prediction_log,
    calculate_metrics,
    print_performance_report
)
from training.drift_detection import (
    calculate_baseline_metrics,
    detect_drift,
    print_drift_report
)
from training.retraining_alerts import AlertSystem


def run_complete_monitoring():
    """
    Run full monitoring pipeline: performance → drift → alerts.
    """
    print("\n" + "="*70)
    print("COMPLETE MODEL MONITORING PIPELINE")
    print("="*70)
    
    # Paths
    predictions_csv = Path("training/logs/predictions.csv")
    baseline_csv = Path("training/logs/baseline_predictions.csv")
    
    # 1. Performance metrics
    print("\n1️⃣  CALCULATING PERFORMANCE METRICS...")
    df = load_prediction_log(predictions_csv)
    if df is not None and len(df) > 0:
        metrics = calculate_metrics(df)
        print_performance_report(metrics)
    else:
        print("[WARN] No predictions to analyze yet\n")
        return
    
    # 2. Drift detection
    print("\n2️⃣  DETECTING MODEL DRIFT...")
    baseline = calculate_baseline_metrics(baseline_csv)
    drift_result = detect_drift(predictions_csv, baseline)
    print_drift_report(drift_result)
    
    # 3. Alerting
    print("\n3️⃣  CHECKING FOR ALERTS...")
    alert_system = AlertSystem()
    needs_retrain = alert_system.check_retraining_needed(drift_result)
    alert_system.print_alerts()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model Status: {'⚠️  NEEDS ATTENTION' if needs_retrain else '✅ HEALTHY'}")
    print(f"Predictions Analyzed: {metrics.get('samples', 'N/A')}")
    print(f"Current RMSE: {metrics.get('rmse', 'N/A'):.2f}")
    print(f"Current Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_complete_monitoring()