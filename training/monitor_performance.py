"""
Simple performance monitoring script.
Load predictions and ground truth, calculate metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict  # ← ADD THIS LINE
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


def load_prediction_log(csv_path: Path) -> pd.DataFrame:
    """
    Load a CSV with predictions and ground truth.
    
    Expected columns: predicted_risk, actual_risk, timestamp
    
    Args:
        csv_path: Path to predictions CSV
    
    Returns:
        DataFrame with predictions
    """
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate performance metrics for predictions.
    
    Args:
        df: DataFrame with 'predicted_risk' and 'actual_risk' columns
    
    Returns:
        Dict with metrics
    """
    if df is None or len(df) == 0:
        return {}
    
    y_true = df['actual_risk'].values
    y_pred = df['predicted_risk'].values
    
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Convert to classification (LOW/MEDIUM/HIGH)
    def score_to_level(score):
        if score < 40:
            return 0  # LOW
        elif score < 80:
            return 1  # MEDIUM
        else:
            return 2  # HIGH
    
    y_true_class = np.array([score_to_level(s) for s in y_true])
    y_pred_class = np.array([score_to_level(s) for s in y_pred])
    accuracy = accuracy_score(y_true_class, y_pred_class)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy,
        'samples': len(df),
        'mean_actual': y_true.mean(),
        'mean_predicted': y_pred.mean(),
    }


def print_performance_report(metrics: Dict[str, float]):
    """Pretty print performance metrics."""
    if not metrics:
        print("[ERROR] No metrics to display")
        return
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE REPORT")
    print("="*70)
    print(f"\nSamples analyzed: {metrics['samples']}")
    print(f"\nRegression Metrics:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"\nClassification Accuracy: {metrics['accuracy']:.2%}")
    print(f"\nData Distribution:")
    print(f"  Mean actual risk:     {metrics['mean_actual']:.1f}")
    print(f"  Mean predicted risk:  {metrics['mean_predicted']:.1f}")
    print("\n" + "="*70 + "\n")


def generate_sample_log():
    """Generate sample prediction log for testing."""
    np.random.seed(42)
    
    # Simulate predictions with some error
    actual = np.random.uniform(0, 100, 100)
    predicted = actual + np.random.normal(0, 8, 100)
    predicted = np.clip(predicted, 0, 100)
    
    df = pd.DataFrame({
        'predicted_risk': predicted,
        'actual_risk': actual,
        'timestamp': pd.date_range('2024-06-01', periods=100, freq='D'),
    })
    
    return df


if __name__ == "__main__":
    # Generate sample log (replace with real data path)
    log_path = Path("training/logs/predictions.csv")
    
    # For demo, generate sample data
    print("Generating sample prediction log for demo...")
    sample_df = generate_sample_log()
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(log_path, index=False)
    print(f"✓ Sample log saved to {log_path}\n")
    
    # Load and analyze
    df = load_prediction_log(log_path)
    metrics = calculate_metrics(df)
    print_performance_report(metrics)