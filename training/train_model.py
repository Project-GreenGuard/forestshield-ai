"""
Model Training Pipeline for ForestShield v4

This script handles the complete training workflow:
1. Load synthetic data from data-generation module
2. Feature engineering and scaling
3. Hyperparameter tuning
4. Model training with RandomForest
5. Isotonic calibration
6. Model export

Usage:
    python training/train_model.py
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import joblib

# Import data generation module
from data_generation import generate_improved_synthetic_data

def create_calibrated_model(risk_model, X_train, y_train, X_test, y_test):
    """
    Calibrate model predictions using isotonic regression.
    Also optimizes thresholds for better classification.
    """
    # Get predictions for calibration
    y_pred_train = risk_model.predict(X_train)
    y_pred_test = risk_model.predict(X_test)
    
    # Fit isotonic calibrator on training predictions
    calibrator = IsotonicRegression(y_min=0, y_max=100, out_of_bounds='clip')
    calibrator.fit(y_pred_train, y_train)
    
    # Apply calibration
    y_pred_train_calib = calibrator.predict(y_pred_train)
    y_pred_test_calib = calibrator.predict(y_pred_test)
    
    # Compute metrics before/after calibration
    mae_before = mean_absolute_error(y_test, y_pred_test)
    r2_before = r2_score(y_test, y_pred_test)
    mae_after = mean_absolute_error(y_test, y_pred_test_calib)
    r2_after = r2_score(y_test, y_pred_test_calib)
    
    # Use official spec thresholds (per AI_PREDICTION_AND_TRAINING_SPEC.md)
    # LOW: 0-30, MEDIUM: 31-60, HIGH: 61-100
    official_thresholds = (30, 60)
    
    # Calculate classification accuracy with official thresholds
    predictions = []
    for score in y_pred_test_calib:
        if score <= 30:
            predictions.append(0)  # LOW
        elif score <= 60:
            predictions.append(1)  # MEDIUM
        else:
            predictions.append(2)  # HIGH
    
    actuals = []
    for score in y_test:
        if score <= 30:
            actuals.append(0)
        elif score <= 60:
            actuals.append(1)
        else:
            actuals.append(2)
    
    classification_accuracy = np.mean(np.array(predictions) == np.array(actuals))
    
    return {
        'model': risk_model,
        'calibrator': calibrator,
        'thresholds': official_thresholds,
        'metrics': {
            'mae_before': mae_before,
            'mae_after': mae_after,
            'r2_before': r2_before,
            'r2_after': r2_after,
            'classification_accuracy': classification_accuracy,
        }
    }

def train_optimized_model():
    """Train model with improved data and hyperparameter tuning."""

    print("FORESTSHIELD MODEL TRAINING PIPELINE (v4)")
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic training data...")
    df = generate_improved_synthetic_data(3000)
    print(f"  ✓ Generated {len(df)} realistic Ontario wildfire observations")
    print(f"    Risk score range: {df['risk_score'].min():.1f} - {df['risk_score'].max():.1f}")
    print(f"    Mean risk: {df['risk_score'].mean():.1f}")
    
    # Step 2: Prepare features
    print("\n[2/5] Preparing features & splitting data...")
    feature_cols = [
        'temperature', 'humidity', 'lat', 'lng', 'nearest_fire_distance',
        'hour_of_day', 'month', 'temp_normalized', 'humidity_inverse',
        'fire_proximity_score', 'day_of_week'
    ]
    X = df[feature_cols].values
    y = df['risk_score'].values
    
    # Standardize features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  ✓ Training set: {len(X_train)} samples")
    print(f"  ✓ Test set: {len(X_test)} samples")
    
    # Step 3: Train model with optimized hyperparameters
    print("\n[3/5] Training RandomForest model...")
    best_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Quick cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='r2')
    print(f"  ✓ 3-Fold CV R² scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  ✓ Average CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    best_model.fit(X_train, y_train)
    
    # Step 4: Evaluate and calibrate
    print("\n[4/5] Evaluating and calibrating model...")
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"  ✓ Training R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"  ✓ Test R²: {test_r2:.4f}, MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
    
    # Calibrate
    calib_result = create_calibrated_model(best_model, X_train, y_train, X_test, y_test)
    
    print(f"  ✓ Isotonic Calibration:")
    print(f"    MAE: {calib_result['metrics']['mae_before']:.2f} → {calib_result['metrics']['mae_after']:.2f}")
    print(f"    R²: {calib_result['metrics']['r2_before']:.4f} → {calib_result['metrics']['r2_after']:.4f}")
    print(f"    Classification Accuracy: {calib_result['metrics']['classification_accuracy']*100:.1f}%")
    print(f"    Official Spec Thresholds: LOW ≤{calib_result['thresholds'][0]}, MEDIUM {calib_result['thresholds'][0]+1}-{calib_result['thresholds'][1]}, HIGH >{calib_result['thresholds'][1]}")
    
    # Step 5: Feature importance
    print("\n[5/5] Feature importance analysis:")
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Attach metadata
    best_model.forestshield_version = "v4-optimized"
    best_model.feature_columns = feature_cols
    best_model.scaler = scaler
    best_model.calibrator = calib_result['calibrator']
    best_model.thresholds = calib_result['thresholds']
    best_model.metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'calibration_accuracy': calib_result['metrics']['classification_accuracy'],
    }
    
    return best_model

def save_model(model, output_dir="../models"):
    """Save trained model."""
    out_path = Path(__file__).parent.parent / "models"
    out_path.mkdir(parents=True, exist_ok=True)
    
    model_path = out_path / "forestshield_v4.joblib"
    joblib.dump(model, model_path)
    
    print(f"\n✓ Model saved to {model_path}")
    return model_path

def main():
    model = train_optimized_model()
    
    print("TRAINING COMPLETE")
    print(f"\nMetrics Summary:")
    for metric, value in model.metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    save_model(model)
    
    print("\n✓ ForestShield v4 model training complete!")
    print("✓ Model ready for deployment to Vertex AI")

if __name__ == "__main__":
    main()
