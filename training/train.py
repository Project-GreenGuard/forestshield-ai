"""
ForestShield AI - Model Training Script

Trains wildfire risk prediction model using preprocessed data from NASA MODIS fires.

References:
- AI_PREDICTION_AND_TRAINING_SPEC.md Phase 1: Gradient Boosted Trees with rule-based labels
- Model output contract: risk_score (0-100), risk_level (LOW/MEDIUM/HIGH), model_version
"""

import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.features import FEATURE_NAMES
from training.data_preparation import load_nasa_data, generate_training_samples, prepare_features


def train_model(X_train, y_train):
    """
    Train Gradient Boosting Regressor model.
    
    Uses scikit-learn's GradientBoostingRegressor with hyperparameters
    optimized for Phase 1 (approximating rule-based risk scores).
    
    Args:
        X_train: Training feature matrix (N, 11)
        y_train: Training risk scores (N,)
    
    Returns:
        Trained GradientBoostingRegressor model
    """
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    print(f"Training samples: {len(X_train):,}")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("\n✓ Training complete")
    
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance on validation set.
    
    Calculates standard regression metrics and feature importance.
    
    Args:
        model: Trained model
        X_val: Validation feature matrix
        y_val: Validation risk scores
    
    Returns:
        Dict with rmse, mae, r2, accuracy_10 metrics
    """
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    print(f"Validation samples: {len(X_val):,}")
    
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, 100)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Accuracy within ±10
    accuracy_10 = np.mean(np.abs(y_val - y_pred) <= 10) * 100
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.3f}")
    print(f"Accuracy (±10): {accuracy_10:.1f}%")
    
    # Feature importance
    print(f"\nTop 5 Features:")
    importances = model.feature_importances_
    for name, importance in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:5]:
        print(f"  {name:25s} {importance:.3f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy_10': accuracy_10,
    }


def main():
    """
    Main training pipeline.
    
    Steps:
    1. Load NASA MODIS fire data (2018-2024)
    2. Generate synthetic sensor observations
    3. Build feature matrix
    4. Train/validation split
    5. Train Gradient Boosting model
    6. Evaluate performance
    7. Save model
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "ForestShield AI - Model Training" + " "*21 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Load NASA data (2018-2024: all 7 years)
        fires = load_nasa_data(data_dir, years=None)  # Load all available years
        
        # Step 2: Generate training samples
        samples = generate_training_samples(fires, samples_per_fire=20)
        
        # Step 3: Build features
        X, y = prepare_features(samples)
        
        # Step 4: Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {len(X_train):,}")
        print(f"Validation set: {len(X_val):,}")
        
        # Step 5: Train model
        model = train_model(X_train, y_train)
        
        # Step 6: Evaluate
        metrics = evaluate_model(model, X_val, y_val)
        
        # Step 7: Save model
        model_path = models_dir / "wildfire_risk_model.pkl"
        
        model_data = {
            'model': model,
            'feature_names': FEATURE_NAMES,
            'metrics': metrics,
            'trained_date': datetime.now().isoformat(),
            'version': 'v1.0-gradient-boost-nasa',
        }
        
        joblib.dump(model_data, model_path)
        
        print("\n" + "="*70)
        print("✅ SUCCESS")
        print("="*70)
        print(f"Model saved to: {model_path}")
        print(f"Performance: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

