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
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    print("Training Gradient Boosting Regressor...")
    model.fit(X_train, y_train)
    print("[OK] Training complete\n")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on held-out test set.
    
    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: True test risk scores
    
    Returns:
        Dict with performance metrics
    """
    print("="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Accuracy within ±10 points (per spec)
    within_10 = np.abs(y_pred - y_test) <= 10
    accuracy = np.mean(within_10) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy_within_10': accuracy,
        'test_samples': len(y_test)
    }
    
    print(f"Test samples: {len(y_test):,}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.3f}")
    print(f"Accuracy (±10 points): {accuracy:.1f}%")
    print("="*70 + "\n")
    
    return metrics


def main():
    """
    Main training pipeline:
    1. Load NASA MODIS fire data
    2. Generate synthetic training samples
    3. Prepare features
    4. Split train/test
    5. Train Gradient Boosting model
    6. Evaluate performance
    7. Save model
    """
    print("\n")
    print("=" * 70)
    print("    ForestShield AI - Model Training")
    print("=" * 70)
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Load NASA data (2018-2024: all 7 years)
        fires = load_nasa_data(data_dir, years=None)  # Load all available years
        
        # Step 2: Generate training samples
        samples = generate_training_samples(fires, samples_per_fire=20)
        
        # Step 3: Prepare features
        X, y = prepare_features(samples)
        
        print(f"\nDataset prepared:")
        print(f"  Training samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Feature names: {', '.join(FEATURE_NAMES[:5])}...")
        
        # Step 4: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Step 5: Train model
        model = train_model(X_train, y_train)
        
        # Step 6: Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 7: Save model with metadata
        model_path = models_dir / "wildfire_risk_model.pkl"
        model_data = {
            'model': model,
            'feature_names': FEATURE_NAMES,
            'metrics': metrics,
            'version': 'v1.0-gradient-boost-nasa',
            'trained_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'fires_used': len(fires)
        }
        
        joblib.dump(model_data, model_path)
        print(f"[OK] Model saved to: {model_path}")
        print(f"[OK] Model version: {model_data['version']}")
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure NASA MODIS CSV files are in the data/ directory")
        print("Download from: https://firms.modaps.eosdis.nasa.gov/")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

