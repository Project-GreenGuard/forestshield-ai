"""
ForestShield AI - Model Training Script

Train Wildfire Risk Model with REAL Labels from Fire Occurrences

Uses actual fire locations as HIGH risk, non-fire locations as LOW risk.
This provides ground truth labels instead of synthetic distance-based labels.
"""

import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.features import FEATURE_NAMES
from training.data_preparation import load_nasa_data
from training.label_generator import create_labeled_dataset, prepare_labeled_features


def main():
    """
    Complete training pipeline using real fire occurrence labels.
    """
    print("\n" + "="*70)
    print("WILDFIRE RISK PREDICTION - TRAINING WITH REAL LABELS")
    print("="*70)
    print("Uses actual fire occurrences as ground truth instead of synthetic labels\n")
    
    # 1. Load historical fire data
    data_dir = Path("data")
    fires_df = load_nasa_data(data_dir, years=None, filter_ontario=True)
    
    # 2. Create labeled dataset
    region_bounds = {
        'min_lon': -95.154327,
        'max_lon': -74.324722,
        'min_lat': 41.913319,
        'max_lat': 56.86895
    }
    
    labeled_df = create_labeled_dataset(
        fires_df,
        region_bounds,
        high_risk_samples=15000,  # Fire locations
        medium_risk_samples=10000,  # Near fires
        low_risk_samples=15000  # No fires
    )
    
    # Save labeled dataset for inspection
    output_dir = Path("training/labeled_data")
    output_dir.mkdir(exist_ok=True)
    labeled_df.to_csv(output_dir / "labeled_training_data.csv", index=False)
    print(f"\n✓ Saved labeled data to {output_dir}/labeled_training_data.csv")
    
    # 3. Build feature matrix
    X, y = prepare_labeled_features(labeled_df)
    
    # 4. Split data
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y > 50).astype(int)
    )
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # 5. Train model
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    print(f"Using GradientBoostingRegressor")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Regression metrics
    print("\nRegression Metrics (Risk Score 0-100):")
    print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
    print(f"  Test RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
    print(f"  Train MAE:  {mean_absolute_error(y_train, y_pred_train):.2f}")
    print(f"  Test MAE:   {mean_absolute_error(y_test, y_pred_test):.2f}")
    print(f"  Train R²:   {r2_score(y_train, y_pred_train):.3f}")
    print(f"  Test R²:    {r2_score(y_test, y_pred_test):.3f}")
    
    # Classification metrics (LOW/MEDIUM/HIGH)
    def to_risk_level(score):
        if score < 40:
            return 0  # LOW
        elif score < 80:
            return 1  # MEDIUM
        else:
            return 2  # HIGH
    
    y_train_class = np.array([to_risk_level(s) for s in y_train])
    y_test_class = np.array([to_risk_level(s) for s in y_test])
    y_pred_train_class = np.array([to_risk_level(s) for s in y_pred_train])
    y_pred_test_class = np.array([to_risk_level(s) for s in y_pred_test])
    
    print("\nClassification Metrics (LOW/MEDIUM/HIGH):")
    print(f"  Train Accuracy: {accuracy_score(y_train_class, y_pred_train_class):.3f}")
    print(f"  Test Accuracy:  {accuracy_score(y_test_class, y_pred_test_class):.3f}")
    
    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    importances = model.feature_importances_
    for name, importance in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:25s}: {importance:.4f}")
    
    # 7. Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = models_dir / "wildfire_risk_model.pkl"
    
    joblib.dump(model, model_file)
    print(f"✓ Model saved: {model_file}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'GradientBoostingRegressor',
        'training_method': 'REAL_FIRE_OCCURRENCE_LABELS',
        'feature_names': FEATURE_NAMES,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'test_accuracy': float(accuracy_score(y_test_class, y_pred_test_class)),
        'label_source': 'NASA_MODIS_HISTORICAL_FIRES_2018-2024'
    }
    
    metadata_file = models_dir / "wildfire_risk_metadata.pkl"
    joblib.dump(metadata, metadata_file)
    print(f"✓ Metadata saved: {metadata_file}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nModel trained on REAL fire occurrence labels:")
    print(f"  • {labeled_df[labeled_df['risk_level'] == 'HIGH'].shape[0]:,} HIGH risk (actual fire locations)")
    print(f"  • {labeled_df[labeled_df['risk_level'] == 'MEDIUM'].shape[0]:,} MEDIUM risk (near fires)")
    print(f"  • {labeled_df[labeled_df['risk_level'] == 'LOW'].shape[0]:,} LOW risk (no fires nearby)")
    print("\nThis model learns from WHERE FIRES ACTUALLY HAPPENED, not synthetic labels.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

