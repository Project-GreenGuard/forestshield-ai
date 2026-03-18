"""
Train Fire Spread Model on labeled data.

⚠️ FIRE SPREAD MODEL (NOT RISK MODEL)

This model predicts fire spread behavior AFTER ignition:
- spread_rate_kmh: How fast fire spreads
- direction_degrees: Direction fire travels
- intensity_level: Fire intensity (0-10)
- area_hectares: Area covered

This is COMPLETELY DIFFERENT from the risk model which predicts
if/where fire will OCCUR.

Model Type: MultiOutputRegressor (trains 4 targets simultaneously)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🔥 FIRE SPREAD MODEL TRAINING (MultiOutput)")
print("="*70)

# ===== SETUP: Create models directory =====
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
print(f"\n📂 Model directory: {MODEL_DIR}")

# ===== STEP 1: Load enhanced data with spread labels =====
print("\n📂 Loading labeled training data with spread labels...")
data_path = Path(__file__).resolve().parent / "labeled_data" / "labeled_training_data_with_spread.csv"

if not data_path.exists():
    print(f"❌ File not found: {data_path}")
    print("Run: python generate_spread_labels.py")
    exit(1)

df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df):,} fire records")
print(f"\nDataset columns: {list(df.columns)}")

# ===== STEP 2: Prepare features =====
print("\n🔧 Preparing features...")

# FIRE SPREAD FEATURES (9 environmental factors)
feature_columns = [
    'temperature',
    'humidity',
    'wind_speed',
    'vegetation_density',
    'soil_moisture',
    'elevation',
    'nearest_water',
    'fire_history',
    'population_density'
]

# SPREAD TARGETS (4 spread-specific outputs)
target_columns = [
    'spread_rate_kmh',
    'direction_degrees',
    'intensity_level',
    'area_hectares'
]

print(f"\n✅ Input Features ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"   {i}. {col}")

print(f"\n✅ Target Columns ({len(target_columns)}):")
for i, col in enumerate(target_columns, 1):
    print(f"   {i}. {col}")

# Extract X and y
X = df[feature_columns].copy()
y = df[target_columns].copy()

print(f"\n📊 Data shapes:")
print(f"   X: {X.shape}")
print(f"   y: {y.shape}")

# Check for missing values
print(f"\n📊 Checking data quality:")
missing_X = X.isnull().sum().sum()
missing_y = y.isnull().sum().sum()

if missing_X > 0 or missing_y > 0:
    print(f"⚠️  Found {missing_X + missing_y} missing values, filling with mean...")
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
else:
    print("✅ No missing values")

# Show target statistics
print(f"\n📊 Target Statistics:")
for col in target_columns:
    print(f"   {col}:")
    print(f"      Mean: {y[col].mean():.2f}")
    print(f"      Min: {y[col].min():.2f}")
    print(f"      Max: {y[col].max():.2f}")

# ===== STEP 3: Scale features =====
print("\n📈 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled (StandardScaler)")

# ===== STEP 4: Split data =====
print("\n✂️  Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✅ Training set: {len(X_train):,} samples")
print(f"✅ Testing set: {len(X_test):,} samples")

# ===== STEP 5: Train MultiOutput model =====
print("\n🤖 Training MultiOutputRegressor with GradientBoosting...")
print("   (Training 4 targets simultaneously)")

base_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    verbose=0
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# ===== STEP 6: Evaluate each target =====
print("\n📊 Model Evaluation (4 Targets):")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

metrics = {}

for i, target in enumerate(target_columns):
    print(f"\n🎯 Target: {target}")
    
    # Training metrics
    train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
    train_mae = mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i])
    train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i]))
    
    # Testing metrics
    test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
    test_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
    test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i]))
    
    print(f"   Train - R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
    print(f"   Test  - R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    metrics[target] = {
        'train_r2': float(train_r2),
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse)
    }

# ===== STEP 7: Feature importance =====
print(f"\n🔍 Feature Importance (averaged across 4 targets):")

importance_sum = np.zeros(len(feature_columns))
for estimator in model.estimators_:
    importance_sum += estimator.feature_importances_

importance_avg = importance_sum / len(model.estimators_)

feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importance_avg
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_df.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# ===== STEP 8: Save model =====
print("\n💾 Saving model and scaler...")

model_path = MODEL_DIR / "fire_spread_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved: {model_path}")

scaler_path = MODEL_DIR / "fire_spread_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved: {scaler_path}")

# ===== STEP 9: Save metadata =====
print("\n💾 Saving metadata...")

metadata = {
    'model_type': 'MultiOutputRegressor',
    'base_estimator': 'GradientBoostingRegressor',
    'feature_columns': feature_columns,
    'target_columns': target_columns,
    'n_features': len(feature_columns),
    'n_targets': len(target_columns),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'feature_importance': feature_importance_df.to_dict('records'),
    'metrics': metrics,
    'created_at': pd.Timestamp.now().isoformat(),
    'description': 'Fire Spread Model - Predicts spread behavior after ignition. NOT risk prediction.'
}

metadata_path = MODEL_DIR / "fire_spread_model_metadata.pkl"
joblib.dump(metadata, metadata_path)
print(f"✅ Metadata saved: {metadata_path}")

# ===== STEP 10: Test predictions =====
print("\n🧪 Testing predictions on 3 random samples:")

for i in range(min(3, len(X_test))):
    print(f"\n   Sample {i+1}:")
    for j, target in enumerate(target_columns):
        actual = y_test.iloc[i, j]
        predicted = y_test_pred[i, j]
        error = abs(actual - predicted)
        error_pct = (error / actual * 100) if actual != 0 else 0
        
        print(f"      {target}:")
        print(f"         Actual: {actual:.2f}")
        print(f"         Predicted: {predicted:.2f}")
        print(f"         Error: {error:.2f} ({error_pct:.1f}%)")

# ===== SUMMARY =====
print("\n" + "="*70)
print("✅ FIRE SPREAD MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Models saved in: {MODEL_DIR}")
print(f"\nTrained Files:")
print(f"  ✅ fire_spread_model.pkl (MultiOutputRegressor)")
print(f"  ✅ fire_spread_scaler.pkl (StandardScaler)")
print(f"  ✅ fire_spread_model_metadata.pkl (Metrics & config)")
print(f"\n🔥 Model Targets:")
print(f"  1. spread_rate_kmh (how fast)")
print(f"  2. direction_degrees (where)")
print(f"  3. intensity_level (how intense)")
print(f"  4. area_hectares (coverage)")
print(f"\n⚠️  IMPORTANT:")
print(f"   This model is COMPLETELY DIFFERENT from the risk model.")
print(f"   Risk Model: Predicts if fire WILL OCCUR (risk_score)")
print(f"   Spread Model: Predicts HOW FIRE WILL SPREAD (4 targets)")
print("\n")