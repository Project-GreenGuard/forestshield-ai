"""
Train Fire Spread Model on labeled data.
Predicts fire spread rate based on environmental factors.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🔥 FIRE SPREAD MODEL TRAINING")
print("="*70)

# ===== SETUP: Create models directory =====
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
print(f"\n📂 Model directory: {MODEL_DIR}")

# ===== STEP 1: Load data =====
print("\n📂 Loading labeled training data...")
data_path = Path(__file__).resolve().parent / "labeled_data" / "labeled_training_data.csv"
df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df)} fire records")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData preview:")
print(df.head())

# ===== STEP 2: Prepare features =====
print("\n🔧 Preparing features...")

# Select features for fire spread prediction
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

# Target: risk_score (higher = faster spread)
X = df[feature_columns].copy()
y = df['risk_score'].copy()

print(f"✅ Features: {feature_columns}")
print(f"✅ Target: risk_score")
print(f"   - Mean: {y.mean():.2f}")
print(f"   - Std: {y.std():.2f}")
print(f"   - Min: {y.min():.2f}")
print(f"   - Max: {y.max():.2f}")

# Check for missing values
print(f"\n📊 Checking data quality:")
missing = X.isnull().sum()
if missing.sum() > 0:
    print(f"⚠️  Missing values:\n{missing[missing > 0]}")
    X = X.fillna(X.mean())
    print("   Filled with mean values")
else:
    print("✅ No missing values")

# ===== STEP 3: Scale features =====
print("\n📈 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled (StandardScaler)")

# ===== STEP 4: Split data =====
print("\n✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Testing set: {len(X_test)} samples")

# ===== STEP 5: Train model =====
print("\n🤖 Training GradientBoostingRegressor...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)
print("✅ Model training complete!")

# ===== STEP 6: Evaluate =====
print("\n📊 Model Evaluation:")

# Training performance
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f"\n🎯 Training Performance:")
print(f"   R² Score: {train_r2:.4f}")
print(f"   MAE: {train_mae:.4f}")
print(f"   RMSE: {train_rmse:.4f}")

# Testing performance
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n🎯 Testing Performance:")
print(f"   R² Score: {test_r2:.4f}")
print(f"   MAE: {test_mae:.4f}")
print(f"   RMSE: {test_rmse:.4f}")

# Feature importance
print(f"\n🔍 Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# ===== STEP 7: Save model =====
print("\n💾 Saving model...")

# ✅ FIX: Use absolute path
model_path = MODEL_DIR / "fire_spread_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved: {model_path}")

# Save scaler
scaler_path = MODEL_DIR / "fire_spread_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved: {scaler_path}")

# Save metadata
metadata = {
    'feature_columns': feature_columns,
    'target': 'risk_score',
    'model_type': 'GradientBoostingRegressor',
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'feature_importance': feature_importance.to_dict('records'),
    'created_at': pd.Timestamp.now().isoformat()
}

metadata_path = MODEL_DIR / "fire_spread_model_metadata.pkl"
joblib.dump(metadata, metadata_path)
print(f"✅ Metadata saved: {metadata_path}")

# ===== STEP 8: Test predictions =====
print("\n🧪 Testing predictions...")

# Test on first 5 test samples
for i in range(min(5, len(X_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    error = abs(actual - predicted)
    
    print(f"   Sample {i+1}:")
    print(f"      Actual: {actual:.2f}")
    print(f"      Predicted: {predicted:.2f}")
    print(f"      Error: {error:.2f}")

print("\n" + "="*70)
print("✅ FIRE SPREAD MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Models saved in: {MODEL_DIR}")
print("\nModel ready for production use in:")
print("  - fire_spread_api.py")
print("  - local_dashboard.py")
print("\n")