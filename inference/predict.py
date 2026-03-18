"""
Model inference - loads trained model and makes predictions
"""
import joblib
import numpy as np
from pathlib import Path

# Load trained model
MODEL_PATH = Path(__file__).resolve().parents[1] / "training" / "models" / "wildfire_risk_model.pkl"
METADATA_PATH = Path(__file__).resolve().parents[1] / "training" / "models" / "wildfire_risk_metadata.pkl"

print(f"🔍 Looking for model at: {MODEL_PATH}")

if not MODEL_PATH.exists():
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Try to load metadata
try:
    metadata = joblib.load(METADATA_PATH)
    print(f"✅ Metadata loaded successfully!")
except:
    print(f"⚠️  Metadata file not found, continuing without it")
    metadata = None


def predict_risk(features: np.ndarray) -> dict:
    """
    Predict fire risk from feature vector
    
    Args:
        features: numpy array of shape (11,) or (1, 11)
    
    Returns:
        dict with 'score' (0-100), 'confidence' (0-1), 'level'
    """
    try:
        print(f"🤖 predict_risk called with features shape: {features.shape}")
        
        # Ensure 2D array for sklearn model
        if isinstance(features, list):
            features = np.array(features)
        
        # Convert 1D to 2D if needed
        if features.ndim == 1:
            print(f"Converting 1D array to 2D: {features.shape} → (1, {len(features)})")
            features = features.reshape(1, -1)
        
        print(f"Final features shape for model: {features.shape}")
        print(f"Features: {features}")
        
        # Make prediction
        prediction = model.predict(features)[0]
        print(f"✅ Raw prediction: {prediction}")
        
        # If prediction is 0-1, convert to 0-100
        if prediction <= 1:
            risk_score = prediction * 100
        else:
            risk_score = prediction
        
        # Confidence based on prediction certainty
        confidence = min(0.95, 0.5 + (abs(prediction - 0.5) * 0.9)) if prediction <= 1 else 0.85
        
        # Determine risk level
        if risk_score >= 61:
            risk_level = 'HIGH'
        elif risk_score >= 31:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        result = {
            'score': float(risk_score),
            'confidence': float(confidence),
            'level': risk_level
        }
        
        print(f"✅ Prediction result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error in predict_risk: {e}")
        import traceback
        traceback.print_exc()
        raise
        
        # If prediction is 0-1, convert to 0-100
        if prediction <= 1:
            risk_score = prediction * 100
        else:
            risk_score = prediction
        
        # Confidence based on prediction certainty
        confidence = min(0.95, 0.5 + (abs(prediction - 0.5) * 0.9))
        
        # Determine risk level
        if risk_score >= 61:
            risk_level = 'HIGH'
        elif risk_score >= 31:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        result = {
            'score': float(risk_score),
            'confidence': float(confidence),
            'level': risk_level
        }
        
        print(f"✅ Prediction successful: Risk={risk_score:.1f}, Confidence={confidence:.2f}")
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise


if __name__ == '__main__':
    # Test the model
    print("\n" + "="*60)
    print("Testing model...")
    print("="*60)
    
    test_features = [28.5, 45, 15, 2.3, 1, 200, 2, 0.5, 7, 180, 15]
    result = predict_risk(test_features)
    print(f"Test result: {result}")
    print("="*60 + "\n")