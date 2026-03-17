"""
Simple Vertex AI client for cloud-based inference.
Falls back to local model if Vertex AI is unavailable.
"""

import os
import requests
import json
from typing import Dict, Any
from pathlib import Path

# Local fallback
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.predict import predict_risk as predict_risk_local


class VertexAIClient:
    """Minimal Vertex AI inference wrapper."""
    
    def __init__(self, use_vertex_ai: bool = False, endpoint_url: str = None):
        """
        Args:
            use_vertex_ai: If True, try to use Vertex AI endpoint
            endpoint_url: Vertex AI endpoint URL (from env var or param)
        """
        self.use_vertex_ai = use_vertex_ai
        self.endpoint_url = endpoint_url or os.getenv("VERTEX_AI_ENDPOINT")
        self.use_local_fallback = not (use_vertex_ai and self.endpoint_url)
    
    def predict(self, sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict risk score using Vertex AI or fallback to local model.
        
        Args:
            sensor_payload: Dict with sensor data
        
        Returns:
            Dict with risk_score, risk_level, confidence, model_version
        """
        if self.use_vertex_ai and self.endpoint_url:
            try:
                return self._predict_vertex_ai(sensor_payload)
            except Exception as e:
                print(f"[WARN] Vertex AI failed ({e}), falling back to local model")
                return predict_risk_local(sensor_payload)
        else:
            # Use local model directly
            return predict_risk_local(sensor_payload)
    
    def _predict_vertex_ai(self, sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Vertex AI endpoint (requires manual model deployment).
        
        Expects endpoint to accept JSON with sensor data and return risk prediction.
        """
        headers = {"Content-Type": "application/json"}
        
        # Simple REST call to Vertex AI endpoint
        response = requests.post(
            self.endpoint_url,
            json={"sensor_data": sensor_payload},
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'risk_score': result.get('risk_score', 0),
                'risk_level': result.get('risk_level', 'UNKNOWN'),
                'confidence': result.get('confidence', 0),
                'model_version': result.get('model_version', 'vertex-ai'),
            }
        else:
            raise Exception(f"Vertex AI returned {response.status_code}")


# Example usage
if __name__ == "__main__":
    # Use local model (no Vertex AI set up)
    client = VertexAIClient(use_vertex_ai=False)
    
    test_payload = {
        'temperature': 32.0,
        'humidity': 25.0,
        'lat': 49.2827,
        'lng': -123.1207,
        'nearestFireDistance': 5.0,
        'timestamp': '2024-07-15T15:30:00Z',
    }
    
    result = client.predict(test_payload)
    print(f"Risk Score: {result['risk_score']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']}")