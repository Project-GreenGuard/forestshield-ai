"""
Heatmap API - Flask REST endpoints for spatial risk visualization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heatmap.risk_predictor import HeatmapPredictor

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = HeatmapPredictor(cache_duration_minutes=30)


@app.route('/heatmap/generate', methods=['POST'])
def generate_heatmap():
    """
    Generate a risk heatmap for a bounding box
    
    POST Body:
        {
            "min_lat": 43.0,
            "max_lat": 44.0,
            "min_lng": -80.0,
            "max_lng": -79.0,
            "resolution": 20,
            "temperature": 28.5,
            "humidity": 35.0,
            "fire_distance_km": 25.0,
            "use_cache": true
        }
    
    Returns:
        Heatmap data with predictions for all grid points
    """
    try:
        data = request.get_json()
        
        # Required parameters
        min_lat = float(data['min_lat'])
        max_lat = float(data['max_lat'])
        min_lng = float(data['min_lng'])
        max_lng = float(data['max_lng'])
        
        # Optional parameters with defaults
        resolution = int(data.get('resolution', 20))
        temperature = float(data.get('temperature', 25.0))
        humidity = float(data.get('humidity', 45.0))
        fire_distance_km = float(data.get('fire_distance_km', 50.0))
        use_cache = data.get('use_cache', True)
        
        # Validate ranges
        if not (2 <= resolution <= 100):
            return jsonify({'error': 'Resolution must be between 2 and 100'}), 400
        
        # Generate heatmap
        result = predictor.generate_heatmap(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lng=min_lng,
            max_lng=max_lng,
            resolution=resolution,
            temperature=temperature,
            humidity=humidity,
            fire_distance_km=fire_distance_km,
            use_cache=use_cache
        )
        
        return jsonify(result), 200
        
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/heatmap/circular', methods=['POST'])
def generate_circular_heatmap():
    """
    Generate a circular heatmap around a center point
    
    POST Body:
        {
            "center_lat": 43.5,
            "center_lng": -79.5,
            "radius_km": 50.0,
            "resolution": 20,
            "temperature": 28.5,
            "humidity": 35.0,
            "fire_distance_km": 25.0
        }
    
    Returns:
        Circular heatmap data
    """
    try:
        data = request.get_json()
        
        # Required parameters
        center_lat = float(data['center_lat'])
        center_lng = float(data['center_lng'])
        radius_km = float(data['radius_km'])
        
        # Optional parameters
        resolution = int(data.get('resolution', 20))
        temperature = float(data.get('temperature', 25.0))
        humidity = float(data.get('humidity', 45.0))
        fire_distance_km = float(data.get('fire_distance_km', 50.0))
        
        # Validate
        if radius_km <= 0 or radius_km > 500:
            return jsonify({'error': 'Radius must be between 0 and 500 km'}), 400
        
        # Generate circular heatmap
        result = predictor.generate_circular_heatmap(
            center_lat=center_lat,
            center_lng=center_lng,
            radius_km=radius_km,
            resolution=resolution,
            temperature=temperature,
            humidity=humidity,
            fire_distance_km=fire_distance_km
        )
        
        return jsonify(result), 200
        
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/heatmap/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = predictor.get_cache_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/heatmap/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cached heatmap data"""
    try:
        predictor.clear_cache()
        return jsonify({'message': 'Cache cleared successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/heatmap/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ForestShield Heatmap API',
        'cache_stats': predictor.get_cache_stats()
    }), 200


if __name__ == '__main__':
    print("🗺️  ForestShield Heatmap API starting...")
    print("📍 Endpoints:")
    print("   POST /heatmap/generate - Generate bounding box heatmap")
    print("   POST /heatmap/circular - Generate circular heatmap")
    print("   GET  /heatmap/cache/stats - Get cache statistics")
    print("   POST /heatmap/cache/clear - Clear cache")
    print("   GET  /heatmap/health - Health check")
    app.run(host='0.0.0.0', port=5001, debug=True)
