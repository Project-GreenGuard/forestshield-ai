"""
ForestShield AI Analytics - API Integration

Flask endpoints for accessing historical analytics and performance metrics.
"""

from flask import Flask, jsonify, request
import sys
from pathlib import Path

# Add AI module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.performance_tracker import PerformanceTracker

app = Flask(__name__)
tracker = PerformanceTracker()


@app.route('/analytics/trends', methods=['GET'])
def get_trends():
    """
    Get risk trends over time.
    
    Query params:
        hours: Hours to look back (default 24)
        device_id: Optional device filter
    """
    hours = int(request.args.get('hours', 24))
    device_id = request.args.get('device_id')
    
    trends = tracker.get_risk_trends(hours=hours, device_id=device_id)
    return jsonify(trends)


@app.route('/analytics/performance', methods=['GET'])
def get_performance():
    """
    Get model performance metrics.
    
    Query params:
        hours: Hours to look back (default 168 = 1 week)
    """
    hours = int(request.args.get('hours', 168))
    
    metrics = tracker.get_performance_metrics(hours=hours)
    return jsonify(metrics)


@app.route('/analytics/drift', methods=['GET'])
def check_drift():
    """
    Check for model drift.
    
    Query params:
        baseline_hours: Baseline period hours (default 168)
        recent_hours: Recent period hours (default 24)
    """
    baseline_hours = int(request.args.get('baseline_hours', 168))
    recent_hours = int(request.args.get('recent_hours', 24))
    
    drift_analysis = tracker.detect_model_drift(
        baseline_hours=baseline_hours,
        recent_hours=recent_hours
    )
    return jsonify(drift_analysis)


@app.route('/analytics/summary', methods=['GET'])
def get_summary():
    """Get comprehensive analytics summary."""
    return jsonify({
        'trends_24h': tracker.get_risk_trends(hours=24),
        'performance_weekly': tracker.get_performance_metrics(hours=168),
        'drift_check': tracker.detect_model_drift()
    })


if __name__ == '__main__':
    print("Analytics API starting on http://localhost:5001")
    print("\nEndpoints:")
    print("  GET /analytics/trends?hours=24&device_id=esp32-01")
    print("  GET /analytics/performance?hours=168")
    print("  GET /analytics/drift")
    print("  GET /analytics/summary")
    app.run(host='0.0.0.0', port=5001, debug=True)
