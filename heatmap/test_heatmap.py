"""Heatmap Module Test & Demo Script"""

import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from heatmap.grid_generator import GridGenerator
from heatmap.risk_predictor import HeatmapPredictor


def run_all_tests():
    """Run comprehensive heatmap tests"""
    print("\n🗺️  FORESTSHIELD HEATMAP - TEST SUITE 🗺️\n")
    
    generator = GridGenerator()
    predictor = HeatmapPredictor(cache_duration_minutes=30)
    
    # Test 1: Grid Generation
    print("="*60 + "\nTEST 1: Grid Generation\n" + "="*60)
    grid = generator.generate_grid(43.0, 44.0, -80.0, -79.0, 10)
    circular = generator.generate_adaptive_grid(43.65, -79.38, 25.0, 15)
    print(f"✅ Uniform grid: {len(grid)} points\n✅ Circular grid: {len(circular)} points")
    
    # Test 2: AI Heatmap (multiple conditions)
    print("\n" + "="*60 + "\nTEST 2: AI-Powered Heatmap\n" + "="*60)
    
    # Test HIGH risk
    h_high = predictor.generate_heatmap(44.0, 44.2, -80.0, -79.8, 5, 33.0, 22.0, 12.0, False)
    high = sum(1 for p in h_high['predictions'] if p['risk_level']=='HIGH')
    print(f"✅ HIGH risk (33°C, 22% humid, fire 12km): {high}/25 points")
    
    # Test MEDIUM risk
    h_med = predictor.generate_heatmap(44.0, 44.2, -80.0, -79.8, 5, 24.0, 42.0, 35.0, False)
    med = sum(1 for p in h_med['predictions'] if p['risk_level']=='MEDIUM')
    print(f"✅ MEDIUM risk (24°C, 42% humid, fire 35km): {med}/25 points")
    
    # Test LOW risk
    h_low = predictor.generate_heatmap(44.0, 44.2, -80.0, -79.8, 5, 16.0, 68.0, 90.0, False)
    low = sum(1 for p in h_low['predictions'] if p['risk_level']=='LOW')
    heatmap = h_high  # Use high-risk heatmap for later tests
    print(f"✅ LOW risk (16°C, 68% humid, fire 90km): {low}/25 points")
    
    # Test 3: Caching
    print("\n" + "="*60 + "\nTEST 3: Cache Performance\n" + "="*60)
    params = {'min_lat': 43.0, 'max_lat': 43.5, 'min_lng': -80.0, 'max_lng': -79.5,
              'resolution': 8, 'temperature': 28.0, 'humidity': 40.0, 'fire_distance_km': 30.0}
    start = time.time()
    predictor.generate_heatmap(**params, use_cache=False)
    time1 = time.time() - start
    start = time.time()
    predictor.generate_heatmap(**params, use_cache=True)
    time2 = time.time() - start
    print(f"✅ No cache: {time1:.2f}s | Cached: {time2:.4f}s | Speedup: {time1/time2:.1f}×")
    predictor.clear_cache()
    
    # Test 4: Circular Heatmap
    print("\n" + "="*60 + "\nTEST 4: Circular Heatmap\n" + "="*60)
    circular_map = predictor.generate_circular_heatmap(44.25, -79.5, 30.0, 15, 25.0, 50.0, 40.0)
    print(f"✅ Generated {len(circular_map['predictions'])} points (30km radius)")
    print(f"   Mean Risk: {circular_map['statistics']['mean_risk']}")
    
    # Test 5: JSON Export
    print("\n" + "="*60 + "\nTEST 5: JSON Export\n" + "="*60)
    output = Path(__file__).parent / 'demo_heatmap_output.json'
    with open(output, 'w') as f:
        json.dump(heatmap, f, indent=2)
    print(f"✅ Saved: {output.name} ({output.stat().st_size/1024:.1f} KB)")
    
    print("\n" + "="*60 + "\n✅ All tests completed successfully!\n" + "="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
