"""
ForestShield AI - Data Preparation Module

Handles loading NASA MODIS fire data and generating synthetic sensor observations
for training the wildfire risk prediction model.

References:
- AI_PREDICTION_AND_TRAINING_SPEC.md Phase 1: Use rule-based risk scores as labels
- NASA FIRMS MODIS data: https://firms.modaps.eosdis.nasa.gov/
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.features import build_feature_vector, FEATURE_NAMES


def load_nasa_data(data_dir: Path, years: list = None) -> pd.DataFrame:
    """
    Load NASA MODIS fire detection data from CSV files.
    
    Args:
        data_dir: Directory containing modis_YYYY_Canada.csv files
        years: List of years to load (e.g., [2023, 2024]). If None, loads all.
    
    Returns:
        Combined DataFrame with all fire detections
    """
    print("\n" + "="*70)
    print("LOADING NASA MODIS DATA")
    print("="*70)
    
    dfs = []
    csv_files = sorted(data_dir.glob("modis_*_Canada.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No NASA CSV files found in {data_dir}")
    
    for csv_file in csv_files:
        # Extract year from filename
        year = int(csv_file.stem.split('_')[1])
        
        if years is not None and year not in years:
            continue
        
        print(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        df['year'] = year
        dfs.append(df)
        print(f"  ✓ {len(df):,} fire detections")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Total fires loaded: {len(combined):,}")
    
    return combined


def generate_training_samples(fires_df: pd.DataFrame, samples_per_fire: int = 20) -> pd.DataFrame:
    """
    Generate synthetic sensor observations around real fire locations.
    
    For each fire, creates sensors at various distances with environmental
    conditions that correlate with fire proximity.
    
    Strategy:
    - Places synthetic sensors at exponentially-distributed distances (0.5-100 km)
    - Generates realistic temperature/humidity based on fire characteristics
    - Assigns rule-based risk scores as training labels (Phase 1 approach)
    
    Args:
        fires_df: DataFrame with NASA fire detections
        samples_per_fire: Number of synthetic sensors per fire
    
    Returns:
        DataFrame with sensor observations and risk scores
    """
    print("\n" + "="*70)
    print("GENERATING TRAINING SAMPLES")
    print("="*70)
    print(f"Fires: {len(fires_df):,}")
    print(f"Samples per fire: {samples_per_fire}")
    
    # Sample fires to avoid excessive data (use up to 5000 fires)
    if len(fires_df) > 5000:
        print(f"Sampling 5000 fires from {len(fires_df):,}...")
        fires_df = fires_df.sample(n=5000, random_state=42)
    
    samples = []
    
    for idx, fire in fires_df.iterrows():
        fire_lat = fire['latitude']
        fire_lon = fire['longitude']
        fire_brightness = fire['brightness']
        fire_frp = fire['frp'] if pd.notna(fire['frp']) else 0
        
        # Parse timestamp
        try:
            timestamp = f"{fire['acq_date']}T{str(fire['acq_time']).zfill(4)[:2]}:{str(fire['acq_time']).zfill(4)[2:]}:00Z"
        except:
            timestamp = "2024-06-15T14:00:00Z"  # Default summer afternoon
        
        # Generate samples at various distances
        for _ in range(samples_per_fire):
            # Distance: more samples close to fire (exponential distribution)
            distance_km = np.random.exponential(25)
            distance_km = np.clip(distance_km, 0.5, 100)
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Calculate sensor position (rough approximation)
            lat_offset = (distance_km / 111.0) * np.cos(angle)
            lon_offset = (distance_km / (111.0 * np.cos(np.radians(fire_lat)))) * np.sin(angle)
            
            sensor_lat = fire_lat + lat_offset
            sensor_lon = fire_lon + lon_offset
            
            # Environmental conditions based on fire and distance
            # Closer = hotter, drier
            base_temp = 20 + (fire_brightness - 300) / 20  # Scale brightness to temp
            temp_decrease = distance_km * 0.15
            temperature = base_temp - temp_decrease + np.random.normal(0, 3)
            temperature = np.clip(temperature, 15, 45)
            
            base_humidity = 60 - (fire_frp / 30)
            humidity_increase = distance_km * 0.2
            humidity = base_humidity + humidity_increase + np.random.normal(0, 8)
            humidity = np.clip(humidity, 20, 80)
            
            # Risk score based on distance (rule-based labels for Phase 1)
            if distance_km < 5:
                risk_base = 85
            elif distance_km < 15:
                risk_base = 65
            elif distance_km < 30:
                risk_base = 40
            elif distance_km < 50:
                risk_base = 25
            else:
                risk_base = 15
            
            # Adjust for fire intensity
            risk_frp_adj = np.clip(fire_frp / 30, 0, 10)
            risk_score = risk_base + risk_frp_adj + np.random.normal(0, 5)
            risk_score = np.clip(risk_score, 0, 100)
            
            sample = {
                'temperature': temperature,
                'humidity': humidity,
                'lat': sensor_lat,
                'lng': sensor_lon,
                'nearestFireDistance': distance_km,
                'timestamp': timestamp,
                'risk_score': risk_score,
            }
            samples.append(sample)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} fires...")
    
    df = pd.DataFrame(samples)
    print(f"\n✓ Generated {len(df):,} training samples")
    print(f"  Risk range: {df['risk_score'].min():.1f} - {df['risk_score'].max():.1f}")
    print(f"  Mean risk: {df['risk_score'].mean():.1f}")
    
    return df


def prepare_features(samples_df: pd.DataFrame) -> tuple:
    """
    Convert samples to feature matrix using 11-feature contract.
    
    Builds feature vectors using utils.features.build_feature_vector(),
    which applies the locked 11-feature transformation.
    
    Args:
        samples_df: DataFrame with sensor observations
    
    Returns:
        X: Feature matrix (N, 11)
        y: Risk scores (N,)
    """
    print("\n" + "="*70)
    print("BUILDING FEATURES")
    print("="*70)
    
    X_list = []
    y_list = []
    
    for idx, row in samples_df.iterrows():
        payload = row.to_dict()
        
        try:
            features = build_feature_vector(payload)
            X_list.append(features)
            y_list.append(payload['risk_score'])
        except Exception as e:
            # Skip invalid samples
            continue
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Features: {', '.join(FEATURE_NAMES[:5])}... (+6 more)")
    
    return X, y