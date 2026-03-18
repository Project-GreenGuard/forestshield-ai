"""
Generate Fire Spread Labels from Historical Fire Data

Creates realistic spread-specific targets:
- spread_rate_kmh: How fast fire spreads
- direction_degrees: Direction fire travels
- intensity_level: Fire intensity (0-10 scale)
- area_hectares: Area covered

These are DIFFERENT from risk_score (which is used by the RISK model).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def generate_spread_labels(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate fire spread targets from environmental conditions.
    
    Uses realistic physics-informed formulas to create synthetic labels.
    """
    print("\n" + "="*70)
    print("GENERATING FIRE SPREAD LABELS")
    print("="*70)
    
    df = labeled_df.copy()
    
    # ===== SPREAD RATE (km/h) =====
    # Formula: Base speed increased by wind, vegetation; decreased by humidity, moisture
    print("\n📊 Generating spread_rate_kmh...")
    
    base_spread = 0.5
    wind_component = df.get('wind_speed', 10) * 0.3  # Wind accelerates spread
    humidity_component = -df.get('humidity', 50) * 0.005  # Humidity slows it
    vegetation_component = df.get('vegetation_density', 0.7) * 0.2  # Dense vegetation spreads faster
    moisture_component = -df.get('soil_moisture', 0.35) * 0.1  # Wet soil slows spread
    
    df['spread_rate_kmh'] = (
        base_spread + 
        wind_component + 
        humidity_component + 
        vegetation_component + 
        moisture_component +
        np.random.normal(0, 0.3, len(df))  # Add realistic noise
    )
    
    # Clamp to realistic range
    df['spread_rate_kmh'] = df['spread_rate_kmh'].clip(0.5, 12)
    
    print(f"   Mean: {df['spread_rate_kmh'].mean():.2f} km/h")
    print(f"   Range: {df['spread_rate_kmh'].min():.2f} - {df['spread_rate_kmh'].max():.2f}")
    
    # ===== DIRECTION (0-360 degrees) =====
    # Fire spreads in wind direction with some variance
    print("\n📊 Generating direction_degrees...")
    
    wind_direction = df.get('wind_direction', 180)
    variance = np.random.normal(0, 20, len(df))  # ±20° variance
    
    df['direction_degrees'] = (wind_direction + variance) % 360
    df['direction_degrees'] = df['direction_degrees'].clip(0, 360)
    
    print(f"   Mean: {df['direction_degrees'].mean():.1f}°")
    print(f"   Range: {df['direction_degrees'].min():.1f}° - {df['direction_degrees'].max():.1f}°")
    
    # ===== INTENSITY LEVEL (0-10 scale) =====
    # Higher with temperature, vegetation, fire history
    print("\n📊 Generating intensity_level...")
    
    temp_factor = (df.get('temperature', 25) - 15) / (40 - 15) * 3  # 0-3 range
    vegetation_factor = df.get('vegetation_density', 0.7) * 3  # 0-3 range
    fire_history_factor = df.get('fire_history', 3) * 0.3  # Fire history effect
    
    df['intensity_level'] = (
        2 +  # Base intensity
        temp_factor +
        vegetation_factor +
        fire_history_factor +
        np.random.normal(0, 0.5, len(df))
    )
    
    df['intensity_level'] = df['intensity_level'].clip(1, 10)
    
    print(f"   Mean: {df['intensity_level'].mean():.2f}")
    print(f"   Range: {df['intensity_level'].min():.2f} - {df['intensity_level'].max():.2f}")
    
    # ===== AREA COVERED (hectares) =====
    # Coverage = spread_rate × time × intensity factor
    # For 12-hour forecast period
    print("\n📊 Generating area_hectares...")
    
    forecast_hours = 12
    area_hectares = (
        df['spread_rate_kmh'] * 
        forecast_hours * 
        (df['intensity_level'] / 10) +
        np.random.normal(0, 1, len(df))
    )
    
    df['area_hectares'] = area_hectares.clip(0.1, 500)
    
    print(f"   Mean: {df['area_hectares'].mean():.2f} ha")
    print(f"   Range: {df['area_hectares'].min():.2f} - {df['area_hectares'].max():.2f} ha")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("SPREAD LABELS SUMMARY")
    print("="*70)
    print(f"\n✅ Added 4 spread-specific columns to {len(df):,} samples")
    print(f"\nColumns added:")
    print(f"  1. spread_rate_kmh      (how fast)")
    print(f"  2. direction_degrees    (where)")
    print(f"  3. intensity_level      (how intense)")
    print(f"  4. area_hectares        (coverage)")
    print(f"\n⚠️  IMPORTANT:")
    print(f"   These are DIFFERENT from risk_score")
    print(f"   risk_score = fire probability at location")
    print(f"   spread_rate = fire propagation after ignition")
    
    return df


def main():
    """Main execution."""
    # Load labeled training data
    labeled_data_path = Path(__file__).resolve().parent / "labeled_data" / "labeled_training_data.csv"
    
    if not labeled_data_path.exists():
        print(f"❌ File not found: {labeled_data_path}")
        print("Run train.py first to generate labeled_training_data.csv")
        return
    
    print(f"📂 Loading {labeled_data_path}")
    df = pd.read_csv(labeled_data_path)
    
    # Generate spread labels
    df_with_spread = generate_spread_labels(df)
    
    # Save enhanced dataset
    output_path = Path(__file__).resolve().parent / "labeled_data" / "labeled_training_data_with_spread.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_spread.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved enhanced dataset: {output_path}")
    print(f"\nNew columns in dataset:")
    print(df_with_spread[['spread_rate_kmh', 'direction_degrees', 'intensity_level', 'area_hectares']].head())


if __name__ == "__main__":
    main()