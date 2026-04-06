"""
ForestShield wildfire risk model training.
GradientBoostingRegressor on NASA MODIS CSVs (2018-2024), Ontario region.
Usage: python training/train.py [--data-dir .] [--model-dir .]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from utils import FEATURE_COLUMNS, compute_risk_label

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data"
MODEL_DIR = _HERE.parent / "models"

ONTARIO = {
    "min_lat": 41.913319,
    "max_lat": 56.86895,
    "min_lon": -95.154327,
    "max_lon": -74.324722,
}

_REQUIRED = {"latitude", "longitude", "bright_t31", "acq_date", "acq_time"}


def load(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        print(f"Loaded {f.name}: {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Total loaded: {len(combined):,} rows from {len(files)} file(s)")
    return combined


def filter_ontario(df: pd.DataFrame) -> pd.DataFrame:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lng = pd.to_numeric(df["longitude"], errors="coerce")

    mask = (
        lat.between(ONTARIO["min_lat"], ONTARIO["max_lat"])
        & lng.between(ONTARIO["min_lon"], ONTARIO["max_lon"])
    )

    out = df[mask].copy()
    print(f"Ontario filter: kept {len(out):,} / {len(df):,} rows")

    if out.empty:
        raise ValueError("No rows remain after Ontario filter.")

    return out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(_REQUIRED)).copy()

    base = pd.DataFrame()

    # MODIS bright_t31 is Kelvin-ish thermal brightness; convert to Celsius proxy
    base["temperature"] = (
        pd.to_numeric(df["bright_t31"], errors="coerce") - 273.15
    ).clip(0.0, 50.0)

    base["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    base["lng"] = pd.to_numeric(df["longitude"], errors="coerce")

    acq = pd.to_datetime(df["acq_date"], errors="coerce")
    base["month"] = acq.dt.month.astype(float)

    base["hour"] = (
        pd.to_numeric(df["acq_time"], errors="coerce")
        .fillna(0)
        .astype(int)
        .floordiv(100)
        .clip(0, 23)
        .astype(float)
    )

    base = base.dropna().reset_index(drop=True)

    if base.empty:
        raise ValueError("No usable rows after preprocessing.")

    rng = np.random.default_rng(42)

    # To keep training balanced and manageable
    max_rows = min(len(base), 5000)
    base = base.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # -----------------------------
    # HIGH RISK: hot + dry + close
    # -----------------------------
    high = base.copy()
    high["temperature"] = np.clip(
        high["temperature"] + rng.uniform(8.0, 18.0, len(high)),
        28.0,
        50.0,
    )
    high["humidity"] = rng.uniform(15.0, 30.0, len(high))
    high["nearest_fire_dist"] = rng.uniform(0.5, 8.0, len(high))

    # -----------------------------------
    # MEDIUM RISK: moderate conditions
    # -----------------------------------
    medium = base.copy()
    medium["temperature"] = np.clip(
        medium["temperature"] + rng.uniform(-2.0, 8.0, len(medium)),
        18.0,
        35.0,
    )
    medium["humidity"] = rng.uniform(31.0, 60.0, len(medium))
    medium["nearest_fire_dist"] = rng.uniform(15.0, 50.0, len(medium))

    # -----------------------------
    # LOW RISK: cool + humid + far
    # -----------------------------
    low = base.copy()
    low["temperature"] = np.clip(
        low["temperature"] + rng.uniform(-12.0, 0.0, len(low)),
        5.0,
        25.0,
    )
    low["humidity"] = rng.uniform(61.0, 90.0, len(low))
    low["nearest_fire_dist"] = rng.uniform(60.0, 150.0, len(low))

    # Combine all samples
    out = pd.concat([high, medium, low], ignore_index=True)

    # Compute labels using the SAME logic as utils.py
    out["risk_score"] = out.apply(compute_risk_label, axis=1)

    # Force class bands to remain aligned with your desired ranges
    high_mask = out.index < len(high)
    medium_mask = (out.index >= len(high)) & (out.index < len(high) + len(medium))
    low_mask = out.index >= (len(high) + len(medium))

    out.loc[high_mask, "risk_score"] = out.loc[high_mask, "risk_score"].clip(61.0, 100.0)
    out.loc[medium_mask, "risk_score"] = out.loc[medium_mask, "risk_score"].clip(31.0, 60.0)
    out.loc[low_mask, "risk_score"] = out.loc[low_mask, "risk_score"].clip(0.0, 30.0)

    out = out.sample(frac=1, random_state=42).reset_index(drop=True)
    out = out[FEATURE_COLUMNS + ["risk_score"]]

    print(f"Training rows after preprocessing: {len(out):,}")
    print(
        f"Risk score stats -> min={out['risk_score'].min():.1f}, "
        f"max={out['risk_score'].max():.1f}, "
        f"mean={out['risk_score'].mean():.1f}"
    )

    return out


def train(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].values
    y = df["risk_score"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=4,
        random_state=42,
    )

    print(f"Training model on {len(X_tr):,} rows...")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    rmse = math.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)

    cv_rmse = float(
        -cross_val_score(
            model,
            X_tr,
            y_tr,
            scoring="neg_root_mean_squared_error",
            cv=5,
        ).mean()
    )

    print(f"RMSE={rmse:.3f} | R2={r2:.4f} | CV-RMSE={cv_rmse:.3f}")

    importances = dict(
        zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist())
    )

    print("Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:20s} {imp:.4f}")

    return model, {
        "test_rmse": round(rmse, 4),
        "test_r2": round(r2, 4),
        "cv_rmse_mean": round(cv_rmse, 4),
        "train_rows": len(X_tr),
        "test_rows": len(X_te),
        "feature_importances": importances,
    }


def save(model, metrics: dict, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "risk_model.joblib")

    meta = {
        "model_version": "v3-ontario-gbr-realistic-bands",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "GradientBoostingRegressor",
        "hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "min_samples_leaf": 4,
        },
        "feature_columns": FEATURE_COLUMNS,
        "risk_thresholds": {
            "LOW": [0, 30],
            "MEDIUM": [31, 60],
            "HIGH": [61, 100],
        },
        "region": "Ontario, Canada",
        "region_bounds": ONTARIO,
        "training_data": (
            "NASA MODIS wildfire detections (Ontario, 2018-2024) with "
            "generated low-, medium-, and high-risk samples using realistic "
            "temperature, humidity, and nearest-fire-distance ranges"
        ),
        "label_method": (
            "Risk scores computed with utils.compute_risk_label() to stay aligned "
            "with inference logic, then clipped to LOW/MEDIUM/HIGH training bands"
        ),
        "metrics": metrics,
    }

    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to: {model_dir / 'risk_model.joblib'}")
    print(f"Saved metadata to: {model_dir / 'model_meta.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    raw = load(args.data_dir)
    ontario = filter_ontario(raw)
    processed = preprocess(ontario)
    model, metrics = train(processed)
    save(model, metrics, args.model_dir)

    print("Done. You can now run inference with the updated model.")


if __name__ == "__main__":
    main()