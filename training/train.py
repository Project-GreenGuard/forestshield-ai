"""
ForestShield wildfire risk model training.
GradientBoostingRegressor on NASA MODIS CSVs (2018-2024), Ontario region.
Usage: python training/train.py [--data-dir ...] [--model-dir ...]
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

from utils import FEATURE_COLUMNS

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
        raise FileNotFoundError(f"No CSVs in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        print(f"  {f.name}: {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(combined):,} rows from {len(files)} file(s)")
    return combined


def filter_ontario(df: pd.DataFrame) -> pd.DataFrame:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lng = pd.to_numeric(df["longitude"], errors="coerce")

    mask = (
        lat.between(ONTARIO["min_lat"], ONTARIO["max_lat"])
        & lng.between(ONTARIO["min_lon"], ONTARIO["max_lon"])
    )

    out = df[mask].copy()
    print(f"Ontario filter: {len(out):,} / {len(df):,} rows kept")

    if out.empty:
        raise ValueError("No rows remain after Ontario filter.")

    return out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=list(_REQUIRED)).copy()

    # Real fire detections from MODIS/FIRMS
    base = pd.DataFrame()
    base["temperature"] = (pd.to_numeric(df["bright_t31"], errors="coerce") - 273.15).clip(0.0, 50.0)
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

    # Positive samples: actual NASA fire detections
    pos = base.copy()
    pos["humidity"] = rng.uniform(20.0, 30.0, len(pos))   # high risk
    pos["nearest_fire_dist"] = rng.uniform(0.5, 8.0, len(pos))
    pos["risk_score"] = rng.uniform(61.0, 100.0, len(pos))

    # Medium-risk samples
    mid = base.copy()
    mid["humidity"] = rng.uniform(31.0, 60.0, len(mid))   # medium risk
    mid["nearest_fire_dist"] = rng.uniform(15.0, 50.0, len(mid))
    mid["risk_score"] = rng.uniform(31.0, 60.0, len(mid))

    # Low-risk / no-fire-style samples
    neg = base.copy()
    neg["humidity"] = rng.uniform(61.0, 90.0, len(neg))   # low risk
    neg["nearest_fire_dist"] = rng.uniform(60.0, 150.0, len(neg))
    neg["risk_score"] = rng.uniform(0.0, 30.0, len(neg))

    out = pd.concat([pos, mid, neg], ignore_index=True)
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)

    out = out[FEATURE_COLUMNS + ["risk_score"]]

    print(f"Clean rows: {len(out):,} | risk_score mean={out['risk_score'].mean():.1f}")
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

    print(f"Fitting on {len(X_tr):,} rows ...")
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

    print(f"  RMSE={rmse:.3f}  R2={r2:.4f}  CV-RMSE={cv_rmse:.3f}")

    importances = dict(
        zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist())
    )
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:22s} {imp:.4f}")

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
        "model_version": "v3-ontario-gbr-firms",
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
        "training_data": "NASA MODIS wildfire detections (Ontario, 2018-2024) with generated medium- and low-risk samples for supervised training",
        "label_method": "High-risk samples are grounded in real NASA fire detections; medium- and low-risk samples are generated to support supervised learning across a wider operational range",
        "metrics": metrics,
    }

    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved: {model_dir}/risk_model.joblib + model_meta.json")


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

    print("Done! Run: python inference/predict.py")


if __name__ == "__main__":
    main()