"""
ForestShield wildfire risk model training.
GradientBoostingRegressor on NASA MODIS CSVs (2018-2024), Ontario region.
Usage: python training/train.py [--data-dir ...] [--model-dir ...]
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import joblib, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from utils import FEATURE_COLUMNS, compute_risk_label

_HERE    = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data"
MODEL_DIR= _HERE.parent / "models"
ONTARIO  = {"min_lat": 41.913319, "max_lat": 56.86895,
             "min_lon": -95.154327, "max_lon": -74.324722}
_REQUIRED= {"latitude", "longitude", "bright_t31", "acq_date", "acq_time"}


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
    lat  = pd.to_numeric(df["latitude"],  errors="coerce")
    lng  = pd.to_numeric(df["longitude"], errors="coerce")
    mask = (lat.between(ONTARIO["min_lat"], ONTARIO["max_lat"]) &
            lng.between(ONTARIO["min_lon"], ONTARIO["max_lon"]))
    out  = df[mask].copy()
    print(f"Ontario filter: {len(out):,} / {len(df):,} rows kept")
    if out.empty:
        raise ValueError("No rows remain after Ontario filter.")
    return out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=list(_REQUIRED)).copy()

    # Extract real MODIS-derived features (temperature, location, time)
    base = pd.DataFrame()
    base["temperature"] = pd.to_numeric(df["bright_t31"], errors="coerce") - 273.15
    base["lat"]         = pd.to_numeric(df["latitude"],  errors="coerce")
    base["lng"]         = pd.to_numeric(df["longitude"], errors="coerce")
    acq            = pd.to_datetime(df["acq_date"], errors="coerce")
    base["month"]  = acq.dt.month.astype(float)
    base["hour"]   = (pd.to_numeric(df["acq_time"], errors="coerce")
                      .fillna(0).astype(int).floordiv(100).clip(0, 23).astype(float))
    base = base.dropna().reset_index(drop=True)

    # MODIS has no humidity or fire_distance channels.
    # Augment each row N_AUG times with synthetic values from realistic
    # operational ranges so the model learns all feature contributions.
    N_AUG = 8
    rng   = np.random.default_rng(42)
    n     = len(base)
    frames = []
    for _ in range(N_AUG):
        aug = base.copy()
        aug["humidity"]          = rng.uniform(5.0, 95.0, n)    # % RH sensor range
        aug["nearest_fire_dist"] = rng.uniform(0.5, 150.0, n)  # km, IoT sensor range
        aug["risk_score"]        = aug.apply(compute_risk_label, axis=1)
        frames.append(aug)

    out = pd.concat(frames, ignore_index=True)[FEATURE_COLUMNS + ["risk_score"]]
    print(f"Clean rows: {len(out):,}  |  risk_score mean={out['risk_score'].mean():.1f}")
    return out


def train(df: pd.DataFrame):
    X, y = df[FEATURE_COLUMNS].values, df["risk_score"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_leaf=4, random_state=42)

    print(f"Fitting on {len(X_tr):,} rows ...")
    model.fit(X_tr, y_tr)

    y_pred  = model.predict(X_te)
    rmse    = math.sqrt(mean_squared_error(y_te, y_pred))
    r2      = r2_score(y_te, y_pred)
    cv_rmse = float(-cross_val_score(model, X_tr, y_tr,
                    scoring="neg_root_mean_squared_error", cv=5).mean())
    print(f"  RMSE={rmse:.3f}  R2={r2:.4f}  CV-RMSE={cv_rmse:.3f}")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:22s} {imp:.4f}")

    return model, {"test_rmse": round(rmse, 4), "test_r2": round(r2, 4),
                   "cv_rmse_mean": round(cv_rmse, 4), "train_rows": len(X_tr),
                   "test_rows": len(X_te), "feature_importances": importances}


def save(model, metrics: dict, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "risk_model.joblib")
    meta = {
        "model_version": "v2-ontario-gbr",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "GradientBoostingRegressor",
        "hyperparameters": {"n_estimators": 300, "learning_rate": 0.05,
                            "max_depth": 5, "subsample": 0.8, "min_samples_leaf": 4},
        "feature_columns": FEATURE_COLUMNS,
        "risk_thresholds": {"LOW": [0, 30], "MEDIUM": [31, 60], "HIGH": [61, 100]},
        "region": "Ontario, Canada", "region_bounds": ONTARIO,
        "training_data": "NASA MODIS CSVs 2018-2024 (Ontario)",
        "label_formula": "0.4*temp_score + 0.3*humidity_score + 0.3*fire_score",
        "metrics": metrics,
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved: {model_dir}/risk_model.joblib + model_meta.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  type=Path, default=DATA_DIR)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    a = p.parse_args()

    raw            = load(a.data_dir)
    ontario        = filter_ontario(raw)
    processed      = preprocess(ontario)
    model, metrics = train(processed)
    save(model, metrics, a.model_dir)
    print("Done! Run: python inference/predict.py")


if __name__ == "__main__":
    main()