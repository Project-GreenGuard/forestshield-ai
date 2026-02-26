"""
Training entrypoint for the ForestShield wildfire risk model.

Quick-start
-----------
Generate a synthetic dataset and train a baseline model in one step::

    python training/train.py --generate 5000 --output models/forestshield_v2.joblib

Train from a real CSV export (DynamoDB → CSV)::

    python training/train.py --data path/to/data.csv --output models/forestshield_v2.joblib

Expected CSV columns
--------------------
temperature, humidity, lat, lng, nearestFireDistance, riskScore
(plus optional: deviceId, timestamp)

The ``riskScore`` column is the regression target (0–100).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Tuple

# Ensure repo root is on sys.path when run directly (python training/train.py).
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore
import joblib

# Keep feature engineering in sync with inference/predict.py
from inference.predict import FEATURE_COLUMNS
from utils import rule_based_risk_score

TARGET_COLUMN = "riskScore"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(source: Path | str) -> pd.DataFrame:
    """
    Load historical sensor data from a CSV file.

    Accepted CSV columns (any extra columns are ignored):
    - temperature, humidity, lat, lng, nearestFireDistance, riskScore
    - Optional: deviceId, timestamp (used to derive time features)

    Parameters
    ----------
    source : path to CSV file or directory of CSV files.

    Returns
    -------
    pd.DataFrame with at minimum the columns in ``FEATURE_COLUMNS``
    and ``TARGET_COLUMN``.
    """
    source = Path(source)
    if source.is_dir():
        frames = [pd.read_csv(f) for f in sorted(source.glob("*.csv"))]
        if not frames:
            raise ValueError(f"No CSV files found in {source}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(source)

    # Normalise column names (backend uses camelCase, CSV exports may vary).
    col_map = {
        "nearestfiredistance": "nearestFireDistance",
        "riskscore": "riskScore",
        "deviceid": "deviceId",
    }
    df.columns = [col_map.get(c.lower(), c) for c in df.columns]

    # Derive time features from timestamp if present.
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["hour_of_day"] = ts.dt.hour.astype(float)
        df["month"] = ts.dt.month.astype(float)
    else:
        df.setdefault("hour_of_day", 12.0)
        df.setdefault("month", 6.0)

    # Rename nearestFireDistance → nearest_fire_distance (snake_case used internally).
    if "nearestFireDistance" in df.columns:
        df["nearest_fire_distance"] = pd.to_numeric(
            df["nearestFireDistance"], errors="coerce"
        ).fillna(-1.0)

    # Fill any remaining missing values.
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in data. "
            "Make sure the CSV contains a 'riskScore' column."
        )

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0.0)

    print(f"[train] Loaded {len(df):,} rows from {source}")
    return df


# ---------------------------------------------------------------------------
# Synthetic data generation (for bootstrapping without real data)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset by sampling plausible sensor readings
    and computing labels with the rule-based formula.

    This lets us train a ML model that approximates the existing logic
    before real DynamoDB data is available.
    """
    rng = np.random.default_rng(seed)

    temperature = rng.uniform(5, 45, n_samples)
    humidity = rng.uniform(10, 95, n_samples)
    # British Columbia bounding box (approx).
    lat = rng.uniform(49.0, 60.0, n_samples)
    lng = rng.uniform(-139.0, -114.0, n_samples)
    # ~30 % of readings have a nearby fire.
    fire_mask = rng.random(n_samples) < 0.30
    fire_dist = np.where(fire_mask, rng.uniform(0.5, 80.0, n_samples), -1.0)
    hour_of_day = rng.integers(0, 24, n_samples).astype(float)
    month = rng.integers(1, 13, n_samples).astype(float)

    risk_scores = np.array([
        rule_based_risk_score(
            temperature=t,
            humidity=h,
            fire_distance=d if d >= 0 else None,
        )
        for t, h, d in zip(temperature, humidity, fire_dist)
    ])

    df = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "lat": lat,
        "lng": lng,
        "nearest_fire_distance": fire_dist,
        "hour_of_day": hour_of_day,
        "month": month,
        TARGET_COLUMN: risk_scores,
    })

    print(f"[train] Generated {n_samples:,} synthetic samples")
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(data: pd.DataFrame) -> Tuple[RandomForestRegressor, dict]:
    """
    Train a RandomForestRegressor on the provided dataset.

    Parameters
    ----------
    data : DataFrame returned by ``load_training_data`` or
           ``generate_synthetic_data``.

    Returns
    -------
    (model, metrics) where metrics is a dict with MAE and R² on the
    held-out test split.
    """
    X = data[FEATURE_COLUMNS].values
    y = data[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Attach metadata so inference code can identify the version.
    model.forestshield_version = "v2-rf-regressor"
    model.feature_columns = FEATURE_COLUMNS

    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print(
        f"[train] Training complete — "
        f"MAE: {metrics['mae']:.2f}  R²: {metrics['r2']:.4f}  "
        f"(train={metrics['n_train']:,} / test={metrics['n_test']:,})"
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, output_path: Path | str) -> None:
    """
    Persist the trained model to disk using joblib.

    Parameters
    ----------
    model       : fitted sklearn estimator
    output_path : destination file path (e.g. ``models/forestshield_v2.joblib``)

    Notes
    -----
    For Vertex AI deployment the .joblib file can be uploaded directly as a
    custom prediction artefact (``model.tar.gz``).  Wrap this call with
    the Vertex AI SDK's ``aiplatform.Model.upload()`` when you are ready
    for cloud deployment.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[train] Model saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ForestShield wildfire risk model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data",
        metavar="PATH",
        help="Path to a CSV file (or directory of CSVs) containing historical sensor data.",
    )
    group.add_argument(
        "--generate",
        metavar="N",
        type=int,
        help="Generate N synthetic training samples instead of loading real data.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="models/forestshield_v2.joblib",
        help="Path where the trained model will be saved (default: models/forestshield_v2.joblib).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # 1. Load or generate data.
    if args.generate:
        data = generate_synthetic_data(n_samples=args.generate, seed=args.seed)
    else:
        data = load_training_data(args.data)

    # 2. Train.
    model, metrics = train_model(data)
    print(f"[train] Metrics: {metrics}")

    # 3. Save.
    save_model(model, args.output)
    print("[train] Done.")


if __name__ == "__main__":
    main()

