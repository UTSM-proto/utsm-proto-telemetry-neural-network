"""
train.py
--------
Trains a RandomForestRegressor on historical lap{N}_distgrid.csv files.

Unlike the previous neural-network version there are no epochs or a
learning rate -- scikit-learn builds all the trees in a single fit()
call. Training is fast even on small datasets and does not require a
GPU.

After fitting, the model is saved to results/best_model.joblib using
joblib (scikit-learn's recommended serialisation format). Training
metrics (MAE, MSE, R²) are printed for both the training and validation
splits, and a compact results/train_results.json is written for later
reference.

Usage (from the project root):
    python -m src.train --data-dir data --val-frac 0.2
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import LapDistGridDataset, build_model, discover_lap_files


def split_lap_ids(data_dir, val_frac=0.2, seed=42):
    """Randomly split available lap ids into train/val groups."""
    laps = [lid for lid, _ in discover_lap_files(data_dir)]
    rng = np.random.default_rng(seed)
    laps = list(laps)
    rng.shuffle(laps)
    n_val = max(1, int(len(laps) * val_frac))
    return laps[n_val:], laps[:n_val]  # train_ids, val_ids


def _metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:10s}  MAE={mae:.4f}  MSE={mse:.4f}  R²={r2:.4f}")
    return {"mae": mae, "mse": mse, "r2": r2}


def train_model(
    data_dir="data",
    results_dir="results",
    val_frac=0.2,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    seed=42,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = split_lap_ids(data_dir, val_frac=val_frac, seed=seed)
    print(f"Laps: {len(train_ids)} train, {len(val_ids)} val")

    train_ds = LapDistGridDataset(data_dir, lap_ids=train_ids)
    val_ds   = LapDistGridDataset(data_dir, lap_ids=val_ids)

    model = build_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )

    print(f"\nFitting RandomForestRegressor ({n_estimators} trees)...")
    model.fit(train_ds.X, train_ds.y)

    print("\nPerformance:")
    train_m = _metrics(train_ds.y, model.predict(train_ds.X), "Train")
    val_m   = _metrics(val_ds.y,   model.predict(val_ds.X),   "Val")
    print(f"  OOB R²  : {model.oob_score_:.4f}  "
          f"(free estimate using out-of-bag samples)")

    out = {
        "n_train": len(train_ids), "n_val": len(val_ids),
        "train_lap_ids": train_ids, "val_lap_ids": val_ids,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "oob_r2": model.oob_score_,
        "train": train_m, "val": val_m,
        "feature_names": train_ds.feature_names,
        "feature_importances": model.feature_importances_.tolist(),
    }
    with open(results_dir / "train_results.json", "w") as f:
        json.dump(out, f, indent=2)

    model_path = results_dir / "best_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")

    return model, out


def _parse_args():
    p = argparse.ArgumentParser(
        description="Train the Random Forest lap energy model."
    )
    p.add_argument("--data-dir",          default="data")
    p.add_argument("--results-dir",       default="results")
    p.add_argument("--val-frac",          type=float, default=0.2)
    p.add_argument("--n-estimators",      type=int,   default=200,
                   help="Number of trees in the forest.")
    p.add_argument("--max-depth",         type=int,   default=None,
                   help="Max tree depth. Omit to grow fully (may overfit on small datasets).")
    p.add_argument("--min-samples-leaf",  type=int,   default=1,
                   help="Min samples per leaf. Raise to 3-5 to regularise on small datasets.")
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        val_frac=args.val_frac,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )