"""
evaluate.py
-----------
Loads the trained model (results/best_model.pt) and reports how closely
predicted total lap energy matches actual recorded energy -- per lap, plus
MAE / RMSE / MAPE summary stats.

Usage (from the project root):
    python -m src.evaluate --data-dir data --results-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import LapDistGridDataset, build_model


def load_trained_model(results_dir="results"):
    # weights_only=False: this checkpoint is one we generated ourselves in
    # train.py (it bundles numpy normalization stats alongside the model
    # weights), so the usual "don't unpickle untrusted files" caution doesn't
    # apply here -- only load checkpoints from a results/ dir you trust.
    ckpt = torch.load(Path(results_dir) / "best_model.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["feature_mean"], ckpt["feature_std"]


def evaluate_model(data_dir="data", results_dir="results", lap_ids=None):
    model, feature_mean, feature_std = load_trained_model(results_dir)
    ds = LapDistGridDataset(
        data_dir, lap_ids=lap_ids,
        feature_mean=feature_mean, feature_std=feature_std,
    )

    with torch.no_grad():
        preds = model(ds.X).squeeze(1).numpy()
    actual = ds.y.squeeze(1).numpy()

    abs_err = np.abs(preds - actual)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(((preds - actual) ** 2).mean()))
    mape = float((abs_err / np.clip(np.abs(actual), 1e-8, None)).mean() * 100)

    print(f"{'lap_id':>8} {'predicted':>12} {'actual':>12} {'abs_err':>10}")
    for lid, p, a, e in zip(ds.lap_ids, preds, actual, abs_err):
        print(f"{lid:8d} {p:12.3f} {a:12.3f} {e:10.3f}")

    print(f"\nMAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    out = {
        "lap_ids": ds.lap_ids,
        "predicted": preds.tolist(),
        "actual": actual.tolist(),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
    out_path = Path(results_dir) / "evaluation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved evaluation details to {out_path}")
    return out


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained lap energy model.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(data_dir=args.data_dir, results_dir=args.results_dir)