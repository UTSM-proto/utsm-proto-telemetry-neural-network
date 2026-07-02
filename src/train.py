"""
train.py
--------
Trains LapEnergyNet on historical lap{N}_distgrid.csv files: feeds each
lap's flattened telemetry through the network, compares the predicted
total energy against the actual recorded energy (MSE loss), and runs
gradient descent to shrink that gap -- this is the "black box" training
loop you described, made concrete.

Usage (from the project root):
    python -m src.train --data-dir data --epochs 200 --val-frac 0.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import LapDistGridDataset, build_model, discover_lap_files


def split_lap_ids(data_dir, val_frac=0.2, seed=42):
    """Randomly split available lap ids into train/val groups."""
    laps = [lid for lid, _ in discover_lap_files(data_dir)]
    rng = np.random.default_rng(seed)
    laps = list(laps)
    rng.shuffle(laps)
    n_val = max(1, int(len(laps) * val_frac))
    return laps[n_val:], laps[:n_val]  # train_ids, val_ids


def train_model(
    data_dir="data",
    results_dir="results",
    epochs=200,
    lr=1e-3,
    batch_size=8,
    val_frac=0.2,
    hidden_dims=(128, 64),
    seed=42,
):
    torch.manual_seed(seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = split_lap_ids(data_dir, val_frac=val_frac, seed=seed)
    train_ds = LapDistGridDataset(data_dir, lap_ids=train_ids)
    # standardize val data using TRAIN statistics, not its own
    val_ds = LapDistGridDataset(
        data_dir, lap_ids=val_ids,
        feature_mean=train_ds.feature_mean, feature_std=train_ds.feature_std,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(train_ds.input_dim, hidden_dims=hidden_dims)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                running += criterion(pred, y).item() * X.size(0)
        val_loss = running / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": train_ds.input_dim,
                    "hidden_dims": hidden_dims,
                    "feature_mean": train_ds.feature_mean,
                    "feature_std": train_ds.feature_std,
                },
                results_dir / "best_model.pt",
            )

        if epoch == 1 or epoch % max(1, epochs // 20) == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}/{epochs}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

    with open(results_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val MSE: {best_val:.4f}")
    print(f"Saved best model to {results_dir / 'best_model.pt'}")
    return model, history


def _parse_args():
    p = argparse.ArgumentParser(description="Train the lap energy prediction network.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
    )