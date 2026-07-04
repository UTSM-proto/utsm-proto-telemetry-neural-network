"""
evaluate.py
-----------
Loads the trained model (results/best_model.pt) and reports how closely
predicted total lap energy matches actual recorded energy -- per lap, plus
MAE / RMSE / MAPE summary stats.

Also saves two plots to the results directory:
  evaluation_plot.png   -- two-panel predicted-vs-actual summary:
    Left panel  : scatter of predicted vs. actual energy per lap, with a
                  perfect-prediction reference line (y = x).
    Right panel : bar chart of absolute error per lap, with a dashed MAE line.
  learning_curve.png    -- train vs. validation MSE loss over every epoch,
                  loaded from train_history.json written by train.py. Lets
                  you see at a glance whether the model is still learning,
                  has converged, or is overfitting.

Usage (from the project root):
    python -m src.evaluate --data-dir data --results-dir results
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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

    plot_path = plot_results(ds.lap_ids, preds, actual, mae, rmse, mape, results_dir)
    print(f"Saved evaluation plot to {plot_path}")

    curve_path = plot_learning_curve(results_dir)
    if curve_path:
        print(f"Saved learning curve to {curve_path}")

    return out


def plot_results(lap_ids, preds, actual, mae, rmse, mape, results_dir="results"):
    """Save a two-panel predicted-vs-actual plot to results/evaluation_plot.png."""
    lap_labels = [f"Lap {lid}" for lid in lap_ids]
    x = np.arange(len(lap_ids))

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Lap Energy Prediction — Evaluation", fontsize=13, fontweight="bold")

    # --- Left panel: predicted vs. actual scatter ---
    lo = min(actual.min(), preds.min()) * 0.97
    hi = max(actual.max(), preds.max()) * 1.03

    ax_scatter.plot([lo, hi], [lo, hi], color="grey", linewidth=1,
                    linestyle="--", label="Perfect prediction")
    ax_scatter.scatter(actual, preds, color="#1f77b4", s=60, zorder=3)

    # label each point with its lap id
    for lid, a, p in zip(lap_ids, actual, preds):
        ax_scatter.annotate(f"Lap {lid}", (a, p),
                            textcoords="offset points", xytext=(6, 3),
                            fontsize=7, color="#444444")

    ax_scatter.set_xlabel("Actual energy (Wh)")
    ax_scatter.set_ylabel("Predicted energy (Wh)")
    ax_scatter.set_title("Predicted vs. Actual")
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.legend(fontsize=8)
    ax_scatter.text(
        0.03, 0.97,
        f"MAE {mae:.2f} Wh\nRMSE {rmse:.2f} Wh\nMAPE {mape:.1f}%",
        transform=ax_scatter.transAxes,
        va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
    )

    # --- Right panel: absolute error per lap bar chart ---
    abs_err = np.abs(preds - actual)
    bar_colors = ["#d62728" if e > mae else "#1f77b4" for e in abs_err]
    ax_bar.bar(x, abs_err, color=bar_colors, width=0.6, zorder=3)
    ax_bar.axhline(mae, color="#ff7f0e", linewidth=1.5,
                   linestyle="--", label=f"MAE ({mae:.2f} Wh)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(lap_labels, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("Absolute error (Wh)")
    ax_bar.set_title("Absolute Error per Lap")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    # colour legend
    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[
            plt.Line2D([0], [0], color="#ff7f0e", linestyle="--", linewidth=1.5,
                       label=f"MAE ({mae:.2f} Wh)"),
            Patch(facecolor="#d62728", label="Above MAE"),
            Patch(facecolor="#1f77b4", label="At or below MAE"),
        ],
        fontsize=8,
    )

    fig.tight_layout()
    plot_path = Path(results_dir) / "evaluation_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_learning_curve(results_dir="results"):
    """Load train_history.json and save a train vs. validation loss curve.

    Returns the output path, or None if train_history.json does not exist
    (e.g. if evaluate is run standalone without a preceding train step).
    """
    history_path = Path(results_dir) / "train_history.json"
    if not history_path.exists():
        print(f"  (skipping learning curve — {history_path} not found)")
        return None

    with open(history_path) as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    # Find the epoch where validation loss was lowest (where best_model.pt was saved)
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = min(val_loss)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=1.8, label="Train")
    ax.plot(epochs, val_loss,   color="#ff7f0e", linewidth=1.8, label="Validation")

    # Mark the best checkpoint
    ax.axvline(best_epoch, color="grey", linewidth=1, linestyle=":",
               label=f"Best checkpoint (epoch {best_epoch})")
    ax.scatter([best_epoch], [best_val], color="#ff7f0e", s=60, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Train vs Validation Loss")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = Path(results_dir) / "learning_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained lap energy model.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(data_dir=args.data_dir, results_dir=args.results_dir)