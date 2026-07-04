"""
evaluate.py
-----------
Loads the trained model (results/best_model.joblib) and reports how
closely predicted total lap energy matches actual recorded energy --
per lap, plus MAE / MSE / R² / MAPE summary stats.

Saves two plots to the results directory:

  evaluation_plot.png   -- two-panel predicted-vs-actual summary:
    Left panel  : scatter of predicted vs. actual energy per lap, with a
                  perfect-prediction reference line (y = x).
    Right panel : bar chart of absolute error per lap, with a dashed
                  MAE reference line.

  feature_importance.png -- horizontal bar chart of each input feature's
                  contribution to the model's predictions, ranked by
                  importance. Replaces the epoch-based learning curve
                  from the neural network version (Random Forests don't
                  train iteratively, so there is no loss-per-epoch curve
                  to plot). This chart answers the question: which
                  sensor / statistic matters most for predicting energy?

Usage (from the project root):
    python -m src.evaluate --data-dir data --results-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import LapDistGridDataset


def load_trained_model(results_dir="results"):
    return joblib.load(Path(results_dir) / "best_model.joblib")


def evaluate_model(data_dir="data", results_dir="results", lap_ids=None):
    model = load_trained_model(results_dir)
    ds = LapDistGridDataset(data_dir, lap_ids=lap_ids)

    preds  = model.predict(ds.X)
    actual = ds.y

    abs_err = np.abs(preds - actual)
    mae  = float(mean_absolute_error(actual, preds))
    mse  = float(mean_squared_error(actual, preds))
    r2   = float(r2_score(actual, preds))
    mape = float((abs_err / np.clip(np.abs(actual), 1e-8, None)).mean() * 100)

    print(f"{'lap_id':>8} {'predicted':>12} {'actual':>12} {'abs_err':>10}")
    for lid, p, a, e in zip(ds.lap_ids, preds, actual, abs_err):
        print(f"{lid:8d} {p:12.3f} {a:12.3f} {e:10.3f}")

    print(f"\nMAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    out = {
        "lap_ids": ds.lap_ids,
        "predicted": preds.tolist(),
        "actual": actual.tolist(),
        "mae": mae, "mse": mse, "r2": r2, "mape": mape,
    }
    out_path = Path(results_dir) / "evaluation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved evaluation details to {out_path}")

    plot_path = plot_results(ds.lap_ids, preds, actual, mae, mse, r2, mape, results_dir)
    print(f"Saved evaluation plot to {plot_path}")

    fi_path = plot_feature_importance(model, ds.feature_names, results_dir)
    print(f"Saved feature importance chart to {fi_path}")

    return out


def plot_results(lap_ids, preds, actual, mae, mse, r2, mape, results_dir="results"):
    """Save a two-panel predicted-vs-actual plot to results/evaluation_plot.png."""
    from matplotlib.patches import Patch

    lap_labels = [f"Lap {lid}" for lid in lap_ids]
    x = np.arange(len(lap_ids))

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Lap Energy Prediction — Evaluation", fontsize=13, fontweight="bold")

    # --- Left: predicted vs. actual scatter ---
    lo = min(actual.min(), preds.min()) * 0.97
    hi = max(actual.max(), preds.max()) * 1.03

    ax_scatter.plot([lo, hi], [lo, hi], color="grey", linewidth=1,
                    linestyle="--", label="Perfect prediction")
    ax_scatter.scatter(actual, preds, color="#1f77b4", s=60, zorder=3)
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
        f"MAE {mae:.2f} Wh\nMSE {mse:.2f}\nR² {r2:.4f}\nMAPE {mape:.1f}%",
        transform=ax_scatter.transAxes,
        va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
    )

    # --- Right: absolute error per lap ---
    abs_err = np.abs(preds - actual)
    bar_colors = ["#d62728" if e > mae else "#1f77b4" for e in abs_err]
    ax_bar.bar(x, abs_err, color=bar_colors, width=0.6, zorder=3)
    ax_bar.axhline(mae, color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(lap_labels, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("Absolute error (Wh)")
    ax_bar.set_title("Absolute Error per Lap")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_bar.legend(handles=[
        plt.Line2D([0], [0], color="#ff7f0e", linestyle="--", linewidth=1.5,
                   label=f"MAE ({mae:.2f} Wh)"),
        Patch(facecolor="#d62728", label="Above MAE"),
        Patch(facecolor="#1f77b4", label="At or below MAE"),
    ], fontsize=8)

    fig.tight_layout()
    out_path = Path(results_dir) / "evaluation_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(model, feat_names, results_dir="results"):
    """Save a ranked horizontal bar chart of Random Forest feature importances.

    This replaces the learning curve from the neural network version.
    Random Forests compute importance scores automatically during training
    (based on how much each feature reduces impurity across all trees).
    Higher = the model relied on that feature more heavily.

    Features are coloured by sensor group so related stats cluster visually.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)  # ascending so most important is at top

    # Assign a colour to each base feature column so all four stats of a
    # given sensor share a colour (makes the chart easier to scan).
    from .model import FEATURE_COLUMNS, STAT_FUNCTIONS
    n_stats = len(STAT_FUNCTIONS)
    palette = plt.cm.tab10.colors
    colors = []
    for i, name in enumerate(feat_names):
        col_idx = next(
            j for j, col in enumerate(FEATURE_COLUMNS) if name.startswith(col)
        )
        colors.append(palette[col_idx % len(palette)])
    colors = [colors[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_names) * 0.35)))
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_names[i] for i in indices], fontsize=8)
    ax.set_xlabel("Feature importance (mean impurity decrease)")
    ax.set_title("Random Forest — Feature Importance", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Legend: one entry per sensor column
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=palette[j % len(palette)], label=col)
        for j, col in enumerate(FEATURE_COLUMNS)
    ]
    ax.legend(handles=legend_handles, fontsize=8, title="Sensor", loc="lower right")

    fig.tight_layout()
    out_path = Path(results_dir) / "feature_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained lap energy model.")
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(data_dir=args.data_dir, results_dir=args.results_dir)