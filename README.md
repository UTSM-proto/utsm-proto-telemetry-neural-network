# UTSM Telemetry Neural Network
**Predicts total lap energy consumption from race telemetry data.**

---
**Obligatory Disclaimer:** This is written in a manner that assumes you know what neural networks are at a beginner level.

## What it does

After each test run, the utsm-proto-telemetry repo data processing pipeline produces a file called `lap{N}_distgrid.csv`. This is a table where every row represents a position along the track, with columns for speed, acceleration, slope, motor temperature, drag due to wind, and so on.

This project trains a small neural network to look at all of that per-position data for a lap and answer one question:

> *"Given these conditions, how much total energy did this lap consume?"*

Once trained on enough historical laps, the network can predict energy for a new lap without needing to wait for the post-run data reduction.

---

## How it learns

The process mirrors the way a student learns from a textbook with an answer key:

1. **Show it a lap's telemetry** (the "question")
2. **Tell it the actual measured energy for that lap** (the "answer")
3. **Compute the error** — how far off was the prediction?
4. **Nudge the network's internal weights** to do slightly better next time
5. **Repeat** for every lap in your dataset, many times over

This is called supervised regression training. The metric being minimised is **Mean Squared Error (MSE)** — essentially the average of (predicted energy − actual energy)² across all laps.

---

## Project structure

```
utsm-proto-telemetry-neural-network/
│
├── data/                        ← put the lap{N}_distgrid.csv files here
│   └── lap_labels.csv           ← actual total energy per lap (see below)
│
├── results/                     ← created automatically during training
│   ├── best_model.pt            ← saved checkpoint of the best weights found
│   ├── train_history.json       ← loss per epoch (useful for plotting)
│   └── evaluation.json          ← per-lap predicted vs. actual results
│
├── src/
│   ├── __init__.py              ← exposes the package API
│   ├── model.py                 ← network architecture + data loading
│   ├── train.py                 ← training loop
│   └── evaluate.py              ← load checkpoint and report accuracy
│
├── requirements.txt
└── README.md
```

---

## The network architecture

The model is a **feed-forward neural network**. There are no recurrent loops or attention heads; data flows in one direction only.

```
Input layer
(speed, accel, slope, motor_temp, force, wind) × 200 track positions
= 1 200 numbers
        │
        ▼
  Linear(1200 → 128) + ReLU     ← "hidden layer 1"
        │
        ▼
  Linear(128 → 64) + ReLU       ← "hidden layer 2"
        │
        ▼
  Linear(64 → 1)                ← single output: predicted energy (Wh)
```

**ReLU** (Rectified Linear Unit) is just `max(0, x)`. It is a simple gate that lets the network learn non-linear relationships between your inputs and the output, rather than being limited to a straight-line fit.

---

## Files in `src/` explained

### `model.py` — data + architecture

Two responsibilities:

- **Loading data:** Reads each `lap{N}_distgrid.csv`, picks the configured feature columns, resamples every lap onto the same number of rows (so all inputs are the same length), and flattens the result into a single vector.
- **The network:** Defines `LapEnergyNet`, the class that holds the layers described above.

### `train.py` — the learning loop

1. Splits laps into a **training set** (~80%) and a **validation set** (~20%).
2. For each training epoch, feeds every training lap through the network, measures the MSE error, and updates the weights via **Adam** (a gradient descent optimiser).
3. After each epoch, checks performance on the validation set, laps the network has *never seen*, to detect overfitting.
4. Saves a checkpoint (`best_model.pt`) whenever validation performance improves.

### `evaluate.py` — checking the result

Loads `best_model.pt` and runs every lap in teg data directory through the frozen network. Prints a table of predicted vs. actual energy per lap, then reports three summary numbers:

| Metric | What it means |
|--------|-------------------|
| **MAE** | Average absolute error in Wh — easiest to interpret |
| **RMSE** | Like MAE but punishes large individual mistakes more heavily |
| **MAPE** | Error as a percentage of actual energy — good for comparing across laps of different lengths |

---

## Setting up your data

### Feature columns (`lap{N}_distgrid.csv`)

Each lap CSV should have at minimum these columns (rename them in `model.py` if yours differ):

| Column | Description |
|--------|-------------|
| `speed` | Speed at this track position |
| `accel` | Acceleration at this track position |
| `slope` | Track gradient at this position |
| `motor_temp` | Motor temperature at this position |
| `force_total` | Computed net force at this position |
| `wind_speed` | Wind speed (relative or absolute) at this position |

Every other column in the CSV is ignored by default.

### Actual energy labels

The network needs a ground-truth answer for each lap to learn from. Set `TARGET_MODE` in `model.py` to one of:

| Mode | When to use it |
|------|---------------|
| `"cumulative_column"` | CSV already has a running total energy column, the last row's value is the lap total |
| `"sum_column"` | CSV has a per-row instantaneous power/energy; sum all rows to get the lap total |
| `"labels_file"` | Total energy comes from a separate source (e.g. , a coulomb counter). Create `data/lap_labels.csv` with columns `lap_id,total_energy_wh` |

---

## Usage

Install dependencies:
```bash
pip install torch pandas numpy
```

Train the model:
```bash
python -m src.train --data-dir data --results-dir results --epochs 200
```

Evaluate after training:
```bash
python -m src.evaluate --data-dir data --results-dir results
```

Both commands accept `--help` for a full list of options.

---

## Tuning tips

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Training loss drops but validation loss stays high | **Overfitting** — the network has memorised the training laps | Reduce `hidden_dims`, collect more laps, or add `--val-frac 0.3` |
| Both losses stay high | **Underfitting** — the network is too small or training too briefly | Increase `hidden_dims` or `--epochs` |
| Validation loss is noisy / unstable | Validation set is too small | Collect more laps, or reduce `--val-frac` |
| Predictions are systematically off by a scale factor | Column units mismatch (e.g. m/s vs km/h) | Normalise units in your pre-processing script before training |