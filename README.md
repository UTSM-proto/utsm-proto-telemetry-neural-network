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

Two main models:
- Random Forest Tree
- ANN/MLP
