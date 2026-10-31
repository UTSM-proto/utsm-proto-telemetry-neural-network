# UTSM Telemetry Model Experiments

## Retrain the Random Forest on Laval data

Start with [the Colab notebook](notebooks/retrain_laval.ipynb). Upload that notebook to Colab, select a CPU runtime, run the cells and upload `Laval All Telem.zip` when prompted.

The workflow validates embedded GPS/wheel telemetry, records calibration assumptions, tunes forests using complete development sessions and compares them with analytical baselines on held-out sessions. It exports models, preprocessing, metrics, plots and a downloadable result archive. See [the full setup and interpretation guide](docs/LAVAL_COLAB.md).

`requirements-colab.txt` is the pinned environment for this new workflow. The older `requirements.txt` and `random_forest_tree/` files describe the historical lap-summary experiment.

**Accuracy is measured, not guaranteed:** the notebook reports when the forest fails to beat the analytical baseline. Constant recorded voltage and unverified current calibration keep default runs exploratory. No raw Laval archive or newly trained model is committed here.

## Historical lap-energy experiments

The original project predicts total lap energy from `lap{N}_distgrid.csv` outputs of the telemetry processing pipeline, using per-position driving information. Two model directions were explored:

- Random Forest (`random_forest_tree/`, historical saved lap-summary artifact).
- ANN/MLP (older development branches).

The new Laval workflow predicts interval power and integrates it to observed driving-session energy. Its versioned feature schema is intentionally separate from the historical saved forest; it does not claim those artifacts are interchangeable.
