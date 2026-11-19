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

## Strategy energy checkpoints

See [setup, prediction, and training](strategy_energy/README.md).

## Benchmark results

| Model | Mean absolute held-out run energy error |
| --- | ---: |
| Telemetry-only MLP | **8.210 Wh** |
| Random Forest | 8.259 Wh |
| Capstone-calibrated physics and neural correction | 8.534 Wh |
| Matched analytical electrical refit | 9.314 Wh |
| 100k hybrid, epoch 3, zero physics weight | 9.324 Wh |
| Original PR hybrid with physics loss | 9.379 Wh |
| 100k hybrid, epoch 7, correction head | 9.790 Wh |
| 100k hybrid, epoch 7, partial unfreezing and run-energy loss | 10.481 Wh |
| Analytical propulsion term alone | 20.907 Wh |

The Wh benchmark uses the same three held-out runs, one per track, and the same accepted moving intervals. It measures replay. The partial-unfreezing experiment selected the last two linear layers and a run-energy loss weight of 1.0 in grouped validation (5.287 Wh). It did not improve held-out error, so the telemetry-only MLP is the default.

The matched analytical row refits the telemetry repo's [weighted electrical model](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/utsm_telemetry/simulation.py#L174) on the same development intervals, with neutral grade and track position. Its native [Front Campus pipeline](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/docs/FRONT_CAMPUS_PHYSICS_MODEL.md) scored **16.84%** mean absolute run-energy error in a separate three-run leave-one-out test with 50 m aggregation. The [Indianapolis same-track analytical fit](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/data/tracks/indianapolis/indianapolis-strategy-report.txt) reported **14.8% underprediction** on the Indy morning run. These percentage benchmarks use different evaluation setups.

The capstone-calibrated model jointly fits bounded road-load parameters and a neural correction. It scored 8.534 Wh in the same retrospective three-run benchmark. Mass and efficiency reached their bounds, so the fitted parameters require independent measurements. See [calibration setup and results](strategy_energy/README.md#capstone-informed-physical-calibration).
