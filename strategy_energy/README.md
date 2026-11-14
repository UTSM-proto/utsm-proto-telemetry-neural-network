# Strategy energy models

Use Python 3.12 in a virtual environment. From this folder, install the pinned dependencies with `python -m pip install -r requirements.txt`. For PyTorch you can use a CPU wheel. No GPU is needed for prediction.

```sh
python predict.py example.csv --model mlp
python predict.py example.csv --model finetuned
```

Provide a CSV with speed_kph, accel_m_s2 and dt_s. Each row represents a time interval. Prediction prints total Wh and can export interval watts with --output. The features describe a known or simulated speed trace. These checkpoints have been evaluated on telemetry replay.

## Reproduce training

Clone the [telemetry repository](https://github.com/UTSM-proto/utsm-proto-telemetry) and check out commit `9376800569082d09c5b7ccfdbd34759ef4314d4e`. This pins the source telemetry, physics and split code without duplicating raw runs. From the root of that checkout, use absolute paths to the files in this folder.

```sh
python /path/to/strategy_energy/finetune_field.py --root . --pretrained /path/to/strategy_energy/checkpoints/pretrained.joblib --output outputs/field-finetune
python /path/to/strategy_energy/train_mlp.py --root . --output outputs/telemetry-only-mlp.joblib
```

The pretrained bundle contains the scaler, weights and optimizer state from the best 100k pilot epoch. To generate your own simulated corpus and resume synthetic training, use the [training runbook](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/9376800569082d09c5b7ccfdbd34759ef4314d4e/docs/HYBRID_ENERGY_SCALE.md).

## Benchmark results

| Model | Mean absolute held-out run energy error |
| --- | ---: |
| Telemetry-only MLP | **8.210 Wh** |
| Random Forest | 8.259 Wh |
| Matched analytical electrical refit | 9.314 Wh |
| 100k hybrid, epoch 3, zero physics weight | 9.324 Wh |
| Original PR hybrid with physics loss | 9.379 Wh |
| 100k hybrid, epoch 7, correction head | 9.790 Wh |
| 100k hybrid, epoch 7, partial unfreezing and run-energy loss | 10.481 Wh |
| Analytical propulsion term alone | 20.907 Wh |

The Wh benchmark uses the same three held-out runs, one per track, and the same accepted moving intervals. It measures replay. The partial-unfreezing experiment selected the last two linear layers and a run-energy loss weight of 1.0 in grouped validation (5.287 Wh). It did not improve held-out error, so the telemetry-only MLP is the default.

The matched analytical row refits the telemetry repo's [weighted electrical model](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/utsm_telemetry/simulation.py#L174) on the same development intervals, with neutral grade and track position. Its native [Front Campus pipeline](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/docs/FRONT_CAMPUS_PHYSICS_MODEL.md) scored **16.84%** mean absolute run-energy error in a separate three-run leave-one-out test with 50 m aggregation. The [Indianapolis same-track analytical fit](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/data/tracks/indianapolis/indianapolis-strategy-report.txt) reported **14.8% underprediction** on the Indy morning run. These percentage benchmarks use different evaluation setups.
