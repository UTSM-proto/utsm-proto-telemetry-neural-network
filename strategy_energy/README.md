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
| Capstone-calibrated physics and neural correction | 8.534 Wh |
| Matched analytical electrical refit | 9.314 Wh |
| 100k hybrid, epoch 3, zero physics weight | 9.324 Wh |
| Original PR hybrid with physics loss | 9.379 Wh |
| 100k hybrid, epoch 7, correction head | 9.790 Wh |
| 100k hybrid, epoch 7, partial unfreezing and run-energy loss | 10.481 Wh |
| Analytical propulsion term alone | 20.907 Wh |

The Wh benchmark uses the same three held-out runs, one per track, and the same accepted moving intervals. It measures replay. The partial-unfreezing experiment selected the last two linear layers and a run-energy loss weight of 1.0 in grouped validation (5.287 Wh). It did not improve held-out error, so the telemetry-only MLP is the default.

The matched analytical row refits the telemetry repo's [weighted electrical model](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/utsm_telemetry/simulation.py#L174) on the same development intervals, with neutral grade and track position. Its native [Front Campus pipeline](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/docs/FRONT_CAMPUS_PHYSICS_MODEL.md) scored **16.84%** mean absolute run-energy error in a separate three-run leave-one-out test with 50 m aggregation. The [Indianapolis same-track analytical fit](https://github.com/UTSM-proto/utsm-proto-telemetry/blob/3aaad5ac28f9c5780414e178102dc703b1679d72/data/tracks/indianapolis/indianapolis-strategy-report.txt) reported **14.8% underprediction** on the Indy morning run. These percentage benchmarks use different evaluation setups.

## Capstone-informed physical calibration

The calibrated model jointly fits bounded physical parameters and a regularized correction head
over the frozen 100k-pretrained neural features. Training uses interval power, integrated run
energy, simulator consistency, and parameter-prior penalties. Zero simulator-consistency weight
competes with positive weights. The signed acceleration force is included before clipping
tractive power for consumption-only targets.

The [capstone report](https://drive.google.com/file/d/1qvCYFOS8ZSFoYRYC22LwWdowOZKjA7A4/view)
estimates rolling resistance at 0.003 (printed p. 49) and gives an optimized-body calculation
with Cd = 0.24 and frontal area = 0.293 m² (Appendix C, printed p. 60).
Its title page is dated March 2017 despite the Drive filename "2019 Capstone.pdf".
These are historical engineering estimates. The total mass and electric efficiency priors below
are assumptions carried from the simulator. The report's monocoque masses do not establish
the current total car-plus-driver mass.

| Parameter | Initial value | Permitted range | Selected effective estimate |
| --- | ---: | ---: | ---: |
| Total car plus driver mass | 100 kg | 60–140 kg | 60 kg |
| Rolling-resistance coefficient | 0.003 | 0.001–0.010 | 0.003081 |
| CdA | 0.07032 m² | 0.040–0.130 m² | 0.06870 m² |
| Electric drivetrain efficiency | 0.82 | 0.50–0.95 | 0.95 |

Mass and efficiency reached bounds. These fitted values describe this regularized model's
effective parameters and require independent measurements before being used as actual car specs.
The field dataset has no consistently aligned grade or curvature, so fitting uses zero grade
and omits cornering drag. Optional grade_pct in prediction affects the analytical force term.

The sweep tested 49 configurations in four whole-run development folds, then increased the
optimization budget from 80 to 400 function evaluations for the best three configurations.
The selected configuration used run-energy weight 0.2, simulator-consistency weight 0.1 and
correction regularization 0.1. Its grouped development error was 5.347 Wh. Longer optimization
returned the same score. The final development fit converged after 11 evaluations.

| Same three-run replay benchmark | Mean absolute run-energy error |
| --- | ---: |
| Telemetry-only MLP | **8.210 Wh** |
| Calibrated physics plus neural correction | 8.534 Wh |
| Previous epoch-7 hybrid correction | 9.790 Wh |
| Previous partial unfreezing | 10.481 Wh |
| Calibrated signed-force physics without correction | 12.025 Wh |
| Signed-force physics with nominal capstone priors | 12.587 Wh |
| Signed-force physics with old simulator parameters | 17.048 Wh |

The selected calibrated model scored 9.422 Wh on Front Campus, 1.996 Wh on Indianapolis and
14.185 Wh on Laval. The baseline MLP remains the default. Configuration selection uses only
development folds. The three held-out runs have been repeatedly inspected across experiments,
so this is a retrospective comparison. A new driving day is needed for a prospective result.
Labels retain recorded voltage and current magnitude without additional sensor calibration.

Predict using the checked-in calibrated artifact from this directory.

```sh
python predict.py example.csv --model calibrated --output predicted.csv
```

Reproduce calibration using the pinned telemetry checkout described above.

```sh
python /path/to/strategy_energy/calibrate_physics.py --root /path/to/telemetry --output /path/to/calibration-run
python -m unittest test_calibrate_physics.py
```

Candidate results are cached in the output directory. The dataset hashes and pretrained
checkpoint hash are checked before reuse. Changing the objective requires a new output directory
or a new objective_version in the source. The output contains calibrated.joblib and report.json.
To score a newly trained artifact with predict.py, place it at checkpoints/calibrated-physics.joblib.
The full benchmark and optimization diagnostics are in benchmarks/capstone-calibration.json.
