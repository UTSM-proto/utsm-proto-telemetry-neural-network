# Retrain on Laval in Colab

Upload `notebooks/retrain_laval.ipynb` to [Google Colab](https://colab.research.google.com/), select a **CPU runtime**, and run the cells in order (or Run all). When prompted, upload **Laval All Telem.zip** from your computer. The notebook embeds a checksum-verified copy of the training helper, frozen analytical reference and dependency list; it does not download source code. The notebook itself is the only file you need to upload.

## What you get
- Archive/file hashes, per-file quality/exclusion counts and a canonical interval dataset.
- A frozen chronological split: September 19 development; September 20 test.
- Session-grouped tuning of four bounded Random Forest candidates, with fold-fitted imputation and equal session-duration weights.
- Comparison with a matched-data analytical regression refit **and the frozen Front Campus analytical model**.
- Development-only fitted models, feature schema, held-out predictions/metrics, chart, HTML report, environment record and a downloadable result ZIP.
- A model-reload prediction example. Optionally copy results to your Google Drive.

The source reference comes from telemetry commit `00a27b6d25cd143354b17170f522a6a44aaa7b01`, file `data/tracks/autodrome-chaudiere/autodrome-chaudiere-model.json`. It was fitted to 15 Front Campus laps. This workflow trains the new models **on the Laval archive**; it does not silently pool the old Front Campus training data into Laval's development split.

## Calibration settings: read before changing
Defaults run an **exploratory experiment**: current magnitude, scale 1, zero 0, `calibration_confirmed=False`. This reproduces neither a verified instrument calibration nor a claim that negative readings are regenerative energy. No old 0.7949 gain is automatically transferred to Laval.

For measured calibration, set:
- `current_mode="negative"` when negative raw current means positive consumption, or `"positive"` for the opposite polarity; `"magnitude"` is an explicitly approximate historical convention.
- `current_zero_mA`: raw sensor zero before scaling.
- `current_scale`: positive calibration gain.
- Per-file overrides in `calibration_overrides` if hardware/zero/gain changed.
- `calibration_confirmed=True` only after validating these with the sensor/bench records.

Every inspected Laval file records 24.289 V. Confirm whether it is a fixed firmware value. Until then, the energy label means **integrated recorded V x calibrated I**, under that constant-voltage assumption. The notebook always reports constant voltage even if current calibration is confirmed.

## What the estimator learns
It predicts interval-average electrical power, then integrates that power on the **same valid moving intervals** for each model. It is a new versioned Random Forest, **not a continuation of results/best_model.joblib**: that older artifact expects 24 lap-summary features and lacks its corresponding checked-in inputs.

Core inputs: causal smoothed speed, speed squared, positive/negative acceleration and speed-acceleration interactions. Enhanced candidates add valid temperature/missingness, short speed history and sensor-source flag. No measured current, target power or target energy is an input. Forests have 200 trees; candidates vary feature set/leaf regularization. These deliberately small experiments run on CPU.

The analytical refit uses the core terms with standardized ridge regression. It is a matched-data control, not an assertion that the original simulator has been identically refitted. The frozen reference retains the original electrical/coast rule, using neutral track position, zero grade and no extra motor-physics floor, on these observed intervals.

## Data handling and scope
- Read ZIP members in memory, never extract or change your source archive.
- Skip unreadable/non-telemetry CSVs and exact/parsed duplicate exports, and report exclusions.
- Split history at reboots, duplicate timestamps, gaps >2 s and invalid speed/electrical readings.
- Prefer valid wheel speed; otherwise valid GPS speed with coordinate/HDOP checks. Reject large sensor disagreement.
- Trailing-three speed median/history uses only past/current readings. Invalid temperature remains missing until a training-fold-fitted imputer handles it.
- Exclude speeds below 2 km/h or above 60 km/h, implausible acceleration, out-of-range current/voltage, and insufficient sessions.
- Integrate current/voltage with trapezoidal intervals, with no interpolation across gaps.
- Sensor flags and HDOP are only checks, not proof GPS is fresh. Inspect `data_quality.csv` and compare wheel/GPS traces when validating sensors.

The archive has no explicit throttle channel and the supplied GPX is untimed geometry. This workflow **does not invent complete laps or claim energy savings from an optimized strategy**. Metrics cover accepted moving intervals within sessions; short time bins are additional diagnostics, not laps. Idle energy and invalid/missing intervals are excluded consistently from both actual and predicted energy. A valid per-lap benchmark needs separately verified lap boundaries.

## Prevent misleading improvement claims
Model selection sees only development sessions, using GroupKFold by whole CSV session. Rows/reboot blocks from one session never cross a fold. September 20 errors are computed after model selection, not used to select hyperparameters. Changing the notebook after seeing these errors makes that date a development set for subsequent claims.

The two-day archive has already been inspected during an audit. These are **retrospective held-out session results**, not a prospective new-day validation. Acquire fresh-day data before making deployment claims.

`random_forest_development.joblib` is fitted only on development sessions. A separately named all-data refit is off by default and requires confirmed calibration plus a forest score better than **every** analytical baseline. Even then, `random_forest_all_data_UNTESTED.joblib` needs a new future test. A worse forest is still exported for inspection but never silently promoted.

## Files in the result ZIP
| File | Meaning |
|---|---|
| `summary.json`, `report.html`, `comparison.png` | Status, model ranking, assumptions and results |
| `data_quality.csv`, `source_manifest.json` | Exclusions, coverage and source/code hashes |
| `split_manifest.json`, `development_cv.csv` | Frozen session groups and tuning scores |
| `canonical_intervals.csv.gz` | Accepted data with exact source row identifiers |
| `held_out_sessions.csv`, `held_out_windows.csv` | Session/bin energy error and coverage |
| `held_out_predictions.csv.gz` | Auditable per-interval actual/predicted values |
| `random_forest_development.joblib` | New forest + fitted imputer + ordered features + metadata |
| `analytical_development.joblib` | Matched analytical model |
| `front_campus_reference.json` | Frozen analytical reference coefficients |
| `feature_schema.json`, `config.json`, `environment.txt`, `laval_training.py` | Reproduction/inference contract |

Local inference (use a model artifact you generated, with its recorded environment):
```python
import joblib
from laval_training import predict_power
model = joblib.load("results/random_forest_development.joblib")
watts = predict_power(model, canonical_features[model["features"]])
```
A single raw sensor row cannot supply causal history: use the same preprocessing and reset boundaries. The notebook includes a round-trip example without using label columns.

## Local execution
```bash
python -m pip install -r requirements-colab.txt
python -m unittest discover -s tests -p 'test_laval*.py' -v
python laval_training.py "/path/to/Laval All Telem.zip" --config configs/laval.example.json --reference configs/front_campus_reference.json --output outputs/laval-run-01
```
Use a new output directory for each run. Existing results are never overwritten. Colab dependencies are pinned; exact actual runtime versions are recorded. Local validation was also performed with Python 3.9/scikit-learn 1.3.2, and is separate from execution on Google's hosted service.

## Next improvements
1. Resolve current polarity/gain and the constant voltage before interpreting accuracy as metered energy accuracy.
2. Repair missing GitHub archive content: the local 20,436-row session must not be replaced by the two-byte GitHub file.
3. Add verified lap boundaries, hardware/battery metadata, and surveyed grade if available.
4. Collect measured throttle/brake commands before using a learned model for strategy counterfactuals.
5. Add Front Campus through an explicit independently calibrated adapter, rather than mixing incompatible schemas.
6. Collect an untouched future day; compare real laps at matched time/driver/vehicle/battery conditions before claiming savings.
