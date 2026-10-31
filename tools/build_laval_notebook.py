"""Generate the upload-only Colab notebook from the reviewed training module."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT / "laval_training.py").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    reference = (ROOT / "configs/front_campus_reference.json").read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    environment_source = (ROOT / "colab_environment.py").read_text(encoding="utf-8")
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text, hidden=False):
        metadata = {"jupyter": {"source_hidden": True}, "cellView": "form"} if hidden else {}
        cells.append({"cell_type": "code", "execution_count": None, "metadata": metadata,
                      "outputs": [], "source": text.splitlines(True)})

    md("""# UTSM: retrain the Random Forest with Laval data
**Upload this notebook to Colab, choose a CPU runtime, then Run all.**
When prompted, upload **Laval All Telem.zip**. No separate Python files or GitHub login are needed.

This notebook contains the reviewed training code, analytical reference and dependency list.
It prepares telemetry, tunes forests using September 19 sessions, tests on September 20 sessions,
compares against analytical models, and downloads the complete result bundle.

**Default results are exploratory:** current calibration is unverified and the archive records constant voltage.
The forest is not assumed to improve on the analytical model. Read the report before choosing a model.
""")
    md("""## 1. Install the pinned CPU environment
Run this first. The pinned NumPy 2 / pandas stack supports Python 3.10-3.13.
The cell checks scientific imports in a separate process and then the active kernel.
**If it asks for a restart, choose Runtime > Restart session, then Run all again.**
Rerunning only the model-loading cell cannot replace old NumPy binaries in memory.
No GPU is required for the forest or analytical model.
""")
    code("DEPENDENCIES_READY = False\n"
         + "import os, sys, subprocess, importlib.util\n"
         + "from pathlib import Path\n"
         + "REQUIREMENTS = " + repr(requirements) + "\n"
         + "BUNDLED_ENVIRONMENT_SOURCE = " + repr(environment_source) + "\n"
         + "IN_COLAB = 'google.colab' in sys.modules or Path('/content').is_dir()\n"
         + "WORK_DIR = Path(os.environ.get('UTSM_WORK_DIR', '/content/utsm-laval' if IN_COLAB else 'laval-runtime')).resolve()\n"
         + "WORK_DIR.mkdir(parents=True, exist_ok=True)\n"
         + "(WORK_DIR / 'requirements-colab.txt').write_text(REQUIREMENTS, encoding='utf-8')\n"
         + "environment_path = WORK_DIR / 'colab_environment.py'\n"
         + "environment_path.write_text(BUNDLED_ENVIRONMENT_SOURCE, encoding='utf-8')\n"
         + "env_spec = importlib.util.spec_from_file_location('utsm_colab_environment', environment_path)\n"
         + "environment = importlib.util.module_from_spec(env_spec)\n"
         + "sys.modules[env_spec.name] = environment\n"
         + "env_spec.loader.exec_module(environment)\n"
         + "environment.ensure_environment(WORK_DIR / 'requirements-colab.txt')\n"
         + "DEPENDENCIES_READY = True\n"
         + "print('Workspace:', WORK_DIR)\n")
    md("""## 2. Load the bundled, versioned implementation
This cell writes only the helper and frozen analytical reference into the runtime workspace.
It does not download source code. The SHA-256 check catches an accidental mismatch in the embedded helper.
""")
    code("if not globals().get('DEPENDENCIES_READY', False):\n"
         + "    raise RuntimeError('Run the dependency cell successfully first. If prompted, restart the session and Run all.')\n"
         + "import hashlib, importlib.util, json\n"
         + "BUNDLED_TRAINING_SOURCE = " + repr(source) + "\n"
         + "BUNDLED_SOURCE_SHA256 = " + repr(source_hash) + "\n"
         + "FRONT_CAMPUS_REFERENCE_JSON = " + repr(reference) + "\n"
         + "assert hashlib.sha256(BUNDLED_TRAINING_SOURCE.encode()).hexdigest() == BUNDLED_SOURCE_SHA256\n"
         + "MODULE_PATH = WORK_DIR / 'laval_training.py'\n"
         + "MODULE_PATH.write_text(BUNDLED_TRAINING_SOURCE, encoding='utf-8')\n"
         + "REFERENCE_PATH = WORK_DIR / 'front_campus_reference.json'\n"
         + "REFERENCE_PATH.write_text(FRONT_CAMPUS_REFERENCE_JSON, encoding='utf-8')\n"
         + "spec = importlib.util.spec_from_file_location('utsm_laval_runtime', MODULE_PATH)\n"
         + "training = importlib.util.module_from_spec(spec)\n"
         + "sys.modules[spec.name] = training\n"
         + "spec.loader.exec_module(training)\n"
         + "print('Training source SHA-256:', BUNDLED_SOURCE_SHA256)\n"
         + "print('Frozen Front Campus source: telemetry commit 00a27b6d25cd143354b17170f522a6a44aaa7b01')\n",
         hidden=True)
    md("""## 3. Configure calibration and data
Leave ARCHIVE_PATH empty to upload the ZIP. Alternatively, enable Drive and enter its mounted file path.
Settings below are examples, not validated sensor measurements.

- **magnitude:** absolute raw current after zero/gain correction (exploratory historical convention).
- **negative:** negative raw current represents consumption. **positive:** positive represents consumption.
- Set current zero/gain using bench records, optionally by filename in CALIBRATION_OVERRIDES.
- CALIBRATION_CONFIRMED only describes your current calibration; constant voltage still limits energy labels.
- Keep September 20 held out. Never tune settings based on its prediction errors.
""")
    code("""ARCHIVE_PATH = os.environ.get("UTSM_ARCHIVE_PATH", "")  # leave blank to upload
USE_GOOGLE_DRIVE = False
SAVE_RESULTS_TO_DRIVE = False
DRIVE_RESULTS_FOLDER = "/content/drive/MyDrive/UTSM-model-results"

CURRENT_MODE = "magnitude"
CURRENT_SCALE = 1.0
CURRENT_ZERO_MA = 0.0
CALIBRATION_CONFIRMED = False
CALIBRATION_OVERRIDES = {}  # {"2026-09-19-19-26.csv": {"current_mode": "negative", "current_scale": 1.0}}
TEST_DATES = ["2026-09-20"]
REFIT_ALL_FOR_RELEASE = False  # requires calibration + improvement; still needs a future test

cfg = training.Config(
    current_mode=CURRENT_MODE, current_scale=CURRENT_SCALE,
    current_zero_mA=CURRENT_ZERO_MA, calibration_confirmed=CALIBRATION_CONFIRMED,
    calibration_overrides=CALIBRATION_OVERRIDES, test_dates=TEST_DATES,
    refit_all_for_release=REFIT_ALL_FOR_RELEASE,
)
cfg.validate()
if not CALIBRATION_CONFIRMED:
    print("EXPLORATORY RUN: current calibration is not confirmed.")
""")
    md("## 4. Provide your ZIP (source files remain unchanged)\n")
    code("""if USE_GOOGLE_DRIVE or SAVE_RESULTS_TO_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Drive mounting is a Colab option; use a local path outside Colab.")
    from google.colab import drive
    drive.mount("/content/drive")

if not ARCHIVE_PATH:
    if not IN_COLAB:
        raise ValueError("Set ARCHIVE_PATH to your ZIP outside Colab.")
    from google.colab import files
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
    if len(zip_names) != 1:
        raise ValueError("Upload exactly one telemetry ZIP.")
    archive_path = WORK_DIR / Path(zip_names[0]).name
    archive_path.write_bytes(uploaded[zip_names[0]])
else:
    archive_path = Path(ARCHIVE_PATH).expanduser().resolve()
if not archive_path.is_file():
    raise FileNotFoundError(archive_path)
print("Archive:", archive_path.name)
""")
    md("""## 5. Inspect retained data and the session split
This uses embedded CSV GPS, not the untimed track GPX. History resets at gaps/reboots;
invalid sensor intervals are excluded. Quality rules are fixed before model tuning.

These outputs are **accepted moving intervals**, not complete-lap labels or total session energy.
Review excluded files, accepted duration, current signs, voltage and sensor disagreement.
Do not loosen rules just because one model's score improves.
""")
    code("""import pandas as pd
try:
    from IPython.display import display, Image
except ImportError:
    display = print
canonical, quality, provenance = training.load_archive(archive_path, cfg)
dev, held_out, split = training.split_sessions(canonical, cfg)
display(quality)
display(pd.DataFrame([
    {"split": "development", "sessions": dev.session_id.nunique(), "rows": len(dev), "valid_minutes": dev.dt_s.sum()/60},
    {"split": "held_out", "sessions": held_out.session_id.nunique(), "rows": len(held_out), "valid_minutes": held_out.dt_s.sum()/60},
]))
print("Archive SHA-256:", provenance["archive_sha256"])
print(json.dumps(split, indent=2))
if quality.get("constant_voltage", pd.Series(dtype=bool)).fillna(False).any():
    print("CONSTANT VOLTAGE: energy labels assume the recorded voltage is correct; voltage sag is unobserved.")
""")
    md("""## 6. Train, tune and compare
Four bounded forest candidates use session-grouped cross-validation on development data only.
Imputation and analytical scaling are fit inside each training fold.
After selecting the forest, evaluate the held-out sessions once against:
1. the selected Random Forest,
2. an analytical regression refit on the same development data,
3. the frozen Front Campus analytical model.

A new output directory is used; previous results are never overwritten. Reusing this date's
scores to choose further changes invalidates it as a final test.
""")
    code("""from datetime import datetime, timezone
run_name = "laval-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
output_dir = Path(os.environ.get("UTSM_OUTPUT_DIR", str(WORK_DIR / "outputs" / run_name))).resolve()
summary = training.run_workflow(archive_path, output_dir, cfg, REFERENCE_PATH)
print("Decision:", summary["recommendation"])
print("Results:", output_dir)
""")
    md("""## 7. Read the results
Lower error is better. Session errors cover only accepted driving intervals.
Time-bin MAE is diagnostic; it is not lap-energy accuracy.

The archive has already been explored in an audit. A real fresh-day test and verified calibration
are still required before claiming race performance or deployment readiness.
""")
    code("""display(pd.DataFrame(summary["model_ranking"]))
display(pd.read_csv(output_dir / "held_out_sessions.csv"))
if "Image" in globals():
    display(Image(filename=str(output_dir / "comparison.png")))
for warning in summary["warnings"]:
    print("-", warning)
""")
    md("""## 8. Reload the forest and test inference
This proves the saved artifact includes the fitted imputer and ordered schema.
The prediction call below uses only input features, not current/power/energy labels.
New raw data must use the same causal preprocessing and calibration.
""")
    code("""import joblib
import numpy as np
model = joblib.load(output_dir / "random_forest_development.joblib")
examples = held_out[model["features"]].head(5)
predicted_watts = training.predict_power(model, examples)
display(examples.assign(predicted_power_w=predicted_watts))
saved_predictions = pd.read_csv(output_dir / "held_out_predictions.csv.gz")
recorded = saved_predictions[saved_predictions.model == "random_forest"].predicted_power_w.head(5).to_numpy()
np.testing.assert_allclose(predicted_watts, recorded)
print("Saved-model prediction round trip passed.")
""")
    md("""## 9. Save/download everything
The ZIP includes models, canonical data, predictions, settings, hashes, source code, reports and the
actual environment. Save it before the runtime ends. No raw telemetry is sent to GitHub.

The notebook does not replace an existing deployed model. An all-data refit is separately named,
off by default, and skipped unless calibration is confirmed and the forest beats all analytical baselines.
""")
    code("""import shutil
result_zip = Path(str(output_dir) + ".zip")
if SAVE_RESULTS_TO_DRIVE:
    destination = Path(DRIVE_RESULTS_FOLDER) / output_dir.name
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(result_zip, destination / result_zip.name)
    print("Saved to Drive:", destination)
if IN_COLAB:
    from google.colab import files
    files.download(str(result_zip))
else:
    print("Download/share this result bundle:", result_zip)
""")
    md("""## Next experiments
First verify sensor gain/polarity and fixed voltage. Then collect lap boundaries and throttle commands.
Keep future-day data independent. Only after a clean CPU baseline should GPU/sequence-network
experiments consume Colab credits.

The notebook source is maintained by tools/build_laval_notebook.py; tests check that its embedded
helper, analytical reference and requirements match the reviewed source files.
""")
    for i, cell in enumerate(cells):
        cell["id"] = f"laval-{i:02d}"
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.13"},
                     "colab": {"name": "retrain_laval.ipynb", "provenance": []},
                     "utsm_training_source_sha256": source_hash},
        "cells": cells,
    }
    path = ROOT / "notebooks/retrain_laval.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
