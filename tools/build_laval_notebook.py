"""Build an upload-only notebook whose scientific work runs in fresh Python processes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT/"laval_training.py").read_text(encoding="utf-8")
    environment_source = (ROOT/"colab_environment.py").read_text(encoding="utf-8")
    worker_source = (ROOT/"laval_notebook_worker.py").read_text(encoding="utf-8")
    requirements = (ROOT/"requirements-colab.txt").read_text(encoding="utf-8")
    reference = (ROOT/"configs/front_campus_reference.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text, hidden=False):
        cells.append({"cell_type": "code", "execution_count": None, "outputs": [],
                      "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
                      "source": text.splitlines(True)})

    md("""# UTSM: retrain with Laval data
Upload this notebook to Colab, select **CPU**, then **Run all**. Upload **Laval All Telem.zip** when prompted.

**No restart loop:** training, data preparation and model inference run in separate Python processes.
This notebook does not import NumPy, pandas or scikit-learn into the interactive kernel.
It therefore works even when Colab has already loaded different versions of those libraries.

Default results remain exploratory: current calibration is unverified and voltage is constant.
The workflow reports whether the forest improves on the analytical baseline; improvement is not assumed.
""")
    md("""## 1. Prepare the worker environment
The pinned Python 3.10-3.13 packages are checked in a fresh Python process.
Existing NumPy/pandas objects in Colab's kernel are left alone. **No session restart is required.**
""")
    code("DEPENDENCIES_READY = False\n"
         "import os, sys, subprocess, importlib.util, json, hashlib\n"
         "from pathlib import Path\n"
         + "REQUIREMENTS = " + repr(requirements) + "\n"
         + "BUNDLED_ENVIRONMENT_SOURCE = " + repr(environment_source) + "\n"
         + """IN_COLAB = "google.colab" in sys.modules or Path("/content").is_dir()
WORK_DIR = Path(os.environ.get("UTSM_WORK_DIR", "/content/utsm-laval" if IN_COLAB else "laval-runtime")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
requirements_path = WORK_DIR/"requirements-colab.txt"
requirements_path.write_text(REQUIREMENTS, encoding="utf-8")
environment_path = WORK_DIR/"colab_environment.py"
environment_path.write_text(BUNDLED_ENVIRONMENT_SOURCE, encoding="utf-8")
env_spec = importlib.util.spec_from_file_location("utsm_colab_environment", environment_path)
environment = importlib.util.module_from_spec(env_spec)
sys.modules[env_spec.name] = environment
env_spec.loader.exec_module(environment)
environment.ensure_environment(requirements_path, validate_active_kernel=False)
DEPENDENCIES_READY = True
print("Workspace:", WORK_DIR)
""")
    md("""## 2. Write the bundled implementation
The notebook embeds the training source, worker and analytical reference. It needs no GitHub login or source download.
The model code is written to disk and loaded only by the separate worker.
""")
    code("if not globals().get('DEPENDENCIES_READY', False):\n"
         "    raise RuntimeError('Run the dependency cell successfully first.')\n"
         + "BUNDLED_TRAINING_SOURCE = " + repr(source) + "\n"
         + "BUNDLED_SOURCE_SHA256 = " + repr(digest) + "\n"
         + "BUNDLED_WORKER_SOURCE = " + repr(worker_source) + "\n"
         + "FRONT_CAMPUS_REFERENCE_JSON = " + repr(reference) + "\n"
         + """assert hashlib.sha256(BUNDLED_TRAINING_SOURCE.encode()).hexdigest() == BUNDLED_SOURCE_SHA256
MODULE_PATH = WORK_DIR/"laval_training.py"
MODULE_PATH.write_text(BUNDLED_TRAINING_SOURCE, encoding="utf-8")
WORKER_PATH = WORK_DIR/"laval_notebook_worker.py"
WORKER_PATH.write_text(BUNDLED_WORKER_SOURCE, encoding="utf-8")
REFERENCE_PATH = WORK_DIR/"front_campus_reference.json"
REFERENCE_PATH.write_text(FRONT_CAMPUS_REFERENCE_JSON, encoding="utf-8")
print("Training source:", BUNDLED_SOURCE_SHA256)

def run_worker(operation, output):
    command = [
        sys.executable, "-I", str(WORKER_PATH), operation,
        "--source", str(MODULE_PATH), "--archive", str(archive_path),
        "--config", str(CONFIG_PATH), "--output", str(output),
        "--reference", str(REFERENCE_PATH),
    ]
    subprocess.run(command, check=True)

# Rendering publishes raw HTML; scientific libraries are never used in the kernel.
def show_rows(rows, limit=30):
    import html
    rows = list(rows)
    if not rows:
        print("No rows")
        return
    columns = list(rows[0])
    markup = "<table><thead><tr>" + "".join("<th>"+html.escape(str(c))+"</th>" for c in columns) + "</tr></thead><tbody>"
    for row in rows[:limit]:
        markup += "<tr>" + "".join("<td>"+html.escape(str(row.get(c, "")))+"</td>" for c in columns) + "</tr>"
    markup += "</tbody></table>"
    try:
        display({"text/html": markup}, raw=True)
    except NameError:
        for row in rows[:limit]:
            print(row)
""", hidden=True)
    md("""## 3. Calibration and inputs
Leave ARCHIVE_PATH empty to upload, or enter a local/Drive ZIP path.
Defaults are assumptions, not measured calibration. Use bench records to set zero/gain/polarity.
Do not tune quality rules or parameters after viewing September 20 errors.
""")
    code("""ARCHIVE_PATH = os.environ.get("UTSM_ARCHIVE_PATH", "")
USE_GOOGLE_DRIVE = False
SAVE_RESULTS_TO_DRIVE = False
DRIVE_RESULTS_FOLDER = "/content/drive/MyDrive/UTSM-model-results"
CURRENT_MODE = "magnitude"  # magnitude | negative | positive
CURRENT_SCALE = 1.0
CURRENT_ZERO_MA = 0.0
CALIBRATION_CONFIRMED = False
CALIBRATION_OVERRIDES = {}
TEST_DATES = ["2026-09-20"]
N_TREES = int(os.environ.get("UTSM_N_ESTIMATORS", "200"))
REFIT_ALL_FOR_RELEASE = False

config = {
    "current_mode": CURRENT_MODE, "current_scale": CURRENT_SCALE,
    "current_zero_mA": CURRENT_ZERO_MA, "calibration_confirmed": CALIBRATION_CONFIRMED,
    "calibration_overrides": CALIBRATION_OVERRIDES, "test_dates": TEST_DATES,
    "n_estimators": N_TREES, "refit_all_for_release": REFIT_ALL_FOR_RELEASE,
}
CONFIG_PATH = WORK_DIR/"config.json"
CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
if not CALIBRATION_CONFIRMED:
    print("EXPLORATORY: current calibration is not confirmed.")
""")
    md("## 4. Provide the ZIP\nThe source archive is read without modification.\n")
    code("""if USE_GOOGLE_DRIVE or SAVE_RESULTS_TO_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Drive mounting is only available in Colab.")
    from google.colab import drive
    drive.mount("/content/drive")
if not ARCHIVE_PATH:
    if not IN_COLAB:
        raise ValueError("Set ARCHIVE_PATH outside Colab.")
    from google.colab import files
    uploaded = files.upload()
    names = [n for n in uploaded if n.lower().endswith(".zip")]
    if len(names) != 1:
        raise ValueError("Upload exactly one telemetry ZIP.")
    archive_path = WORK_DIR/Path(names[0]).name
    archive_path.write_bytes(uploaded[names[0]])
else:
    archive_path = Path(ARCHIVE_PATH).expanduser().resolve()
if not archive_path.is_file():
    raise FileNotFoundError(archive_path)
print("Archive:", archive_path.name)
""")
    md("""## 5. Preview quality and split
Preparation runs in the worker. Review valid duration, excluded files, current sign and sensor disagreement.
Metrics cover accepted moving intervals, not complete laps or total session energy.
""")
    code("""import csv
preview_dir = WORK_DIR/"preview"
run_worker("preview", preview_dir)
preview = json.loads((preview_dir/"preview.json").read_text(encoding="utf-8"))
with (preview_dir/"data_quality.csv").open(encoding="utf-8", newline="") as stream:
    show_rows(csv.DictReader(stream))
show_rows(preview["coverage"])
print(json.dumps(preview["split_manifest"], indent=2))
print("Archive SHA-256:", preview["source_manifest"]["archive_sha256"])
if preview["constant_voltage"]:
    print("CONSTANT VOLTAGE: labels assume recorded voltage is correct; battery sag is unobserved.")
""")
    md("""## 6. Retrain and compare
Session-grouped development tuning chooses a forest before inspecting September 20 errors.
Compare the forest with a matched analytical refit and frozen Front Campus model.
All computation runs in a fresh Python process with the verified on-disk packages.
""")
    code("""from datetime import datetime, timezone
run_name = "laval-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
output_dir = Path(os.environ.get("UTSM_OUTPUT_DIR", str(WORK_DIR/"outputs"/run_name))).resolve()
run_worker("train", output_dir)
summary = json.loads((output_dir/"summary.json").read_text(encoding="utf-8"))
print("Decision:", summary["recommendation"])
print("Results:", output_dir)
""")
    md("""## 7. Results
Lower error is better. These are retrospective held-out driving-session results.
Independent future-day data and verified calibration are required before claiming race savings.
""")
    code("""show_rows(summary["model_ranking"])
with (output_dir/"held_out_sessions.csv").open(encoding="utf-8", newline="") as stream:
    show_rows(csv.DictReader(stream))
try:
    import base64
    image_data = base64.b64encode((output_dir/"comparison.png").read_bytes()).decode("ascii")
    display({"text/html": '<img style="max-width:100%" src="data:image/png;base64,'+image_data+'">'}, raw=True)
except NameError:
    print("Chart:", output_dir/"comparison.png")
for warning in summary["warnings"]:
    print("-", warning)
""")
    md("""## 8. Reload the model and verify predictions
Serialization and inference checks also run in the worker, without importing joblib/NumPy into the kernel.
The prediction inputs exclude current, power and energy labels.
""")
    code("""run_worker("check-model", output_dir)
inference = json.loads((output_dir/"inference_example.json").read_text(encoding="utf-8"))
assert inference["passed"]
show_rows(inference["predictions"])
print("Saved-model prediction round trip passed.")
# Include the checked inference example in the final download.
import zipfile
result_zip = Path(str(output_dir)+".zip")
with zipfile.ZipFile(result_zip, "a", zipfile.ZIP_DEFLATED) as bundle:
    if "inference_example.json" not in bundle.namelist():
        bundle.write(output_dir/"inference_example.json", "inference_example.json")
""")
    md("""## 9. Save/download
Save the ZIP before ending the runtime. It contains trained models, preprocessing, metrics, source and reports.
The notebook does not replace the deployed model. A worse forest is not silently promoted.
""")
    code("""import shutil
if SAVE_RESULTS_TO_DRIVE:
    destination = Path(DRIVE_RESULTS_FOLDER)/output_dir.name
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(result_zip, destination/result_zip.name)
    print("Saved to Drive:", destination)
if IN_COLAB:
    from google.colab import files
    files.download(str(result_zip))
else:
    print("Result bundle:", result_zip)
""")
    md("""## Next steps
Verify current calibration and fixed voltage first, then add verified lap boundaries and throttle commands.
Keep a future day independent. CPU baselines should justify any later GPU experiments.
The notebook is generated from reviewed source; tests check its embedded files and execution isolation.
""")
    for i, cell in enumerate(cells):
        cell["id"] = f"laval-{i:02d}"
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.13"},
                     "colab": {"name": "retrain_laval.ipynb", "provenance": []},
                     "utsm_training_source_sha256": digest,
                     "utsm_execution_mode": "fresh-python-worker"},
        "cells": cells,
    }
    path = ROOT/"notebooks/retrain_laval.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
