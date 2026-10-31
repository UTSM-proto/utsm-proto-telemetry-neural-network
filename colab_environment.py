"""Standard-library-only environment setup for Colab's in-process Python kernel."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

MODULES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
           "scikit-learn": "sklearn", "matplotlib": "matplotlib", "joblib": "joblib"}
RESTART_MESSAGE = (
    "Packages on disk are ready, but this kernel still contains old scientific-library binaries. "
    "In Colab choose Runtime > Restart session (or Restart runtime), then Run all again. "
    "Do not rerun only the model-loading cell."
)


def parse_pins(text):
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        if not sep or name not in MODULES or not version:
            raise ValueError("Expected supported exact dependency pins; got: " + line)
        pins[name] = version
    return pins


def installed_versions(pins):
    versions = {}
    for package in pins:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def loaded_versions(pins):
    return {package: vars(sys.modules[MODULES[package]]).get("__version__")
            for package in pins if MODULES[package] in sys.modules}


def requires_restart(pins, before, after, loaded, repaired=False):
    # Includes partially imported packages, which can lack __version__.
    return any(repaired or before.get(package) != after.get(package)
               or loaded_version != pins[package]
               for package, loaded_version in loaded.items())


def validate_kernel(pins):
    try:
        for package, expected in pins.items():
            module = importlib.import_module(MODULES[package])
            actual = getattr(module, "__version__", None)
            if actual != expected:
                raise RuntimeError(f"{package}: loaded {actual}, expected {expected}")
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Ridge
        # Exercise the random C extension from the user's failing traceback.
        np.random.default_rng(0).normal(size=2)
        np.random.RandomState(0).normal(size=2)
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        Ridge().fit(data, [0.0, 1.0, 2.0]).predict(data)
    except Exception as exc:
        raise RuntimeError(RESTART_MESSAGE + "\nKernel check: " + str(exc)) from exc


def probe_fresh_process(pins):
    probe = """
import importlib, json, sys
pins = json.loads(sys.argv[1])
modules = json.loads(sys.argv[2])
for package, expected in pins.items():
    module = importlib.import_module(modules[package])
    assert module.__version__ == expected, (package, module.__version__, expected)
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
np.random.default_rng(0).normal(size=2)
np.random.RandomState(0).normal(size=2)
x = pd.DataFrame({'x': [0., 1., 2.]})
Ridge().fit(x, [0., 1., 2.]).predict(x)
print('Fresh-process scientific import and binary checks passed.')
"""
    return subprocess.run([sys.executable, "-c", probe, json.dumps(pins), json.dumps(MODULES)],
                          text=True, capture_output=True)


def ensure_environment(requirements_path):
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("This pinned notebook supports Python 3.10-3.13; choose a supported Colab runtime.")
    path = Path(requirements_path)
    pins = parse_pins(path.read_text(encoding="utf-8"))
    before = installed_versions(pins)
    # Check sys.modules only; never import a binary package before installation.
    loaded = loaded_versions(pins)
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "--only-binary=:all:",
               "--no-cache-dir", "-r", str(path)]
    changed = before != pins
    if changed:
        subprocess.check_call(command)
    importlib.invalidate_caches()
    probe = probe_fresh_process(pins)
    repaired = False
    if probe.returncode:
        # Metadata can claim correct versions while partial installs contain stale binaries.
        subprocess.check_call(command + ["--force-reinstall"])
        repaired = True
        importlib.invalidate_caches()
        probe = probe_fresh_process(pins)
    if probe.returncode:
        raise RuntimeError("Scientific package checks failed even in a fresh Python process. "
                           "Use a clean Colab runtime and rerun installation.\n" + probe.stderr[-4000:])
    after = installed_versions(pins)
    if after != pins:
        raise RuntimeError("Installed versions do not match the notebook pins: " + str(after))
    if requires_restart(pins, before, after, loaded, repaired):
        raise RuntimeError(RESTART_MESSAGE)
    # Catch partially loaded NumPy internals even if __version__ appears correct.
    validate_kernel(pins)
    print(probe.stdout.strip())
    print("Python:", sys.version.split()[0])
    print("Ready:", json.dumps(after, sort_keys=True))
    return after
