"""Regressions for binary-package installs inside an already-running Colab kernel."""
import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import colab_environment as env

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_old_loaded_numpy_requires_restart_even_after_successful_install(self):
        pins = {"numpy": "2.2.6"}
        self.assertTrue(env.requires_restart(pins, {"numpy": "1.26.4"}, pins,
                                             {"numpy": "1.26.4"}))

    def test_matching_disk_but_old_kernel_requires_restart(self):
        pins = {"numpy": "2.2.6"}
        self.assertTrue(env.requires_restart(pins, pins, pins, {"numpy": "1.26.4"}))

    def test_partial_import_and_force_repair_require_restart(self):
        pins = {"numpy": "2.2.6"}
        self.assertTrue(env.requires_restart(pins, pins, pins, {"numpy": None}))
        self.assertTrue(env.requires_restart(pins, pins, pins, {"numpy": "2.2.6"}, repaired=True))

    def test_clean_or_matching_kernel_needs_no_restart(self):
        pins = {"numpy": "2.2.6"}
        self.assertFalse(env.requires_restart(pins, {}, pins, {}))
        self.assertFalse(env.requires_restart(pins, pins, pins, {"numpy": "2.2.6"}))

    def test_guard_stops_before_loading_training_module(self):
        nb = json.loads((ROOT/"notebooks/retrain_laval.ipynb").read_text(encoding="utf-8"))
        cell = next(c for c in nb["cells"] if c["cell_type"] == "code"
                    and "BUNDLED_TRAINING_SOURCE =" in "".join(c["source"]))
        with self.assertRaisesRegex(RuntimeError, "dependency cell"):
            exec(compile("".join(cell["source"]), "load-training-cell", "exec"), {})

    def test_mixed_kernel_is_rejected_after_fresh_process_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"requirements.txt"
            path.write_text("numpy==2.2.6\n")
            pins={"numpy":"2.2.6"}
            with patch.object(env.sys,"version_info",(3,13)), \
                 patch.object(env,"installed_versions",return_value=pins), \
                 patch.object(env,"loaded_versions",return_value={"numpy":"1.26.4"}), \
                 patch.object(env,"probe_fresh_process",return_value=SimpleNamespace(returncode=0,stdout="ok")), \
                 patch.object(env,"validate_kernel") as kernel_check:
                with self.assertRaisesRegex(RuntimeError,"Restart session"):
                    env.ensure_environment(path)
                kernel_check.assert_not_called()

    def test_pip_requests_binary_wheels_and_checks_active_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"requirements.txt"
            path.write_text("numpy==2.2.6\n")
            pins={"numpy":"2.2.6"}
            with patch.object(env.sys,"version_info",(3,13)), \
                 patch.object(env,"installed_versions",side_effect=[{"numpy":None},pins]), \
                 patch.object(env,"loaded_versions",return_value={}), \
                 patch.object(env.subprocess,"check_call") as pip, \
                 patch.object(env,"probe_fresh_process",return_value=SimpleNamespace(returncode=0,stdout="ok")), \
                 patch.object(env,"validate_kernel") as kernel_check:
                self.assertEqual(env.ensure_environment(path),pins)
                self.assertIn("--only-binary=:all:",pip.call_args.args[0])
                kernel_check.assert_called_once_with(pins)

    def test_corrupt_disk_install_is_repaired_then_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"requirements.txt"
            path.write_text("numpy==2.2.6\n")
            pins={"numpy":"2.2.6"}
            with patch.object(env.sys,"version_info",(3,13)), \
                 patch.object(env,"installed_versions",return_value=pins), \
                 patch.object(env,"loaded_versions",return_value={}), \
                 patch.object(env.subprocess,"check_call") as pip, \
                 patch.object(env,"probe_fresh_process",side_effect=[
                     SimpleNamespace(returncode=1,stderr="dtype size changed"),
                     SimpleNamespace(returncode=0,stdout="ok")]), \
                 patch.object(env,"validate_kernel"):
                env.ensure_environment(path)
                self.assertIn("--force-reinstall",pip.call_args.args[0])

    def test_binary_failure_becomes_actionable_restart_message(self):
        with patch.object(env.importlib,"import_module",side_effect=ValueError("numpy.dtype size changed")):
            with self.assertRaisesRegex(RuntimeError,"Restart session"):
                env.validate_kernel({"numpy":"2.2.6"})


if __name__ == "__main__":
    unittest.main()
