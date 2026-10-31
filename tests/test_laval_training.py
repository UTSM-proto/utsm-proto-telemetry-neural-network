"""Meaningful data-integrity and model-export tests; no external data/network."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import laval_training as lt


def raw(n=120, offset=0):
    t = np.arange(n)
    speed = 20 + np.sin(t/12) * 2 + offset
    return pd.DataFrame({
        "timestamp_ms": t * 500, "current_mA": -(3000 + 200*np.sin(t/10)),
        "voltage_mV": np.full(n, 24289), "temperature_C": np.full(n, 25.0),
        "temperature_valid": 1, "wheel_speed_valid": 1, "wheel_speed_kmph": speed,
        "gps_location_valid": 1, "gps_lat": 46.37, "gps_long": -70.94,
        "gps_hdop": 0.8, "gps_speed_kmph": speed,
    })


class DataTests(unittest.TestCase):
    def test_current_sign_zero_and_per_file_override(self):
        cfg = lt.Config(current_mode="negative", current_scale=2, current_zero_mA=10)
        actual = lt.calibrated_current(pd.Series([-90.0, 30.0]), cfg, "a.csv")
        np.testing.assert_allclose(actual, [200, 0])
        cfg.calibration_overrides = {"a.csv": {"current_mode": "positive", "current_scale": 1}}
        np.testing.assert_allclose(lt.calibrated_current(pd.Series([30.0]), cfg, "a.csv"), [20])

    def test_interval_energy_is_trapezoidal_and_not_target_features(self):
        frame, report = lt.prepare_session(raw(), "2026-09-19-a.csv", lt.Config())
        first = frame.iloc[0]
        r = raw()
        i = int(first.source_row)
        expected = ((-r.current_mA.iloc[i-1]) + (-r.current_mA.iloc[i]))/2/1000*24.289*0.5
        self.assertAlmostEqual(first.actual_energy_j, expected)
        self.assertTrue(report["constant_voltage"])
        self.assertFalse(set(lt.ENHANCED_FEATURES) & {"current_mA", "target_power_w", "actual_energy_j", "voltage_mV"})

    def test_no_energy_or_history_bridges_gap_or_reboot(self):
        r = raw()
        r.loc[40:, "timestamp_ms"] += 10000
        r.loc[80:, "timestamp_ms"] -= 50000
        frame, report = lt.prepare_session(r, "2026-09-19-a.csv", lt.Config())
        self.assertNotIn(40, set(frame.source_row))
        self.assertNotIn(80, set(frame.source_row))
        self.assertEqual(report["gap_rows"], 1)
        self.assertEqual(report["timestamp_reset_or_duplicate_rows"], 1)

    def test_invalid_sensor_breaks_history(self):
        r = raw()
        r.loc[40, ["wheel_speed_valid", "gps_location_valid"]] = 0
        frame, _ = lt.prepare_session(r, "2026-09-19-a.csv", lt.Config())
        self.assertNotIn(40, set(frame.source_row))
        self.assertNotIn(41, set(frame.source_row))

    def test_features_are_causal(self):
        a, _ = lt.prepare_session(raw(), "2026-09-19-a.csv", lt.Config())
        r = raw()
        r.loc[70:, ["wheel_speed_kmph", "gps_speed_kmph"]] += 10
        b, _ = lt.prepare_session(r, "2026-09-19-a.csv", lt.Config())
        np.testing.assert_allclose(
            a.loc[a.source_row < 70, lt.ENHANCED_FEATURES],
            b.loc[b.source_row < 70, lt.ENHANCED_FEATURES], equal_nan=True)

    def test_disagreement_and_invalid_temperature_not_filled_from_future(self):
        r = raw()
        r.loc[:50, "temperature_valid"] = 0
        r.loc[40, "wheel_speed_kmph"] = 50
        frame, report = lt.prepare_session(r, "2026-09-19-a.csv", lt.Config())
        self.assertEqual(report["speed_disagreement_rows"], 1)
        self.assertNotIn(40, set(frame.source_row))
        self.assertTrue(frame.loc[frame.source_row <= 50, "temperature_C"].isna().all())

    def test_archive_deduplicates_line_endings_without_extracting(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/"data.zip"
            csv = raw().to_csv(index=False)
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("2026-09-19-a.csv", csv)
                z.writestr("copies/2026-09-19-copy.csv", csv.replace("\n", "\r\n"))
            frame, quality, _ = lt.load_archive(archive, lt.Config())
            self.assertEqual(frame.session_id.nunique(), 1)
            self.assertIn("excluded_duplicate_content", quality.status.tolist())
            self.assertFalse((Path(temp)/"2026-09-19-copy.csv").exists())

    def test_stale_gps_clock_does_not_supply_speed(self):
        r = raw()
        r["wheel_speed_valid"] = 0
        r["gps_time_valid"] = 1
        r["gps_time_utc"] = "19:00:00"
        r["gps_centisecond"] = 0
        cfg = lt.Config(min_session_rows=2, min_session_duration_s=0.1)
        frame, _ = lt.prepare_session(r, "2026-09-19-a.csv", cfg)
        self.assertTrue((frame.source_row <= 4).all())
        self.assertGreater(len(frame), 0)

    def test_split_requires_dates_and_disjoint_whole_sessions(self):
        frames = [lt.prepare_session(raw(offset=i), f"2026-09-{19 if i<2 else 20}-{i}.csv", lt.Config())[0]
                  for i in range(3)]
        dev, test, manifest = lt.split_sessions(pd.concat(frames), lt.Config())
        self.assertEqual(dev.session_id.nunique(), 2)
        self.assertFalse(set(dev.session_id) & set(test.session_id))
        self.assertEqual(len(manifest["test_sessions"]), 1)
        with self.assertRaises(ValueError):
            lt.split_sessions(pd.concat(frames), lt.Config(test_dates=["2026-09-18"]))

    def test_release_requires_confirmed_calibration(self):
        with self.assertRaises(ValueError):
            lt.Config(refit_all_for_release=True).validate()


class WorkflowTests(unittest.TestCase):
    def test_full_workflow_export_roundtrip_and_no_holdout_during_tuning(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/"data.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for i in range(3):
                    z.writestr(f"2026-09-{19 if i<2 else 20}-{i}.csv",
                               raw(offset=i).to_csv(index=False))
            out = Path(temp)/"results"
            cfg = lt.Config(n_estimators=4)
            seen = []
            original = lt.train_forest
            def observe(frame, candidate, config):
                seen.append(sorted(frame.date.unique().tolist()))
                return original(frame, candidate, config)
            with patch.object(lt, "train_forest", side_effect=observe):
                summary = lt.run_workflow(archive, out, cfg, candidates=lt.CANDIDATES[:1])
            self.assertTrue(all(dates == ["2026-09-19"] for dates in seen))
            self.assertIn("EXPLORATORY", summary["status"])
            saved = joblib.load(out/"random_forest_development.joblib")
            frame, _, _ = lt.load_archive(archive, cfg)
            x = frame[frame.date == "2026-09-20"]
            a = lt.predict_power(saved, x)
            b = lt.predict_power(saved, x[saved["features"]])
            np.testing.assert_allclose(a, b)
            self.assertEqual(len(a), len(x))
            self.assertTrue(Path(str(out)+".zip").is_file())
            for file in ["source_manifest.json","split_manifest.json","data_quality.csv","feature_schema.json",
                         "held_out_sessions.csv","summary.json","report.html","comparison.png"]:
                self.assertTrue((out/file).is_file(), file)
            self.assertFalse((out/"random_forest_all_data_UNTESTED.joblib").exists())
            json.loads((out/"summary.json").read_text())
            with self.assertRaises(FileExistsError):
                lt.run_workflow(archive, out, cfg)

    def test_failed_forest_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/"data.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for i in range(3):
                    z.writestr(f"2026-09-{19 if i<2 else 20}-{i}.csv", raw(offset=i).to_csv(index=False))
            cfg = lt.Config(n_estimators=2, calibration_confirmed=True, refit_all_for_release=True)
            def controlled(bundle, frame):
                return np.zeros(len(frame)) if bundle["kind"] == "random_forest" else frame.target_power_w.to_numpy()
            with patch.object(lt, "predict_power", side_effect=controlled):
                result = lt.run_workflow(archive, Path(temp)/"out", cfg, candidates=lt.CANDIDATES[:1])
            self.assertFalse(result["forest_beats_all_baselines"])
            self.assertFalse((Path(temp)/"out/random_forest_all_data_UNTESTED.joblib").exists())

    def test_frozen_reference_known_hold_and_coast(self):
        reference = {"current_coeffs": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     "power_coeffs": [100]+[0]*12, "on_current_mA": 20000,
                     "cruise_current_mA": 1000, "median_voltage_v": 24}
        frame = pd.DataFrame({"speed_kph": [25,25], "accel_m_s2": [0,-2]})
        prediction = lt.predict_front_campus(frame, reference)
        np.testing.assert_allclose(prediction, [100,0])

    def test_missing_inference_feature_is_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Missing prediction features"):
            lt.predict_power({"schema_version": lt.SCHEMA_VERSION, "features": ["speed_kph"]}, pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
