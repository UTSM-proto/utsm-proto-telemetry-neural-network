"""Checks for the differentiable calibration and portable checkpoint."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from calibrate_physics import LOWER, UPPER, physical_power, predict, solve


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(
            speed_kph=[10.,20.,30.,25.,15.,22.],
            accel_m_s2=[0.,.2,-.3,.1,0.,.05], grade_pct=[0.]*6,
            dt_s=[1.]*6, run=["a"]*3+["b"]*3))

    def test_physical_gradient(self):
        parameters = np.array([.1,-.2,.2,-.1])
        _, derivative = physical_power(self.frame, parameters)
        for index in range(4):
            delta = np.zeros(4)
            delta[index] = 1e-5
            numeric = (physical_power(self.frame, parameters + delta)[0]
                       - physical_power(self.frame, parameters - delta)[0]) / 2e-5
            np.testing.assert_allclose(derivative[:,index], numeric, rtol=1e-5, atol=1e-6)

    def test_recovery_bounds_and_reload(self):
        physical, _ = physical_power(self.frame, np.zeros(4))
        self.frame["target_w"] = np.logaddexp(0, physical*20)/20*200
        model, diagnostics = solve(self.frame, None, dict(
            kind="physics_only", energy_weight=1., physics_weight=0., ridge=.01))
        parameters = np.array(list(diagnostics["physical_parameters"].values()))
        self.assertTrue(np.all(parameters >= LOWER) and np.all(parameters <= UPPER))
        np.testing.assert_allclose(predict(model,self.frame), self.frame.target_w, atol=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"calibrated.joblib"
            joblib.dump(model,path)
            np.testing.assert_allclose(predict(model,self.frame),
                                       predict(joblib.load(path),self.frame), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
