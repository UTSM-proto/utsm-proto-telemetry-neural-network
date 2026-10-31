"""Reproducible, session-held-out Laval power-model training.

Python 3.9+; see requirements-colab.txt for the Colab environment.
Raw archives are read in memory, never extracted or modified.
This is an offline power/energy estimator, not a throttle controller.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "laval-power-v1"
CORE_FEATURES = [
    "speed_kph", "speed_squared", "positive_accel", "negative_accel",
    "positive_accel_speed", "negative_accel_speed",
]
ENHANCED_FEATURES = CORE_FEATURES + [
    "temperature_C", "temperature_missing", "speed_mean_3", "speed_std_3",
    "wheel_speed_used",
]
REQUIRED = {"timestamp_ms", "current_mA", "voltage_mV"}
CANDIDATES = [
    {"name": "rf_core_leaf20", "features": "core", "min_samples_leaf": 20, "max_depth": 12},
    {"name": "rf_enhanced_leaf20", "features": "enhanced", "min_samples_leaf": 20, "max_depth": 12},
    {"name": "rf_enhanced_leaf5", "features": "enhanced", "min_samples_leaf": 5, "max_depth": 12},
    {"name": "rf_enhanced_leaf50", "features": "enhanced", "min_samples_leaf": 50, "max_depth": None},
]


@dataclass
class Config:
    # These defaults are explicitly exploratory, not a sensor calibration claim.
    current_mode: str = "magnitude"  # magnitude | positive | negative
    current_scale: float = 1.0
    current_zero_mA: float = 0.0
    calibration_confirmed: bool = False
    calibration_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    test_dates: list[str] = field(default_factory=lambda: ["2026-09-20"])
    max_gap_s: float = 2.0
    min_speed_kph: float = 2.0
    max_speed_kph: float = 60.0
    max_abs_accel_m_s2: float = 4.0
    max_abs_current_mA: float = 60000.0
    max_hdop: float = 3.0
    max_speed_disagreement_kph: float = 10.0
    min_session_rows: int = 50
    min_session_duration_s: float = 20.0
    window_seconds: float = 10.0
    min_window_duration_s: float = 5.0
    n_estimators: int = 200
    n_jobs: int = 2
    seed: int = 42
    refit_all_for_release: bool = False

    def validate(self):
        if self.current_mode not in {"magnitude", "positive", "negative"}:
            raise ValueError("current_mode must be magnitude, positive or negative")
        if not np.isfinite(self.current_scale) or self.current_scale <= 0:
            raise ValueError("current_scale must be positive and finite")
        if not np.isfinite(self.current_zero_mA):
            raise ValueError("current_zero_mA must be finite")
        if not self.test_dates or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in self.test_dates):
            raise ValueError("Provide explicit ISO test_dates before training")
        positive = [self.max_gap_s, self.max_speed_kph, self.max_abs_accel_m_s2,
                    self.max_abs_current_mA, self.max_hdop, self.max_speed_disagreement_kph,
                    self.min_session_duration_s, self.window_seconds, self.min_window_duration_s]
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Time, speed and quality limits must be positive and finite")
        if not 0 <= self.min_speed_kph < self.max_speed_kph:
            raise ValueError("Invalid speed range")
        if self.min_session_rows < 2 or self.n_estimators < 1 or self.n_jobs < 1:
            raise ValueError("Invalid sample or forest configuration")
        if self.min_window_duration_s > self.window_seconds:
            raise ValueError("min_window_duration_s cannot exceed window_seconds")
        if self.refit_all_for_release and not self.calibration_confirmed:
            raise ValueError("Confirm calibration before requesting a release refit")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def number(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def calibrated_current(values: pd.Series, cfg: Config, filename: str) -> pd.Series:
    override = cfg.calibration_overrides.get(filename, {})
    unknown = set(override) - {"current_mode", "current_scale", "current_zero_mA"}
    if unknown:
        raise ValueError(f"Unknown calibration keys for {filename}: {sorted(unknown)}")
    mode = override.get("current_mode", cfg.current_mode)
    scale = float(override.get("current_scale", cfg.current_scale))
    zero = float(override.get("current_zero_mA", cfg.current_zero_mA))
    if mode not in {"magnitude", "positive", "negative"} or not np.isfinite(scale) or scale <= 0 or not np.isfinite(zero):
        raise ValueError(f"Invalid calibration for {filename}")
    corrected = (values - zero) * scale
    if mode == "magnitude":
        return corrected.abs()
    # Consumption only: regen is not silently counted as positive consumption.
    return (corrected if mode == "positive" else -corrected).clip(lower=0)


def causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rows must be in source order, with blocks reset after outages/reboots."""
    out = df.copy()
    groups = out.groupby(["session_id", "block_id"], sort=False)
    smooth = groups["raw_speed_kph"].transform(
        lambda s: s.rolling(3, min_periods=1).median()
    )
    out["speed_kph"] = smooth
    prev_speed = smooth.groupby([out.session_id, out.block_id], sort=False).shift()
    out["accel_m_s2"] = (smooth - prev_speed) / (3.6 * out["dt_s"])
    speed_groups = smooth.groupby([out.session_id, out.block_id], sort=False)
    out["speed_mean_3"] = speed_groups.transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["speed_std_3"] = speed_groups.transform(lambda s: s.rolling(3, min_periods=1).std(ddof=0))
    out["speed_squared"] = smooth ** 2
    out["positive_accel"] = out.accel_m_s2.clip(lower=0)
    out["negative_accel"] = (-out.accel_m_s2).clip(lower=0)
    out["positive_accel_speed"] = out.positive_accel * smooth
    out["negative_accel_speed"] = out.negative_accel * smooth
    out["temperature_missing"] = out.temperature_C.isna().astype(float)
    return out


def prepare_session(raw: pd.DataFrame, filename: str, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Build valid observed-driving intervals without bridging invalid rows."""
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")
    df = raw.copy().reset_index(drop=True)
    stamp = re.match(r"(\d{4}-\d{2}-\d{2})", Path(filename).name)
    if not stamp:
        raise ValueError(f"{filename}: date must begin YYYY-MM-DD")
    session = Path(filename).stem
    df["session_id"] = session
    df["date"] = stamp.group(1)
    df["source_row"] = np.arange(len(df))
    ts = number(df, "timestamp_ms")
    current_raw = number(df, "current_mA")
    current = calibrated_current(current_raw, cfg, Path(filename).name)
    voltage = number(df, "voltage_mV") / 1000.0
    dt = ts.diff() / 1000.0
    temporal = dt.gt(0) & dt.le(cfg.max_gap_s) & ts.notna() & ts.shift().notna()

    wheel = number(df, "wheel_speed_kmph")
    if "wheel_speed_kph" in df:
        wheel = number(df, "wheel_speed_kph")
    gps = number(df, "gps_speed_kmph")
    wheel_ok = number(df, "wheel_speed_valid", 0).eq(1) & wheel.between(0, cfg.max_speed_kph)
    lat = number(df, "gps_lat")
    lon = number(df, "gps_long")
    gps_ok = (number(df, "gps_location_valid", 0).eq(1)
              & gps.between(0, cfg.max_speed_kph)
              & lat.between(-90, 90) & lon.between(-180, 180)
              & (lat.abs() + lon.abs()).gt(0)
              & number(df, "gps_hdop").between(0, cfg.max_hdop))
    if "gps_time_valid" in df:
        gps_ok &= number(df, "gps_time_valid", 0).eq(1)
    if "gps_time_utc" in df:
        clock = df["gps_time_utc"].astype(str)
        if "gps_centisecond" in df:
            clock = clock + ":" + df["gps_centisecond"].astype(str)
        clock_changed = clock.ne(clock.shift()) | dt.le(0)
        last_clock_update_ms = ts.where(clock_changed).ffill()
        gps_ok &= (ts - last_clock_update_ms).between(0, cfg.max_gap_s * 1000)
    # Reject disagreement instead of choosing the sensor that makes a model win.
    disagreement = wheel_ok & gps_ok & (wheel - gps).abs().gt(cfg.max_speed_disagreement_kph)
    speed = wheel.where(wheel_ok, gps.where(gps_ok)).mask(disagreement)
    temperature = number(df, "temperature_C")
    if "motor_temperature_C" in df:
        temperature = number(df, "motor_temperature_C")
    temp_flag = "motor_temperature_valid" if "motor_temperature_valid" in df else "temperature_valid"
    temperature = temperature.where(number(df, temp_flag, 0).eq(1) & temperature.between(-40, 180))
    electrical = (current_raw.abs().le(cfg.max_abs_current_mA)
                  & voltage.between(18, 30) & current.notna())
    sensor_ok = electrical & speed.notna()
    # Every invalid reading or timestamp discontinuity begins a new causal block.
    boundaries = (~temporal) | (~sensor_ok) | (~sensor_ok.shift(fill_value=False))
    block = boundaries.cumsum()
    df["block_id"] = block
    df["time_s"] = ts / 1000.0
    df["dt_s"] = dt
    df["raw_speed_kph"] = speed
    df["temperature_C"] = temperature
    df["wheel_speed_used"] = wheel_ok.astype(float)
    df["corrected_current_mA"] = current
    df["voltage_v"] = voltage
    power = current / 1000.0 * voltage
    # Target is interval-average power; integration uses the same intervals for all models.
    df["target_power_w"] = (power + power.shift()) / 2.0
    valid_interval = temporal & sensor_ok & sensor_ok.shift(fill_value=False)
    df = causal_features(df)
    valid = (valid_interval & df.speed_kph.between(cfg.min_speed_kph, cfg.max_speed_kph)
             & df.accel_m_s2.abs().le(cfg.max_abs_accel_m_s2))
    df["actual_energy_j"] = df.target_power_w * df.dt_s
    finite = np.isfinite(df[CORE_FEATURES + ["target_power_w", "dt_s"]].to_numpy()).all(axis=1)
    valid &= finite
    kept = df.loc[valid].copy()
    # Separate report groups at every excluded interval; no invented whole laps.
    continuity = kept.source_row.diff().ne(1) | kept.block_id.diff().ne(0)
    kept["evaluation_block"] = continuity.cumsum().astype(int)
    report = {
        "filename": filename, "session_id": session, "date": stamp.group(1),
        "raw_rows": len(raw), "accepted_rows": len(kept),
        "accepted_duration_s": float(kept.dt_s.sum()),
        "file_span_s": float((ts.max() - ts.min()) / 1000) if ts.notna().any() else 0,
        "negative_current_rows": int(current_raw.lt(0).sum()),
        "gps_valid_rows": int(gps_ok.sum()), "wheel_valid_rows": int(wheel_ok.sum()),
        "speed_disagreement_rows": int(disagreement.sum()),
        "timestamp_reset_or_duplicate_rows": int(dt.le(0).sum()),
        "gap_rows": int(dt.gt(cfg.max_gap_s).sum()),
        "constant_voltage": bool(voltage.nunique(dropna=True) <= 1),
        "voltage_median_v": float(voltage.median()) if voltage.notna().any() else None,
        "status": "accepted",
    }
    if len(kept) < cfg.min_session_rows or kept.dt_s.sum() < cfg.min_session_duration_s:
        report["status"] = "excluded_insufficient_valid_driving"
        report["eligible_rows_before_session_filter"] = len(kept)
        report["accepted_rows"] = 0
        report["accepted_duration_s"] = 0.0
        kept = kept.iloc[:0]
    return kept, report


def load_archive(archive: str | Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg.validate()
    archive = Path(archive)
    if archive.stat().st_size > 500_000_000:
        raise ValueError("Archive exceeds 500 MB; split the archive before training")
    source = archive.read_bytes()
    if len(source) > 500_000_000:
        raise ValueError("Archive exceeds 500 MB; split the archive before training")
    frames, reports, hashes = [], [], {}
    seen, seen_rows, session_ids = set(), set(), set()
    with zipfile.ZipFile(io.BytesIO(source)) as z:
        infos = [i for i in z.infolist() if i.filename.lower().endswith(".csv") and not i.is_dir()
                 and not any(p.startswith(".") or p == "__MACOSX" for p in Path(i.filename).parts)]
        if sum(i.file_size for i in infos) > 1_000_000_000:
            raise ValueError("Uncompressed CSV contents exceed 1 GB")
        for info in sorted(infos, key=lambda i: i.filename):
            data = z.read(info)  # no path extraction
            normalized = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            digest = sha256(normalized.encode())
            hashes[info.filename] = {"sha256": sha256(data), "normalized_sha256": digest, "bytes": len(data)}
            if digest in seen:
                reports.append({"filename": info.filename, "status": "excluded_duplicate_content"})
                continue
            seen.add(digest)
            try:
                raw = pd.read_csv(io.StringIO(normalized))
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                reports.append({"filename": info.filename, "status": "excluded_unreadable", "reason": str(exc)})
                continue
            semantic = sha256(json.dumps(raw.columns.tolist()).encode() + pd.util.hash_pandas_object(raw, index=False).to_numpy().tobytes())
            hashes[info.filename]["parsed_rows_sha256"] = semantic
            if semantic in seen_rows:
                reports.append({"filename": info.filename, "status": "excluded_duplicate_content"})
                continue
            seen_rows.add(semantic)
            if not REQUIRED.issubset(raw.columns):
                reports.append({"filename": info.filename, "status": "excluded_missing_columns",
                                "reason": str(sorted(REQUIRED - set(raw.columns)))})
                continue
            frame, report = prepare_session(raw, info.filename, cfg)
            if report["session_id"] in session_ids:
                raise ValueError("Duplicate session basename in archive: " + report["session_id"])
            session_ids.add(report["session_id"])
            reports.append(report)
            if not frame.empty:
                frames.append(frame)
    unknown_overrides = set(cfg.calibration_overrides) - {Path(n).name for n in hashes}
    if unknown_overrides:
        raise ValueError("Calibration overrides name missing CSVs: " + str(sorted(unknown_overrides)))
    if not frames:
        raise ValueError("No usable driving intervals. Inspect calibration, sensor flags and data schema.")
    frame = pd.concat(frames, ignore_index=True)
    if frame[["session_id", "source_row"]].duplicated().any():
        raise AssertionError("Duplicate source rows")
    # A reboot must not create a separate training/test group for the same CSV.
    provenance = {"archive": archive.name, "archive_sha256": sha256(source), "files": hashes}
    return frame, pd.DataFrame(reports), provenance


def split_sessions(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    test = frame[frame.date.isin(cfg.test_dates)].copy()
    dev = frame[~frame.date.isin(cfg.test_dates)].copy()
    if test.empty:
        raise ValueError("No valid test sessions on configured test_dates")
    if dev.session_id.nunique() < 2:
        raise ValueError("Need at least two usable development sessions for grouped tuning")
    if dev.date.max() >= test.date.min():
        raise ValueError("Development dates must precede test dates; do not randomly split rows")
    if set(dev.session_id) & set(test.session_id):
        raise AssertionError("Session leakage")
    # Exact numeric rows can recur naturally at rest; duplicate timestamps/exports need stronger checks.
    # Identical sequences across files are excluded above by normalized content hash.
    manifest = {"development_sessions": sorted(dev.session_id.unique().tolist()),
                "test_sessions": sorted(test.session_id.unique().tolist()),
                "test_dates": cfg.test_dates, "development_rows": len(dev), "test_rows": len(test)}
    return dev, test, manifest


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    # Equal total weight per session, distributed by actual observed duration.
    session_seconds = frame.groupby("session_id").dt_s.transform("sum").to_numpy()
    weights = frame.dt_s.to_numpy() / session_seconds
    return weights * len(frame) / frame.session_id.nunique()


def feature_names(kind: str) -> list[str]:
    if kind == "core":
        return list(CORE_FEATURES)
    if kind == "enhanced":
        return list(ENHANCED_FEATURES)
    raise ValueError(f"Unknown feature set: {kind}")


def train_forest(frame: pd.DataFrame, candidate: dict, cfg: Config) -> dict:
    names = feature_names(candidate["features"])
    estimator = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        RandomForestRegressor(n_estimators=cfg.n_estimators, random_state=cfg.seed,
                              n_jobs=cfg.n_jobs, min_samples_leaf=candidate["min_samples_leaf"],
                              max_depth=candidate["max_depth"]),
    )
    estimator.fit(frame[names], frame.target_power_w,
                  randomforestregressor__sample_weight=sample_weights(frame))
    return {"schema_version": SCHEMA_VERSION, "kind": "random_forest", "features": names,
            "estimator": estimator, "candidate": candidate}


def train_analytical(frame: pd.DataFrame) -> dict:
    # A matched-data analytical refit; distinct from the frozen Front Campus deployment artifact.
    # Standardization parameters are learned only from this fold.
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(frame[CORE_FEATURES], frame.target_power_w,
                  ridge__sample_weight=sample_weights(frame))
    return {"schema_version": SCHEMA_VERSION, "kind": "analytical_refit",
            "features": list(CORE_FEATURES), "estimator": estimator}


def predict_power(bundle: dict, feature_frame: pd.DataFrame) -> np.ndarray:
    """Predict interval power from canonical causal features; no label columns required."""
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported feature schema")
    missing = set(bundle["features"]) - set(feature_frame.columns)
    if missing:
        raise ValueError("Missing prediction features: " + str(sorted(missing)))
    return np.maximum(bundle["estimator"].predict(feature_frame[bundle["features"]]), 0.0)


def predict_front_campus(frame: pd.DataFrame, reference: dict) -> np.ndarray:
    """Frozen historical model electrical rule, no motor physics floor, neutral position.

    Historical acceleration-action threshold is 1.0 in classify_strategy_action;
    preserve it for the frozen reference rather than silently changing the incumbent.
    Grade is zero: unreliable GPS altitude is not treated as surveyed grade.
    """
    speed = frame.speed_kph.to_numpy()
    accel = frame.accel_m_s2.to_numpy()
    pos, neg = np.maximum(accel, 0), np.maximum(-accel, 0)
    action_acc = accel > 1.0
    action_coast = accel < -1.0
    action_hold = ~(action_acc | action_coast)
    zeros = np.zeros(len(frame))
    design = np.column_stack([np.ones(len(frame)), speed, speed**2, pos, neg, zeros, zeros,
                              zeros, action_acc, action_hold, action_coast, pos*speed, neg*speed])
    raw_current = np.maximum(design @ np.asarray(reference["current_coeffs"]), 0)
    raw_power = np.maximum(design @ np.asarray(reference["power_coeffs"]), 0)
    voltage = float(reference["median_voltage_v"])
    on = np.full(len(frame), float(reference["on_current_mA"]))
    cruise = float(reference["cruise_current_mA"])
    duty_acc = np.clip(0.22 + pos*1.8, 0.18, 1.0)
    on = np.where(action_acc, np.maximum(on, raw_current / np.maximum(duty_acc, 0.05)), on)
    avg = np.where(action_acc, np.maximum(raw_current, on*duty_acc), np.maximum(raw_current, cruise))
    on = np.where(action_hold, np.maximum(on, avg*1.35), on)
    duty_hold = np.clip(avg/np.maximum(on, 1), 0.04, 0.85)
    avg = np.where(action_hold, on*duty_hold, avg)
    power = np.maximum(raw_power, avg/1000*voltage)
    return np.where(action_coast, 0.0, power)


def evaluation(frame: pd.DataFrame, predicted_power: np.ndarray, name: str, cfg: Config):
    if len(frame) != len(predicted_power):
        raise ValueError("Prediction count does not match evaluation rows")
    rows = frame[["session_id", "date", "source_row", "time_s", "block_id",
                  "evaluation_block", "dt_s", "actual_energy_j", "target_power_w"]].copy()
    rows["model"] = name
    rows["predicted_power_w"] = np.asarray(predicted_power)
    rows["predicted_energy_j"] = rows.predicted_power_w * rows.dt_s
    rows["absolute_power_error_w"] = (rows.predicted_power_w - rows.target_power_w).abs()
    grouped = rows.groupby(["model", "session_id", "date"], sort=True)
    sessions = grouped.agg(actual_energy_j=("actual_energy_j", "sum"),
                           predicted_energy_j=("predicted_energy_j", "sum"),
                           valid_duration_s=("dt_s", "sum"),
                           power_mae_w=("absolute_power_error_w", "mean")).reset_index()
    denom = sessions.actual_energy_j.replace(0, np.nan)
    sessions["signed_error_pct"] = (sessions.predicted_energy_j - sessions.actual_energy_j) / denom * 100
    sessions["absolute_error_pct"] = sessions.signed_error_pct.abs()
    rows["window"] = np.floor(rows.time_s / cfg.window_seconds).astype(int)
    windows = rows.groupby(["model", "session_id", "evaluation_block", "window"], sort=True).agg(
        actual_energy_j=("actual_energy_j", "sum"), predicted_energy_j=("predicted_energy_j", "sum"),
        valid_duration_s=("dt_s", "sum")).reset_index()
    windows = windows[windows.valid_duration_s >= cfg.min_window_duration_s].copy()
    windows["absolute_error_pct"] = ((windows.predicted_energy_j - windows.actual_energy_j).abs()
                                    / windows.actual_energy_j.replace(0, np.nan) * 100)
    return sessions, windows, rows


def tune_forest(dev: pd.DataFrame, cfg: Config, candidates: list[dict] | None = None):
    candidates = candidates or CANDIDATES
    cv = GroupKFold(n_splits=min(3, dev.session_id.nunique()))
    folds = list(cv.split(dev, groups=dev.session_id))
    results = []
    for candidate in candidates:
        all_sessions = []
        for fold, (train_index, val_index) in enumerate(folds):
            train, val = dev.iloc[train_index], dev.iloc[val_index]
            assert not set(train.session_id) & set(val.session_id)
            bundle = train_forest(train, candidate, cfg)
            metrics, _, _ = evaluation(val, predict_power(bundle, val), candidate["name"], cfg)
            metrics["fold"] = fold
            all_sessions.append(metrics)
        metrics = pd.concat(all_sessions, ignore_index=True)
        score = float(metrics.absolute_error_pct.mean())
        if not np.isfinite(score):
            raise ValueError("No finite development CV score; inspect zero-energy sessions")
        results.append({"candidate": candidate["name"], "cv_mean_session_error_pct": score,
                        "features": candidate["features"], "min_samples_leaf": candidate["min_samples_leaf"],
                        "max_depth": candidate["max_depth"]})
    result = pd.DataFrame(results).sort_values(["cv_mean_session_error_pct", "candidate"]).reset_index(drop=True)
    selected = next(c for c in candidates if c["name"] == result.iloc[0].candidate)
    return selected, result


def save_report(out: Path, summary: dict, sessions: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pivot = sessions.pivot(index="session_id", columns="model", values="absolute_error_pct")
    ax = pivot.plot.bar(figsize=(11, 5), ylabel="Absolute covered-driving energy error (%)",
                        title="Held-out Laval sessions (lower is better)")
    ax.set_xlabel("Session")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=160)
    plt.close()
    ranking = pd.DataFrame(summary["model_ranking"])
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Laval model report</title>"
            "<style>body{max-width:1000px;margin:40px auto;font:16px Arial;color:#123}"
            "td,th{padding:9px;border-bottom:1px solid #ccd}img{max-width:100%}</style></head><body>"
            "<h1>Laval Random Forest vs analytical</h1><p><b>" + summary["status"] + "</b></p>"
            "<p>Same held-out covered-driving intervals. These are not whole-lap errors or strategy savings.</p>"
            + "<p><b>" + summary["recommendation"] + "</b></p>"
            + ranking.to_html(index=False, float_format=lambda x: f"{x:.2f}")
            + "<h2>Warnings</h2><ul>" + "".join("<li>" + w + "</li>" for w in summary["warnings"])
            + "</ul><img src='comparison.png'><h2>Per-session results</h2>"
            + sessions.to_html(index=False, float_format=lambda x: f"{x:.2f}") + "</body></html>")
    (out / "report.html").write_text(html, encoding="utf-8")


def run_workflow(archive: str | Path, output: str | Path, cfg: Config,
                 reference_path: str | Path | None = None, candidates=None) -> dict:
    cfg.validate()
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty/new output directory; existing results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    frame, quality, provenance = load_archive(archive, cfg)
    provenance["implementation_sha256"] = sha256(Path(__file__).read_bytes())
    quality.to_csv(out / "data_quality.csv", index=False)
    (out / "source_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    frame.to_csv(out / "canonical_intervals.csv.gz", index=False, compression="gzip")
    dev, test, split = split_sessions(frame, cfg)
    (out / "split_manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    selected, cv_scores = tune_forest(dev, cfg, candidates)
    cv_scores.to_csv(out / "development_cv.csv", index=False)
    forest = train_forest(dev, selected, cfg)
    analytical = train_analytical(dev)
    # Do not inspect test scores until feature/model selection has completed.
    predictions = {"random_forest": predict_power(forest, test),
                   "analytical_refit": predict_power(analytical, test)}
    if reference_path is not None:
        reference_bytes = Path(reference_path).read_bytes()
        reference = json.loads(reference_bytes)
        predictions["frozen_front_campus"] = predict_front_campus(test, reference)
        (out / "front_campus_reference.json").write_bytes(reference_bytes)
        provenance["front_campus_reference_sha256"] = sha256(reference_bytes)
        (out / "source_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    session_tables, window_tables, prediction_tables = [], [], []
    ranking = []
    for name, pred in predictions.items():
        sessions, windows, rows = evaluation(test, pred, name, cfg)
        session_tables.append(sessions)
        window_tables.append(windows)
        prediction_tables.append(rows)
        finite = sessions.absolute_error_pct.dropna()
        if finite.empty:
            raise ValueError("No positive-energy held-out sessions; percentage errors are undefined")
        ranking.append({"model": name, "mean_session_error_pct": float(finite.mean()),
                        "median_session_error_pct": float(finite.median()),
                        "p90_session_error_pct": float(finite.quantile(0.9)),
                        "mean_signed_session_error_pct": float(sessions.signed_error_pct.mean()),
                        "window_energy_mae_wh": float((windows.predicted_energy_j-windows.actual_energy_j).abs().mean()/3600)
                        if len(windows) else None})
    sessions = pd.concat(session_tables, ignore_index=True)
    sessions.to_csv(out / "held_out_sessions.csv", index=False)
    pd.concat(window_tables, ignore_index=True).to_csv(out / "held_out_windows.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(out / "held_out_predictions.csv.gz", index=False, compression="gzip")
    warnings = [
        "Only September 19-20 Laval data; no untouched future driving day is established.",
        "Errors cover accepted moving intervals, not whole laps or all recorded session energy.",
        "No measured throttle channel: this estimator does not establish counterfactual strategy savings.",
        "The analytical refit uses matched core features; it is not the historical deployment simulator.",
        "Frozen Front Campus uses zero grade and no additional motor-physics floor; this is observed-driving replay.",
        "This archive has already been explored in an audit; this is retrospective session-held-out evaluation, not a new prospective field test.",
        "Repeated tuning after viewing these test errors invalidates the holdout."
    ]
    if not cfg.calibration_confirmed:
        warnings.insert(0, "UNVERIFIED current calibration: all models and errors are exploratory only.")
    if quality.get("constant_voltage", pd.Series(dtype=bool)).fillna(False).any():
        warnings.insert(0, "Constant recorded voltage detected: energy labels assume recorded voltage is correct; battery sag is unobserved.")
    status = "CALIBRATION CONFIRMED BY USER; FIELD VALIDATION STILL REQUIRED" if cfg.calibration_confirmed else "EXPLORATORY - CALIBRATION NOT CONFIRMED"
    forest.update({"config": asdict(cfg), "source_manifest": provenance, "split_manifest": split,
                   "status": status, "training_scope": "development_sessions_only"})
    analytical.update({"config": asdict(cfg), "status": status, "training_scope": "development_sessions_only"})
    joblib.dump(forest, out / "random_forest_development.joblib")
    joblib.dump(analytical, out / "analytical_development.joblib")
    (out / "feature_schema.json").write_text(json.dumps({
        "version": SCHEMA_VERSION, "target": "trapezoidal interval-average recorded electrical power (W)",
        "random_forest_features": forest["features"], "analytical_features": CORE_FEATURES,
        "preprocessing": "causal trailing-3 speed median/history; reset at session/outage; fold-fitted imputation",
        "inference": "predict_power(bundle, canonical_feature_dataframe); raw telemetry requires prepare_session",
    }, indent=2), encoding="utf-8")
    summary = {"status": status, "selected_candidate": selected, "warnings": warnings,
               "development_sessions": split["development_sessions"], "test_sessions": split["test_sessions"],
               "accepted_rows": len(frame), "development_rows": len(dev), "test_rows": len(test),
               "accepted_duration_s": float(frame.dt_s.sum()), "test_duration_s": float(test.dt_s.sum()),
               "model_ranking": sorted(ranking, key=lambda r: r["mean_session_error_pct"]),
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                               "joblib": joblib.__version__}}
    # Explicitly optional and separately named; it has no independent test after this refit.
    baseline_errors = [r["mean_session_error_pct"] for r in ranking if r["model"] != "random_forest"]
    rf_error = next(r["mean_session_error_pct"] for r in ranking if r["model"] == "random_forest")
    improvement = bool(rf_error < min(baseline_errors))
    summary["forest_beats_all_baselines"] = improvement
    summary["recommendation"] = "Candidate improves retrospective energy error; obtain a fresh-day test." if improvement else "No improvement over every baseline demonstrated. Keep the best existing baseline."
    if cfg.refit_all_for_release and improvement:
        release = train_forest(frame, selected, cfg)
        release.update({"config": asdict(cfg), "source_manifest": provenance,
                        "status": "REFIT ON ALL DATA - NEEDS A NEW FUTURE TEST",
                        "training_scope": "all_sessions_including_previous_test"})
        joblib.dump(release, out / "random_forest_all_data_UNTESTED.joblib")
    if cfg.refit_all_for_release and not improvement:
        summary["warnings"].append("All-data release refit skipped: forest did not beat every analytical baseline.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    save_report(out, summary, sessions)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except (OSError, subprocess.CalledProcessError):
        freeze = "pip freeze unavailable; see summary.json for package versions\n"
    (out / "environment.txt").write_text(freeze, encoding="utf-8")
    # Copy the actual training implementation so results remain usable away from GitHub.
    (out / "laval_training.py").write_bytes(Path(__file__).read_bytes())
    with zipfile.ZipFile(str(out) + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file():
                z.write(path, arcname=path.name)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--reference", help="Optional frozen Front Campus coefficient JSON")
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    run_workflow(args.archive, args.output, cfg, args.reference)


if __name__ == "__main__":
    main()
