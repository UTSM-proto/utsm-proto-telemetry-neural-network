"""Calibrate bounded road-load parameters and a pretrained neural residual head."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

FEATURES = ["speed_kph", "accel_m_s2"]
PARAMETERS = ["total_mass_kg", "rolling_resistance", "cd_area_m2", "efficiency"]
PRIOR = np.array([100.0, 0.003, 0.24 * 0.293, 0.82])
PRIOR_SCALE = np.array([25.0, 0.002, 0.03, 0.15])
LOWER = np.array([60.0, 0.001, 0.04, 0.5])
UPPER = np.array([140.0, 0.010, 0.13, 0.95])
POWER_SCALE = 200.0


def physical_power(frame, normalized):
    """Signed tractive force, clipped for consumption-only battery labels."""
    mass, crr, cda, efficiency = PRIOR + PRIOR_SCALE * normalized
    speed = frame.speed_kph.to_numpy() / 3.6
    accel = frame.accel_m_s2.to_numpy()
    grade = np.asarray(frame.get("grade_pct", np.zeros(len(frame)))) / 100
    force_per_mass = accel + 9.80665 * (crr + grade)
    force = mass * force_per_mass + 0.5 * 1.225 * cda * speed**2
    active = force > 0
    power = np.maximum(force, 0) * speed / efficiency
    jacobian = np.column_stack([
        force_per_mass * speed / efficiency * active,
        9.80665 * mass * speed / efficiency * active,
        0.5 * 1.225 * speed**3 / efficiency * active,
        -power / efficiency,
    ]) * PRIOR_SCALE
    return power / POWER_SCALE, jacobian / POWER_SCALE


def representation(frame, kind, encoder=None):
    speed = frame.speed_kph.to_numpy() / 30
    accel = frame.accel_m_s2.to_numpy()
    core = np.column_stack([
        speed, speed**2, speed**3, np.maximum(accel, 0),
        np.minimum(accel, 0), speed * accel, speed * np.abs(accel),
    ])
    if kind == "physics_only":
        return np.empty((len(frame), 0))
    if kind == "compact":
        return core
    hidden = encoder.steps[0][1].transform(frame[FEATURES])
    mlp = encoder.steps[-1][1]
    for weight, bias in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        hidden = np.maximum(hidden @ weight + bias, 0)
    return np.column_stack([core, hidden])


def integration(frame):
    groups, index = np.unique(frame.run, return_inverse=True)
    matrix = np.eye(len(groups))[index].T * frame.dt_s.to_numpy()
    return matrix * POWER_SCALE / 3600


def prepared_features(frame, kind, encoder, stats=None):
    raw = representation(frame, kind, encoder)
    if stats is None:
        stats = (raw.mean(axis=0), np.maximum(raw.std(axis=0), 0.01))
    mean, scale = stats
    normalized = (raw - mean) / scale
    # physics_only has no intercept or learned correction.
    if kind != "physics_only":
        normalized = np.column_stack([np.ones(len(frame)), normalized])
    return normalized, stats


def solve(frame, encoder, options, max_nfev=80):
    features, stats = prepared_features(frame, options["kind"], encoder)
    energy = integration(frame)
    weights = (energy / energy.sum(axis=1, keepdims=True)).mean(axis=0)
    measured = frame.target_w.to_numpy() / POWER_SCALE
    measured_energy = energy @ measured
    n = features.shape[1]
    prior_weight = options.get("prior_weight", 0.01)
    root_interval = np.sqrt(weights)
    root_physics = np.sqrt(options["physics_weight"] * weights)
    root_energy = np.sqrt(options["energy_weight"] / len(energy)) / 10
    root_ridge = np.sqrt(options["ridge"])
    root_prior = np.sqrt(prior_weight)

    def residual_and_jacobian(parameters):
        physics, derivative = physical_power(frame, parameters[:4])
        correction = features @ parameters[4:]
        raw = physics + correction
        prediction = np.logaddexp(0, raw * 20) / 20
        derivative = np.column_stack([derivative, features]) * expit(raw * 20)[:, None]
        data_residual = (prediction - measured) * root_interval
        energy_residual = (energy @ prediction - measured_energy) * root_energy
        # Explicit simulator-consistency loss, weighted independently of the prior.
        physics_residual = (prediction - physics) * root_physics
        physics_derivative = derivative.copy()
        physics_derivative[:, :4] -= physical_power(frame, parameters[:4])[1]
        regularization = np.r_[parameters[:4] * root_prior, parameters[4:] * root_ridge]
        regularization_jacobian = np.diag(np.r_[np.full(4, root_prior), np.full(n, root_ridge)])
        return np.r_[data_residual, energy_residual, physics_residual, regularization], np.vstack([
            derivative * root_interval[:, None],
            energy @ derivative * root_energy,
            physics_derivative * root_physics[:, None],
            regularization_jacobian,
        ])

    cache = {}
    def evaluate(parameters):
        if "parameters" not in cache or not np.array_equal(cache["parameters"], parameters):
            residual, jacobian = residual_and_jacobian(parameters)
            cache.update(parameters=parameters.copy(), residual=residual, jacobian=jacobian)
        return cache

    bounds = (np.r_[(LOWER - PRIOR) / PRIOR_SCALE, np.full(n, -np.inf)],
              np.r_[(UPPER - PRIOR) / PRIOR_SCALE, np.full(n, np.inf)])
    initial = np.zeros(4 + n)
    result = least_squares(
        lambda p: evaluate(p)["residual"], initial,
        jac=lambda p: evaluate(p)["jacobian"], bounds=bounds, max_nfev=max_nfev,
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    model = dict(schema_version=1, kind=options["kind"], parameters=result.x,
                 feature_mean=stats[0], feature_scale=stats[1],
                 encoder=encoder if options["kind"] == "pretrained" else None)
    diagnostics = dict(cost=float(result.cost), optimality=float(result.optimality),
                       evaluations=int(result.nfev), converged=bool(result.success),
                       physical_parameters=dict(zip(PARAMETERS, (PRIOR + PRIOR_SCALE * result.x[:4]).tolist())))
    return model, diagnostics


def predict(model, frame):
    features, _ = prepared_features(
        frame, model["kind"], model["encoder"], (model["feature_mean"], model["feature_scale"]))
    physics, _ = physical_power(frame, model["parameters"][:4])
    return np.logaddexp(0, (physics + features @ model["parameters"][4:]) * 20) / 20 * POWER_SCALE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).parent / "checkpoints/pretrained.joblib")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    from utsm_telemetry.hybrid_energy import field_data, score

    args.output.mkdir(parents=True, exist_ok=True)
    encoder = joblib.load(args.pretrained)["pretrained"]
    frame, sources = field_data(args.root)
    dev, holdout = frame.loc[~frame.holdout], frame.loc[frame.holdout]
    fingerprint = dict(
        sources=sources,
        encoder_sha256=hashlib.sha256(args.pretrained.read_bytes()).hexdigest(),
        prior=PRIOR.tolist(), bounds=[LOWER.tolist(), UPPER.tolist()],
        objective_version=1,
    )
    manifest_path = args.output / "training_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != fingerprint:
        raise ValueError("Dataset or encoder changed. Use a new output directory.")
    manifest_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    folds = list(GroupKFold(n_splits=4).split(dev, groups=dev.run))
    # Zero consistency weight competes with positive weights in every residual family.
    candidates = [
        dict(kind=kind, energy_weight=energy_weight, physics_weight=physics_weight, ridge=0.01)
        for kind in ("compact", "pretrained")
        for energy_weight in (0.2, 1.0, 5.0)
        for physics_weight in (0.0, 0.01, 0.1)
    ]
    candidates += [dict(kind="physics_only", energy_weight=w, physics_weight=0.0, ridge=0.01)
                   for w in (0.2, 1.0, 5.0)]
    candidates += [
        dict(kind="pretrained", energy_weight=energy_weight, physics_weight=physics_weight, ridge=ridge)
        for energy_weight in (0.0, 0.05, 0.2)
        for physics_weight in (0.0, 0.1)
        for ridge in (0.001, 0.1)
    ]
    candidates += [
        dict(kind="pretrained", energy_weight=energy_weight, physics_weight=1.0, ridge=0.01)
        for energy_weight in (0.05, 0.2)
    ]
    candidates += [
        dict(kind="pretrained", energy_weight=energy_weight, physics_weight=0.1, ridge=0.01,
             prior_weight=prior_weight)
        for energy_weight in (0.05, 0.2)
        for prior_weight in (0.1, 1.0)
    ]
    candidates += [
        dict(kind="pretrained", energy_weight=0.2, physics_weight=0.0, ridge=0.01,
             prior_weight=prior_weight)
        for prior_weight in (0.1, 1.0)
    ]
    candidates += [
        dict(kind="pretrained", energy_weight=0.2, physics_weight=physics_weight, ridge=ridge)
        for physics_weight in (0.0, 0.1)
        for ridge in (0.3, 1.0)
    ]
    candidates += [
        dict(kind="pretrained", energy_weight=0.1, physics_weight=physics_weight, ridge=ridge)
        for physics_weight in (0.0, 0.1)
        for ridge in (0.1, 0.3)
    ]
    records = []
    with threadpool_limits(limits=2):
        for index, options in enumerate(candidates):
            path = args.output / f"candidate-{index:02}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["options"] != options:
                    raise ValueError("Cached configuration differs. Use a new output directory.")
            else:
                errors, diagnostics = [], []
                for train_indices, validation_indices in folds:
                    training, validation = dev.iloc[train_indices], dev.iloc[validation_indices]
                    model, info = solve(training, encoder, options)
                    errors.extend(r["absolute_error_wh"] for r in score(validation, predict(model, validation))["runs"])
                    diagnostics.append(info)
                record = dict(options=options, development_mae_wh=float(np.mean(errors)), diagnostics=diagnostics)
                path.write_text(json.dumps(record, indent=2))
            records.append(record)
            print(index, options, "CV Wh", round(record["development_mae_wh"], 4), flush=True)

        selected = min(records, key=lambda row: row["development_mae_wh"])
        # Retrain the development-selected model for a higher optimization budget,
        # and use grouped validation to decide whether longer optimization helps.
        extended = []
        for record in sorted(records, key=lambda row: row["development_mae_wh"])[:3]:
            errors = []
            for train_indices, validation_indices in folds:
                training, validation = dev.iloc[train_indices], dev.iloc[validation_indices]
                model, _ = solve(training, encoder, record["options"], max_nfev=400)
                errors.extend(r["absolute_error_wh"] for r in score(validation, predict(model, validation))["runs"])
            refined = dict(options=record["options"], development_mae_wh=float(np.mean(errors)), max_nfev=400)
            extended.append(refined)
            print("Extended", refined, flush=True)
        selected = min([dict(**r, max_nfev=80) for r in records] + extended,
                       key=lambda row: row["development_mae_wh"])
        model, diagnostics = solve(dev, encoder, selected["options"], selected["max_nfev"])
        # The holdout is evaluated once after configuration and training-budget selection.
        metrics = score(holdout, predict(model, holdout))
        def physics_baseline(parameters):
            return dict(kind="physics_only", parameters=parameters, feature_mean=np.empty(0),
                        feature_scale=np.empty(0), encoder=None)
        default_parameters = (np.array([100., .008, .07235, .82]) - PRIOR) / PRIOR_SCALE
        baselines = {
            "signed_force_default_parameters": score(holdout, predict(physics_baseline(default_parameters), holdout)),
            "signed_force_capstone_parameters": score(holdout, predict(physics_baseline(np.zeros(4)), holdout)),
        }
        physical_selected = min((r for r in records if r["options"]["kind"] == "physics_only"),
                                key=lambda row: row["development_mae_wh"])
        physical_model, physical_diagnostics = solve(dev, encoder, physical_selected["options"], 400)
        baselines["calibrated_physics_without_residual"] = score(holdout, predict(physical_model, holdout))
        mlp_path = Path(__file__).parent / "checkpoints/telemetry-only-mlp.joblib"
        if mlp_path.exists():
            mlp = joblib.load(mlp_path)
            baselines["telemetry_only_mlp"] = score(holdout, np.maximum(mlp.predict(holdout[FEATURES]), 0))
        report = dict(baselines=baselines, physical_only_diagnostics=physical_diagnostics,
                      training_manifest=fingerprint, selected=selected, diagnostics=diagnostics, candidates=records,
                      extended=extended, metrics=metrics, sources=sources,
                      prior=dict(zip(PARAMETERS, PRIOR.tolist())),
                      bounds=dict(zip(PARAMETERS, np.column_stack([LOWER, UPPER]).tolist())),
                      caveats=["Retrospective benchmark repeatedly inspected in earlier experiments",
                               "Grade is unavailable in the shared replay dataset and is zero",
                               "Physical parameters are regularized effective estimates, not independent measurements",
                               "Current magnitude and recorded voltage define labels, without new sensor calibration"])
        model["training_options"] = selected
        joblib.dump(model, args.output / "calibrated.joblib", compress=3)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("SELECTED", selected["options"], selected["development_mae_wh"], flush=True)
        print("HOLDOUT", metrics["absolute_error_wh"], metrics["by_track"], flush=True)


if __name__ == "__main__":
    main()
