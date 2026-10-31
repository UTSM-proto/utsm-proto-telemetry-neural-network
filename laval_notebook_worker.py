"""Execute all scientific operations outside the interactive notebook kernel."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_training(path):
    spec = importlib.util.spec_from_file_location("utsm_laval_worker_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["preview", "train", "check-model"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    args = parser.parse_args()
    training = load_training(Path(args.source).resolve())
    cfg = training.Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    cfg.validate()
    out = Path(args.output).resolve()
    if args.operation == "train":
        training.run_workflow(args.archive, out, cfg, args.reference)
        return
    if args.operation == "preview":
        canonical, quality, provenance = training.load_archive(args.archive, cfg)
        dev, test, split = training.split_sessions(canonical, cfg)
        out.mkdir(parents=True, exist_ok=True)
        quality.to_csv(out/"data_quality.csv", index=False)
        summary = {
            "source_manifest": provenance, "split_manifest": split,
            "coverage": [
                {"split": "development", "sessions": int(dev.session_id.nunique()),
                 "rows": len(dev), "valid_minutes": float(dev.dt_s.sum()/60)},
                {"split": "held_out", "sessions": int(test.session_id.nunique()),
                 "rows": len(test), "valid_minutes": float(test.dt_s.sum()/60)},
            ],
            "constant_voltage": bool(quality.get("constant_voltage",
                                     training.pd.Series(dtype=bool)).fillna(False).any()),
        }
        (out/"preview.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Preview ready:", out)
        return
    # Read only saved features/outputs. Loading/predicting never imports NumPy in the parent.
    import joblib
    import numpy as np
    import pandas as pd
    model = joblib.load(out/"random_forest_development.joblib")
    canonical = pd.read_csv(out/"canonical_intervals.csv.gz")
    split = json.loads((out/"split_manifest.json").read_text(encoding="utf-8"))
    held = canonical[canonical.session_id.isin(split["test_sessions"])]
    examples = held[model["features"]].head(5)
    watts = training.predict_power(model, examples)
    saved = pd.read_csv(out/"held_out_predictions.csv.gz")
    recorded = saved[saved.model == "random_forest"].predicted_power_w.head(5).to_numpy()
    np.testing.assert_allclose(watts, recorded)
    result = {"passed": True, "predictions": json.loads(
        examples.assign(predicted_power_w=watts).to_json(orient="records"))}
    (out/"inference_example.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved-model prediction round trip passed.")


if __name__ == "__main__":
    main()
