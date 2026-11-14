"""Fit the telemetry-only baseline from the pinned telemetry checkout."""
import argparse
import sys
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    from utsm_telemetry.hybrid_energy import FEATURES, field_data, network, score, weights

    with threadpool_limits(limits=2):
        frame, _ = field_data(args.root)
        dev, test = frame.loc[~frame.holdout], frame.loc[frame.holdout]
        model = network(64).fit(dev[FEATURES], dev.target_w,
                                mlpregressor__sample_weight=weights(dev))
        print(score(test, model.predict(test[FEATURES]).clip(min=0))["absolute_error_wh"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)


if __name__ == "__main__":
    main()
