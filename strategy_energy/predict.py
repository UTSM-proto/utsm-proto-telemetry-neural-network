"""Score interval features with a checked-in energy checkpoint."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
FEATURES = ["speed_kph", "accel_m_s2"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", choices=["mlp", "finetuned"], default="mlp")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if (frame.empty or not set([*FEATURES, "dt_s"]).issubset(frame)
            or not np.isfinite(frame[[*FEATURES, "dt_s"]].to_numpy(dtype=float)).all()
            or (frame.dt_s <= 0).any()):
        parser.error("Supply finite speed_kph, accel_m_s2 and positive dt_s for each interval.")
    x = frame[FEATURES]
    if args.model == "mlp":
        model = joblib.load(ROOT / "checkpoints/telemetry-only-mlp.joblib")
        watts = np.maximum(model.predict(x), 0)
    else:
        artifact = torch.load(ROOT / "checkpoints/field-finetuned.pt", map_location="cpu", weights_only=True)
        layers = []
        width = len(artifact["features"])
        for i, output_width in enumerate(artifact["widths"]):
            layers.append(nn.Linear(width, output_width))
            if i + 1 < len(artifact["widths"]):
                layers.append(nn.ReLU())
            width = output_width
        model = nn.Sequential(*layers)
        model.load_state_dict(artifact["state"])
        normalized = (x[artifact["features"]].to_numpy() - artifact["mean"]) / artifact["scale"]
        with torch.inference_mode():
            watts = nn.functional.softplus(
                model(torch.tensor(normalized, dtype=torch.float32)).flatten(),
                beta=artifact["softplus_beta"]).numpy() * artifact["power_scale"]
    print(f"{np.dot(watts, frame.dt_s) / 3600:.4f} Wh")
    if args.output:
        frame.assign(predicted_power_w=watts, predicted_energy_wh=watts * frame.dt_s / 3600).to_csv(
            args.output, index=False)


if __name__ == "__main__":
    main()
