"""Tune the pretrained encoder end to end against interval and run energy."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

FEATURES = ["speed_kph", "accel_m_s2"]

POWER_SCALE = 200.0


def data_tensors(frame, scaler):
    x = torch.tensor(scaler.transform(frame[FEATURES]), dtype=torch.float32)
    y = torch.tensor(frame.target_w.to_numpy() / POWER_SCALE, dtype=torch.float32)
    dt = torch.tensor(frame.dt_s.to_numpy(), dtype=torch.float32)
    bins = torch.tensor(np.unique(frame.run, return_inverse=True)[1])
    integration = torch.nn.functional.one_hot(bins).T.float() * dt
    weights = (integration / integration.sum(dim=1, keepdim=True)).mean(dim=0)
    return x, y, integration * (POWER_SCALE / 3600), weights


def model_from_checkpoint(pretrained):
    mlp = pretrained.steps[-1][1]
    layers = []
    for i, (coef, bias) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        layer = nn.Linear(*coef.shape)
        scale = POWER_SCALE if i == len(mlp.coefs_) - 1 else 1
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(coef.T / scale))
            layer.bias.copy_(torch.from_numpy(bias / scale))
        layers.append(layer)
        if scale == 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def fit(pretrained, training, mode, lr, run_weight, checkpoints):
    model = model_from_checkpoint(pretrained)
    linear = [m for m in model if isinstance(m, nn.Linear)]
    for layer in linear[:-mode]:
        layer.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0)
    x, y, integration, weights = training
    anchor = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    target_energy = integration @ y

    def closure():
        optimizer.zero_grad()
        predicted = torch.nn.functional.softplus(model(x).flatten(), beta=20)
        interval = ((predicted - y).square() * weights).sum()
        run = torch.nn.functional.smooth_l1_loss(
            (integration @ predicted - target_energy) / 10, torch.zeros_like(target_energy))
        drift = sum((p - start).square().mean() for p, start in zip(
            (p for p in model.parameters() if p.requires_grad), anchor))
        loss = interval + run_weight * run + 0.01 * drift
        loss.backward()
        return loss

    for step in range(1, max(checkpoints) + 1):
        closure()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            yield step, model


@torch.no_grad()
def predict(model, x):
    return torch.nn.functional.softplus(model(x).flatten(), beta=20).numpy() * POWER_SCALE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    from utsm_telemetry.hybrid_energy import field_data, score
    torch.set_num_threads(1)
    torch.manual_seed(42)
    pretrained = joblib.load(args.pretrained)["pretrained"]
    scaler = pretrained.steps[0][1]
    frame, sources = field_data(args.root)
    dev = frame.loc[~frame.holdout]
    test = frame.loc[frame.holdout]
    folds = [(data_tensors(dev.iloc[a], scaler), dev.iloc[b], data_tensors(dev.iloc[b], scaler)[0])
             for a, b in GroupKFold(n_splits=4).split(dev, groups=dev.run)]
    results = []
    checkpoints = (15, 45, 100)
    for mode, lr, run_weight in itertools.product(
            (1, 2, 3), (0.003,), (0.0, 0.2, 1.0)):
        options = dict(mode=mode, lr=lr, run_weight=run_weight)
        errors = {epoch: [] for epoch in checkpoints}
        for training, validation, val_x in folds:
            for epoch, model in fit(pretrained, training, **options, checkpoints=checkpoints):
                errors[epoch].extend(r["absolute_error_wh"] for r in score(
                    validation, predict(model, val_x))["runs"])
        for epoch in checkpoints:
            results.append(dict(**options, optimizer_steps=epoch,
                                development_mae_wh=float(np.mean(errors[epoch]))))
        print(min(results[-len(checkpoints):], key=lambda row: row["development_mae_wh"]), flush=True)
    selected = min(results, key=lambda result: result["development_mae_wh"])
    options = {k: v for k, v in selected.items() if k != "development_mae_wh"}
    epochs = options.pop("optimizer_steps")
    _, model = next(fit(pretrained, data_tensors(dev, scaler), **options, checkpoints=(epochs,)))
    metrics = score(test, predict(model, data_tensors(test, scaler)[0]))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), widths=[m.out_features for m in model
                    if isinstance(m, nn.Linear)], features=FEATURES, mean=scaler.mean_.tolist(),
                    scale=scaler.scale_.tolist(), power_scale=POWER_SCALE, softplus_beta=20),
               args.output / "model.pt")
    report = dict(selected=selected, validation=results, metrics=metrics, sources=sources)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Selected", selected, "holdout MAE Wh", metrics["absolute_error_wh"])


if __name__ == "__main__":
    main()
