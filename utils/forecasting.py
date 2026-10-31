"""NARMA10 generation and 200-step feedback prediction with fixed reservoir readouts."""
from __future__ import annotations
import numpy as np
import torch
from .regression import _scale_arrays, add_fixed_linear_projection, project_latent_inputs


def generate_narma10(length, seed):
    rng = np.random.default_rng(int(seed))
    drive = rng.uniform(0.0, 0.5, int(length)).astype(np.float64)
    state = np.zeros(int(length), dtype=np.float64)
    for t in range(9, int(length) - 1):
        state[t + 1] = (
            0.3 * state[t]
            + 0.05 * state[t] * state[t - 9 : t + 1].sum()
            + 1.5 * drive[t - 9] * drive[t]
            + 0.1
        )
    return drive, state


def make_narma10_samples(drive, state, origins):
    inputs = np.stack(
        [
            np.concatenate((state[t - 9 : t + 1], drive[t - 9 : t + 1]))
            for t in origins
        ]
    )
    targets = state[origins + 1, None]
    return inputs.astype(np.float64), targets.astype(np.float64)


def preprocess_narma10(config, device):
    train_start = int(config["washout"])
    validation_start = train_start + int(config["train_origins"])
    test_start = validation_start + int(config["validation_origins"])
    length = test_start + int(config["test_origins"]) + 1
    drive, state = generate_narma10(length, int(config["base_seed"]) + 17)
    train_indices = np.arange(train_start, validation_start, dtype=np.int64)
    validation_indices = np.arange(validation_start, test_start, dtype=np.int64)
    test_indices = np.arange(
        test_start, test_start + int(config["test_origins"]), dtype=np.int64
    )
    x_train_raw, y_train_raw = make_narma10_samples(drive, state, train_indices)
    x_validation_raw, y_validation_raw = make_narma10_samples(
        drive, state, validation_indices
    )
    x_test_raw, y_test_raw = make_narma10_samples(drive, state, test_indices)
    data = _scale_arrays(
        x_train_raw,
        x_validation_raw,
        x_test_raw,
        y_train_raw,
        y_validation_raw,
        y_test_raw,
        device=device,
        validation_limit=int(config["validation_limit"]),
    )
    data = add_fixed_linear_projection(
        data,
        projection_dim=int(config["optical_dim"]),
        projection_seed=int(config["projection_seed"]),
    )
    raw = dict(
        x_train=x_train_raw,
        x_validation=x_validation_raw,
        x_test=x_test_raw,
        y_train=y_train_raw,
        y_validation=y_validation_raw,
        y_test=y_test_raw,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )
    return data, raw, drive, state, test_start


def quantize_mask(mask):
    return torch.round(mask.float().clamp(0.0, 1.0) * 255.0) / 255.0


class ReservoirReadoutEnsemble:
    """One independent ridge readout per fixed random TM."""

    def __init__(self, x_support, y_support, ridge_lambda, tms):
        weights = []
        scales = []
        targets = y_support.float().reshape(len(y_support), -1)
        for tm in tms:
            support_raw = torch.abs(
                torch.exp(1j * torch.pi * x_support.float()) @ tm
            ) ** 2
            scale = support_raw.amax().clamp_min(1e-12)
            features = (support_raw / scale).float()
            if len(features) <= features.shape[1]:
                coefficient = torch.linalg.solve(
                    features @ features.T
                    + float(ridge_lambda) * torch.eye(len(features), device=features.device),
                    targets,
                )
                weight = features.T @ coefficient
            else:
                weight = torch.linalg.solve(
                    features.T @ features
                    + float(ridge_lambda)
                    * torch.eye(features.shape[1], device=features.device),
                    features.T @ targets,
                )
            weights.append(weight)
            scales.append(scale)
        self.tms = torch.stack(list(tms), dim=0)
        self.weights = torch.stack(weights, dim=0)
        self.scales = torch.stack(scales).reshape(-1, 1, 1)

    @torch.no_grad()
    def predict(self, x_query):
        """Return (TM, query, target), without averaging predictions across TMs.

        Queries can be shared (query, input) or TM-specific (TM, query, input).
        Autonomous feedback uses the latter so histories remain independent.
        """
        field = torch.exp(1j * torch.pi * x_query.float())
        if field.ndim == 2:
            output_field = torch.einsum("bi,tio->tbo", field, self.tms)
        elif field.ndim == 3 and field.shape[0] == len(self.tms):
            output_field = torch.einsum("tbi,tio->tbo", field, self.tms)
        else:
            raise ValueError("Queries must have shape (B, D) or (number of TMs, B, D).")
        features = (torch.abs(output_field) ** 2 / self.scales).float()
        return torch.einsum("tbo,tok->tbk", features, self.weights)


def r2_per_tm(predictions, target):
    """Score each trajectory separately; callers aggregate the resulting scores."""
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if predictions.ndim != 2 or predictions.shape[1] != len(target):
        raise ValueError("Expected (TM, time) predictions and a matching target trajectory.")
    denominator = max(float(np.sum((target - target.mean()) ** 2)), 1e-15)
    return 1.0 - np.sum((predictions - target[None]) ** 2, axis=1) / denominator


@torch.no_grad()
def autonomous_rollout(model, drive, state, start, steps, data, *, quantize8=False):
    """Return independent (TM, time) trajectories with one feedback history per TM.

    All devices start with the same ten observed states and known driving input.
    Thereafter each device receives only its own predictions as state feedback.
    Input scaling, target scaling and optional mask quantization are unchanged.
    """
    count = len(model.tms)
    history = np.broadcast_to(np.asarray(state[start - 9:start + 1], dtype=np.float64),
                              (count, 10)).copy()
    lower = np.asarray(data["x_lower"], dtype=np.float64)
    scale = np.asarray(data["x_scale"], dtype=np.float64)
    y_mean = float(np.asarray(data["y_mean"]).reshape(-1)[0])
    y_std = float(np.asarray(data["y_std"]).reshape(-1)[0])
    predictions = np.empty((count, int(steps)), dtype=np.float64)
    for offset in range(int(steps)):
        t = int(start + offset)
        external_input = np.broadcast_to(drive[t - 9:t + 1], (count, 10))
        raw_input = np.concatenate((history, external_input), axis=1)
        native = np.clip((raw_input - lower) / scale, 0.0, 1.0).astype(np.float32)
        latent = torch.from_numpy(native[:, None, :]).to(data["x_train"].device)
        mask = project_latent_inputs(latent, data)
        if quantize8:
            mask = quantize_mask(mask)
        prediction_z = model.predict(mask)[:, 0, 0].cpu().numpy().astype(np.float64)
        prediction = prediction_z * y_std + y_mean
        predictions[:, offset] = prediction
        history = np.concatenate((history[:, 1:], prediction[:, None]), axis=1)
    return predictions
