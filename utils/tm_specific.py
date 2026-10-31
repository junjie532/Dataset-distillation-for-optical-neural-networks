"""Distill a synthetic dataset through a measured complex transmission matrix."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from .datasets import class_balanced_indices
from .reservoir_onn import set_seed


def load_measured_tm(path, device):
    """Load the calibrated T-only matrix shipped with the physical experiment."""
    record = loadmat(path, squeeze_me=True)
    if not bool(record["strict_no_b"]):
        raise ValueError("A T-only measured model is required")
    matrix = np.asarray(record["T"], dtype=np.complex64)
    if matrix.shape != (784, 1024) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 784 x 1024 measured TM")
    return torch.as_tensor(matrix, device=device)


def distill_measured_tm(
    data,
    tm,
    seed,
    steps=1500,
    batch_size=128,
    ridge_lambda=0.01,
    batch_seed=7,
    learning_rate=0.01,
    snapshot_interval=150,
):
    """Reproduce the measured-TM optimizer, including its precomputed batches.

    Seeds 0..4 share batch seed 7. Inputs, targets and a log-temperature are
    optimized with Adam; inputs are clamped to [0, 1] after each update.
    Test scores are recorded for display, never used for model selection.
    """
    set_seed(seed)
    x_train, y_train = data["x_train"], data["y_train"]
    device = x_train.device
    indices = class_balanced_indices(y_train, 1, 10, seed)
    x_initial = x_train[indices].clone()
    x = x_initial.clone().requires_grad_(True)
    y = F.one_hot(y_train[indices], 10).float().requires_grad_(True)
    log_temperature = torch.tensor(0.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [
            {"params": [x], "lr": learning_rate},
            {"params": [y], "lr": learning_rate},
            {"params": [log_temperature], "lr": learning_rate},
        ]
    )
    generator = torch.Generator(device=device).manual_seed(batch_seed)
    query_indices = torch.randint(
        len(x_train), (steps, batch_size), generator=generator, device=device
    )
    losses, snapshots, accuracy, images, labels, temperatures = [], [], [], [], [], []

    def snapshot(step):
        score = evaluate_measured_tm(
            x.detach(),
            y.detach(),
            data["x_test"],
            data["y_test"],
            tm,
            ridge_lambda,
            1.0,
        )
        snapshots.append(step)
        accuracy.append(score)
        images.append(x.detach().cpu().numpy().copy())
        labels.append(y.detach().cpu().numpy().copy())
        temperatures.append(float(log_temperature.detach().exp()))

    snapshot(0)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        query = query_indices[step - 1]
        features = measured_tm_forward(torch.cat([x, x_train[query]]), tm, 1.0)
        logits = ridge_logits(features[:10], y, features[10:], ridge_lambda)
        loss = F.cross_entropy(logits / log_temperature.exp(), y_train[query])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
        losses.append(float(loss.detach()))
        if step % snapshot_interval == 0 or step == steps:
            snapshot(step)
    return {
        "x_coreset": x.detach(),
        "y_coreset": y.detach(),
        "source_indices": indices,
        "x_initial": x_initial,
        "loss_history": np.asarray(losses),
        "snapshot_steps": np.asarray(snapshots),
        "accuracy_history": np.asarray(accuracy),
        "x_trajectory": np.asarray(images),
        "y_trajectory": np.asarray(labels),
        "temperature_history": np.asarray(temperatures),
        "query_indices": query_indices,
        "seed": seed,
        "batch_seed": batch_seed,
        "ridge_lambda": ridge_lambda,
    }


def ridge_logits(
    support_features: torch.Tensor,
    support_targets: torch.Tensor,
    query_features: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    support_features = support_features.float()
    support_targets = support_targets.float()
    query_features = query_features.float()
    if len(support_features) <= support_features.shape[1]:
        identity = torch.eye(
            len(support_features),
            dtype=support_features.dtype,
            device=support_features.device,
        )
        coefficients = torch.linalg.solve(
            support_features @ support_features.T + float(ridge_lambda) * identity,
            support_targets,
        )
        return query_features @ support_features.T @ coefficients
    identity = torch.eye(
        support_features.shape[1],
        dtype=support_features.dtype,
        device=support_features.device,
    )
    weight = torch.linalg.solve(
        support_features.T @ support_features + float(ridge_lambda) * identity,
        support_features.T @ support_targets,
    )
    return query_features @ weight


def measured_tm_forward(
    x: torch.Tensor, tm: torch.Tensor, alpha: float, eps: float = 1e-12
) -> torch.Tensor:
    """The sole optical forward: intensity from T, with no additive field."""
    field = torch.exp(1j * float(alpha) * torch.pi * x.float()) @ tm
    intensity = torch.abs(field) ** 2
    return intensity / torch.clamp(torch.amax(intensity), min=float(eps))


@torch.no_grad()
def evaluate_measured_tm(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    tm: torch.Tensor,
    ridge_lambda: float,
    alpha: float,
) -> float:
    support_size = len(x_support)
    features = measured_tm_forward(torch.cat([x_support, x_query]), tm, alpha)
    logits = ridge_logits(
        features[:support_size], y_support, features[support_size:], ridge_lambda
    )
    return float((logits.argmax(1) == y_query).float().mean().item())
