"""Reservoir-type ONN forward model, transmission matrices and ridge readout."""

from __future__ import annotations

import os
import random
from collections.abc import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> int:
    """Seed Python, NumPy and PyTorch for reproducible runs on a fixed GPU."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def random_tm(
    input_dim: int,
    output_dim: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create one reproducible complex Gaussian transmission matrix."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    real = torch.randn(input_dim, output_dim, generator=generator, device=device)
    imag = torch.randn(input_dim, output_dim, generator=generator, device=device)
    return ((real + 1j * imag) / input_dim**0.5).to(torch.complex64)


def tm_ensemble(
    input_dim: int,
    output_dim: int,
    seeds: Iterable[int],
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Create a deterministic list of transmission matrices."""

    return [random_tm(input_dim, output_dim, seed, device) for seed in seeds]


def reservoir_forward(
    x: torch.Tensor,
    tm: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return ``|exp(i * alpha * pi * x) @ tm|^2`` with global normalization.

    Masks remain continuous in ``[0, 1]``. No phase or intensity quantization is
    applied. Concatenate the coreset and query batch before calling this function
    so that both use the same normalization scale.
    """

    x = x.float()
    field = torch.exp(1j * float(alpha) * torch.pi * x)
    intensity = torch.abs(field @ tm) ** 2
    return intensity / torch.clamp(torch.amax(intensity), min=float(eps))


def fit_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge_lambda: float,
    solver: str = "primal",
) -> torch.Tensor:
    """Fit uncentered ridge without an intercept; primal is the default."""

    features = features.float()
    targets = targets.float()
    if solver == "dual":
        identity = torch.eye(features.shape[0], dtype=features.dtype, device=features.device)
        return features.T @ torch.linalg.solve_ex(
            features @ features.T + float(ridge_lambda) * identity, targets,
            check_errors=False,
        ).result
    if solver != "primal":
        raise ValueError(f"Unknown ridge solver: {solver}")
    feature_dim = features.shape[1]
    identity = torch.eye(feature_dim, dtype=features.dtype, device=features.device)
    return torch.linalg.solve(
        features.T @ features + float(ridge_lambda) * identity,
        features.T @ targets,
    )
