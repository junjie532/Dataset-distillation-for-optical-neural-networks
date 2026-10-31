"""Classification evaluation over transmission-matrix ensembles."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
from .reservoir_onn import fit_ridge, reservoir_forward

@torch.no_grad()
def evaluate_coreset_on_tm(
    x_coreset: torch.Tensor,
    y_coreset: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    tm: torch.Tensor,
    ridge_lambda: float,
    alpha: float = 1.0,
) -> float:
    coreset_size = x_coreset.shape[0]
    features = reservoir_forward(torch.cat([x_coreset, x_query]), tm, alpha)
    weight = fit_ridge(features[:coreset_size], y_coreset, ridge_lambda)
    prediction = torch.argmax(features[coreset_size:] @ weight, dim=1)
    return float((prediction == y_query).float().mean().item())


def evaluate_coreset(
    x_coreset: torch.Tensor,
    y_coreset: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    tms: Sequence[torch.Tensor],
    ridge_lambda: float,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    accuracy = np.asarray(
        [
            evaluate_coreset_on_tm(
                x_coreset, y_coreset, x_query, y_query, tm, ridge_lambda, alpha
            )
            for tm in tms
        ],
        dtype=np.float64,
    )
    return {
        "accuracy_per_tm": accuracy,
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
    }


