"""Continuous-target distillation and support-calibrated reservoir-type ONN readouts."""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import torch
from .reservoir_onn import fit_ridge, reservoir_forward

SHARED_SUPPORT_QUERY = 'shared_support_query'
SUPPORT_ONLY_TRAIN_CALIBRATION = 'support_only_train_calibration'
OPTICAL_INTENSITY_NORMALIZATION_POLICIES = frozenset({SHARED_SUPPORT_QUERY, SUPPORT_ONLY_TRAIN_CALIBRATION})
RAW_FEATURE_NAMES = ('AT', 'V', 'AP', 'RH')
TARGET_NAME = 'PE'


def project_latent_inputs(latent_x, data):
    """Map native sigmoid coordinates through the fixed training-fitted projection."""
    if 'projection_matrix' not in data:
        return latent_x.float()
    projected = (2.0 * latent_x.float() - 1.0) @ data['projection_matrix'].T
    return ((projected - data['projection_lower']) / data['projection_scale']).clamp(0.0, 1.0)


def validate_optical_intensity_normalization_policy(policy: str) -> str:
    """Return a supported optical-intensity policy or fail loudly.

    Policy names are deliberately exact protocol identifiers: silently
    lower-casing, aliasing, or accepting ``None`` could cause a resumed run to
    switch normalization without changing its apparent configuration.
    """

    if not isinstance(policy, str) or policy not in OPTICAL_INTENSITY_NORMALIZATION_POLICIES:
        expected = ", ".join(sorted(OPTICAL_INTENSITY_NORMALIZATION_POLICIES))
        raise ValueError(
            f"unsupported optical intensity normalization policy {policy!r}; "
            f"expected exactly one of: {expected}"
        )
    return policy


@dataclass
class RegressionDistillationResult:
    x_init: torch.Tensor
    x_distilled: torch.Tensor
    y_init: torch.Tensor
    y_distilled: torch.Tensor
    source_indices: torch.Tensor
    loss_history: np.ndarray
    validation_steps: np.ndarray
    validation_r2: np.ndarray
    best_step: int
    best_validation_r2: float
    latent_x_init: torch.Tensor
    latent_x_distilled: torch.Tensor


@dataclass
class QueryFeatureCache:
    """Unnormalised query intensities, one tensor per TM.

    Caching avoids recomputing a large fixed query set for every support.
    ``raw_maxima`` is retained solely for the legacy shared-support-query
    protocol.  The support-only protocol consumes ``raw_features`` but never
    reads ``raw_maxima``.
    """

    raw_features: list[torch.Tensor]
    raw_maxima: list[torch.Tensor]
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION


def _raw_intensity(x: torch.Tensor, tm: torch.Tensor) -> torch.Tensor:
    field = torch.exp(1j * torch.pi * x.float())
    return (torch.abs(field @ tm) ** 2).float()


@torch.no_grad()
def query_feature_cache(
    x_query: torch.Tensor,
    tms: Sequence[torch.Tensor],
    *,
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION,
) -> QueryFeatureCache:
    policy = validate_optical_intensity_normalization_policy(
        optical_intensity_normalization_policy
    )
    raw = [_raw_intensity(x_query, tm) for tm in tms]
    return QueryFeatureCache(
        raw,
        [value.amax() for value in raw],
        optical_intensity_normalization_policy=policy,
    )


def dual_ridge_predict(
    support_features: torch.Tensor,
    support_targets: torch.Tensor,
    query_features: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Exact dual of the uncentered Exp1--Exp4 ridge readout."""

    support_features = support_features.float()
    query_features = query_features.float()
    support_targets = support_targets.float()
    if support_targets.ndim == 1:
        support_targets = support_targets[:, None]
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


def ridge_predict(
    support_features: torch.Tensor,
    support_targets: torch.Tensor,
    query_features: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Use the cheaper exact primal or dual form without changing the model."""

    targets = support_targets.float()
    if targets.ndim == 1:
        targets = targets[:, None]
    if len(support_features) <= support_features.shape[1]:
        return dual_ridge_predict(support_features, targets, query_features, ridge_lambda)
    weight = fit_ridge(support_features, targets, ridge_lambda)
    return query_features.float() @ weight


def optical_ridge_predict(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    tm: torch.Tensor,
    ridge_lambda: float,
    *,
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION,
) -> torch.Tensor:
    """Differentiable prediction under an explicit intensity policy.

    For ``support_only_train_calibration``, let ``I_S`` and ``I_Q`` be the raw
    support and query intensities for one TM and ``m_S=max(I_S)``.  Both feature
    matrices are divided by the same ``max(m_S, 1e-12)``.  Consequently a
    query prediction is invariant to adding, removing, or reordering other
    query samples.  The legacy branch intentionally retains the exact
    concatenate-then-:func:`reservoir_forward` implementation.
    """

    policy = validate_optical_intensity_normalization_policy(
        optical_intensity_normalization_policy
    )
    support_size = len(x_support)
    if policy == SHARED_SUPPORT_QUERY:
        features = reservoir_forward(torch.cat([x_support, x_query]), tm)
        support_features = features[:support_size]
        query_features = features[support_size:]
    else:
        support_raw = _raw_intensity(x_support, tm)
        scale = support_raw.amax().clamp_min(1e-12)
        support_features = support_raw / scale
        query_features = _raw_intensity(x_query, tm) / scale
    return ridge_predict(
        support_features, y_support, query_features, ridge_lambda
    )


@torch.no_grad()
def cached_optical_ridge_predict(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    tm: torch.Tensor,
    query_raw: torch.Tensor,
    query_raw_max: torch.Tensor | None,
    ridge_lambda: float,
    *,
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION,
) -> torch.Tensor:
    """Prediction equivalent to :func:`optical_ridge_predict` using a cache."""

    policy = validate_optical_intensity_normalization_policy(
        optical_intensity_normalization_policy
    )
    support_raw = _raw_intensity(x_support, tm)
    if policy == SUPPORT_ONLY_TRAIN_CALIBRATION:
        # Do not inspect query_raw_max here: strict inductive prediction is
        # calibrated exclusively by the support intensities.
        scale = support_raw.amax().clamp_min(1e-12)
    else:
        if query_raw_max is None:
            raise ValueError(
                "shared_support_query cache prediction requires query_raw_max"
            )
        scale = torch.maximum(support_raw.amax(), query_raw_max).clamp_min(1e-12)
    return ridge_predict(support_raw / scale, y_support, query_raw / scale, ridge_lambda)


def r2_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().reshape(len(prediction), -1)
    target = target.float().reshape(len(target), -1)
    residual = torch.sum((prediction - target) ** 2)
    total = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2)
    return float((1.0 - residual / total.clamp_min(1e-12)).item())


def regression_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.float().reshape(len(prediction), -1)
    target = target.float().reshape(len(target), -1)
    error = prediction - target
    return {
        "r2": r2_value(prediction, target),
        "mse": float(torch.mean(error**2).item()),
        "rmse": float(torch.sqrt(torch.mean(error**2)).item()),
        "mae": float(torch.mean(torch.abs(error)).item()),
    }


@torch.no_grad()
def evaluate_regression(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    tms: Sequence[torch.Tensor],
    ridge_lambda: float,
    *,
    query_cache: QueryFeatureCache | None = None,
    return_predictions: bool = False,
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION,
) -> dict[str, Any]:
    """Evaluate R2/MSE/MAE over an ensemble of unseen TMs."""

    policy = validate_optical_intensity_normalization_policy(
        optical_intensity_normalization_policy
    )
    if query_cache is not None and len(query_cache.raw_features) != len(tms):
        raise ValueError("query cache and TM ensemble have different lengths")
    if query_cache is not None:
        cache_policy = validate_optical_intensity_normalization_policy(
            query_cache.optical_intensity_normalization_policy
        )
        if cache_policy != policy:
            raise ValueError(
                "query cache optical normalization policy mismatch: "
                f"cache={cache_policy!r}, evaluation={policy!r}"
            )
        if policy == SHARED_SUPPORT_QUERY and len(query_cache.raw_maxima) != len(tms):
            raise ValueError("shared-support-query cache is missing raw maxima")
    predictions: list[torch.Tensor] = []
    for index, tm in enumerate(tms):
        if query_cache is None:
            prediction = optical_ridge_predict(
                x_support,
                y_support,
                x_query,
                tm,
                ridge_lambda,
                optical_intensity_normalization_policy=policy,
            )
        else:
            # Index raw_maxima only for the legacy policy.  A support-only
            # cache may deliberately omit it, which guards against accidental
            # query-dependent calibration.
            query_raw_max = (
                query_cache.raw_maxima[index]
                if policy == SHARED_SUPPORT_QUERY
                else None
            )
            prediction = cached_optical_ridge_predict(
                x_support,
                y_support,
                tm,
                query_cache.raw_features[index],
                query_raw_max,
                ridge_lambda,
                optical_intensity_normalization_policy=policy,
            )
        predictions.append(prediction)
    values = [regression_metrics(prediction, y_query) for prediction in predictions]
    result: dict[str, Any] = {}
    for metric in ("r2", "mse", "rmse", "mae"):
        array = np.asarray([item[metric] for item in values], dtype=np.float64)
        result[f"{metric}_per_tm"] = array
        result[f"{metric}_mean"] = float(array.mean())
        result[f"{metric}_std"] = (
            float(array.std(ddof=1)) if len(array) > 1 else 0.0
        )
    if return_predictions:
        result["prediction_per_tm"] = torch.stack(predictions).cpu().numpy()
        result["prediction_ensemble"] = (
            torch.stack(predictions).mean(dim=0).cpu().numpy()
        )
    return result


def distill_regression(
    data: dict[str, Any],
    *,
    support_size: int,
    ridge_lambda: float,
    train_tms: Sequence[torch.Tensor],
    validation_tms: Sequence[torch.Tensor],
    steps: int,
    batch_size: int,
    tms_per_step: int,
    lr_x: float,
    lr_y: float,
    gradient_clip: float,
    seed: int,
    eval_every: int = 50,
    validation_cache: QueryFeatureCache | None = None,
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION,
) -> RegressionDistillationResult:
    """Distill continuous X+Y and retain the validation-selected checkpoint."""

    x_train = data["x_train"]
    y_train = data["y_train"]
    policy = validate_optical_intensity_normalization_policy(
        optical_intensity_normalization_policy
    )
    generator = torch.Generator(device=x_train.device).manual_seed(int(seed))
    source_indices = torch.randperm(
        len(x_train), generator=generator, device=x_train.device
    )[: int(support_size)]
    x_init = x_train[source_indices].detach().clone()
    y_init = y_train[source_indices].detach().clone()
    # For projected tabular data, optimize the original low-dimensional
    # variables and pass them through the same fixed A used by every real
    # sample.  This keeps distilled samples inside the physically valid
    # projected-input manifold instead of inventing arbitrary D-pixel masks.
    latent_train = data.get("x_train_latent", x_train)
    latent_x_init = latent_train[source_indices].detach().clone()
    clipped = latent_x_init.clamp(1e-4, 1.0 - 1e-4)
    x_parameter = torch.log(clipped / (1.0 - clipped)).requires_grad_(True)
    y_parameter = y_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [x_parameter], "lr": float(lr_x)},
            {"params": [y_parameter], "lr": float(lr_y)},
        ]
    )

    losses: list[float] = []
    validation_steps: list[int] = []
    validation_r2: list[float] = []
    best_score = -math.inf
    best_step = 0
    best_x: torch.Tensor | None = None
    best_y: torch.Tensor | None = None
    best_latent: torch.Tensor | None = None

    def current() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_x = torch.sigmoid(x_parameter)
        x_onn = project_latent_inputs(latent_x, data)
        return x_onn, y_parameter, latent_x

    def select(step: int) -> None:
        nonlocal best_score, best_step, best_x, best_y, best_latent
        x_current, y_current, latent_current = current()
        evaluation = evaluate_regression(
            x_current.detach(),
            y_current.detach(),
            data["x_val_select"],
            data["y_val_select"],
            validation_tms,
            ridge_lambda,
            query_cache=validation_cache,
            optical_intensity_normalization_policy=policy,
        )
        score = float(evaluation["r2_mean"])
        validation_steps.append(int(step))
        validation_r2.append(score)
        if score > best_score:
            best_score = score
            best_step = int(step)
            best_x = x_current.detach().clone()
            best_y = y_current.detach().clone()
            best_latent = latent_current.detach().clone()

    select(0)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        query_indices = torch.randint(
            0,
            len(x_train),
            (min(int(batch_size), len(x_train)),),
            generator=generator,
            device=x_train.device,
        )
        x_query = x_train[query_indices]
        y_query = y_train[query_indices]
        x_current, y_current, _ = current()
        tm_indices = torch.randint(
            0,
            len(train_tms),
            (int(tms_per_step),),
            generator=generator,
            device=x_train.device,
        ).tolist()
        loss = torch.zeros((), device=x_train.device)
        for tm_index in tm_indices:
            prediction = optical_ridge_predict(
                x_current,
                y_current,
                x_query,
                train_tms[tm_index],
                ridge_lambda,
                optical_intensity_normalization_policy=policy,
            )
            loss = loss + torch.mean(
                (prediction.reshape_as(y_query) - y_query.float()) ** 2
            )
        loss = loss / len(tm_indices)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite regression distillation loss at step {step}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [x_parameter, y_parameter], float(gradient_clip)
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if step % int(eval_every) == 0 or step == int(steps):
            select(step)

    if best_x is None or best_y is None or best_latent is None:
        raise RuntimeError("regression validation checkpoint selection failed")
    return RegressionDistillationResult(
        x_init=x_init,
        x_distilled=best_x,
        y_init=y_init,
        y_distilled=best_y,
        source_indices=source_indices,
        loss_history=np.asarray(losses, dtype=np.float64),
        validation_steps=np.asarray(validation_steps, dtype=np.int64),
        validation_r2=np.asarray(validation_r2, dtype=np.float64),
        best_step=best_step,
        best_validation_r2=best_score,
        latent_x_init=latent_x_init,
        latent_x_distilled=best_latent,
    )


def random_support_indices(length: int, size: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(length, generator=generator, device=device)[: int(size)]


@torch.no_grad()
def search_random_lambda(
    data: dict[str, Any],
    support_sizes: Sequence[int],
    lambda_grid: Sequence[float],
    validation_tms: Sequence[torch.Tensor],
    *,
    draws: int,
    seed: int,
    validation_cache: QueryFeatureCache | None = None,
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION,
) -> dict[str, Any]:
    policy = validate_optical_intensity_normalization_policy(
        optical_intensity_normalization_policy
    )
    scores = np.zeros((len(support_sizes), len(lambda_grid)), dtype=np.float64)
    device = str(data["x_train"].device)
    for size_index, size in enumerate(support_sizes):
        for draw in range(int(draws)):
            indices = random_support_indices(
                len(data["x_train"]), size, seed + 100 * int(size) + draw, device
            )
            for lambda_index, ridge_lambda in enumerate(lambda_grid):
                result = evaluate_regression(
                    data["x_train"][indices],
                    data["y_train"][indices],
                    data["x_val_select"],
                    data["y_val_select"],
                    validation_tms,
                    float(ridge_lambda),
                    query_cache=validation_cache,
                    optical_intensity_normalization_policy=policy,
                )
                scores[size_index, lambda_index] += float(result["r2_mean"])
        scores[size_index] /= int(draws)
    best_indices = scores.argmax(axis=1)
    return {
        "scores": scores,
        "best_lambda": np.asarray(
            [lambda_grid[index] for index in best_indices], dtype=np.float64
        ),
        "lambda_by_size": {
            int(size): float(lambda_grid[index])
            for size, index in zip(support_sizes, best_indices)
        },
    }


@torch.no_grad()
def search_full_lambda(
    data: dict[str, Any],
    lambda_grid: Sequence[float],
    validation_tms: Sequence[torch.Tensor],
    *,
    validation_cache: QueryFeatureCache | None = None,
    optical_intensity_normalization_policy: str = SUPPORT_ONLY_TRAIN_CALIBRATION,
) -> dict[str, Any]:
    policy = validate_optical_intensity_normalization_policy(
        optical_intensity_normalization_policy
    )
    if validation_cache is not None:
        cache_policy = validate_optical_intensity_normalization_policy(
            validation_cache.optical_intensity_normalization_policy
        )
        if cache_policy != policy:
            raise ValueError(
                "query cache optical normalization policy mismatch: "
                f"cache={cache_policy!r}, evaluation={policy!r}"
            )
    scores = np.zeros(len(lambda_grid), dtype=np.float64)
    for tm_index, tm in enumerate(validation_tms):
        support_raw = _raw_intensity(data["x_train"], tm)
        if validation_cache is None:
            query_raw = _raw_intensity(data["x_val_select"], tm)
            query_max = query_raw.amax()
        else:
            query_raw = validation_cache.raw_features[tm_index]
            query_max = (
                validation_cache.raw_maxima[tm_index]
                if policy == SHARED_SUPPORT_QUERY
                else None
            )
        if policy == SUPPORT_ONLY_TRAIN_CALIBRATION:
            scale = support_raw.amax().clamp_min(1e-12)
        else:
            if query_max is None:
                raise ValueError(
                    "shared_support_query full-ridge_lambda search requires query maxima"
                )
            scale = torch.maximum(support_raw.amax(), query_max).clamp_min(1e-12)
        support_features = support_raw / scale
        query_features = query_raw / scale
        for lambda_index, ridge_lambda in enumerate(lambda_grid):
            prediction = ridge_predict(
                support_features, data["y_train"], query_features, float(ridge_lambda)
            )
            scores[lambda_index] += r2_value(
                prediction, data["y_val_select"]
            )
    scores /= len(validation_tms)
    best_index = int(scores.argmax())
    return {
        "scores": scores,
        "best_lambda": float(lambda_grid[best_index]),
    }


def _select_validation_subset(
    x_val: torch.Tensor, y_val: torch.Tensor, limit: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if len(x_val) <= int(limit):
        indices = np.arange(len(x_val), dtype=np.int64)
    else:
        indices = np.linspace(0, len(x_val) - 1, int(limit)).round().astype(np.int64)
    index_tensor = torch.as_tensor(indices, device=x_val.device)
    return x_val[index_tensor], y_val[index_tensor], indices


def _scale_arrays(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    *,
    device: str,
    validation_limit: int,
) -> dict[str, Any]:
    lower = np.quantile(x_train, 0.005, axis=0)
    upper = np.quantile(x_train, 0.995, axis=0)
    scale = np.maximum(upper - lower, 1e-6)

    def transform_x(value: np.ndarray) -> torch.Tensor:
        scaled = np.clip((value - lower) / scale, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(scaled).to(device)

    y_train_2d = np.asarray(y_train, dtype=np.float64).reshape(len(y_train), -1)
    y_val_2d = np.asarray(y_val, dtype=np.float64).reshape(len(y_val), -1)
    y_test_2d = np.asarray(y_test, dtype=np.float64).reshape(len(y_test), -1)
    y_mean = y_train_2d.mean(axis=0, keepdims=True)
    y_std = np.maximum(y_train_2d.std(axis=0, keepdims=True), 1e-6)

    def transform_y(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((value - y_mean) / y_std).astype(np.float32)).to(
            device
        )

    x_train_t = transform_x(x_train)
    x_val_t = transform_x(x_val)
    x_test_t = transform_x(x_test)
    y_train_t = transform_y(y_train_2d)
    y_val_t = transform_y(y_val_2d)
    y_test_t = transform_y(y_test_2d)
    x_select, y_select, select_indices = _select_validation_subset(
        x_val_t, y_val_t, validation_limit
    )
    return {
        "x_train": x_train_t,
        "y_train": y_train_t,
        "x_val": x_val_t,
        "y_val": y_val_t,
        "x_test": x_test_t,
        "y_test": y_test_t,
        "x_val_select": x_select,
        "y_val_select": y_select,
        "validation_selection_indices": select_indices,
        "x_lower": lower.astype(np.float32),
        "x_upper": upper.astype(np.float32),
        "x_scale": scale.astype(np.float32),
        "y_mean": y_mean.astype(np.float32).reshape(-1),
        "y_std": y_std.astype(np.float32).reshape(-1),
        "y_test_raw": y_test_2d.astype(np.float32),
        "input_dim": int(x_train.shape[1]),
        "target_dim": int(y_train_2d.shape[1]),
    }


def add_fixed_linear_projection(
    data: dict[str, Any],
    *,
    projection_dim: int,
    projection_seed: int,
) -> dict[str, Any]:
    """Lift native coordinates with a fixed Gaussian matrix ``A``.

    The matrix never sees targets and is held fixed for training, validation,
    test and all TM ensembles.  Its rows have variance ``1 / raw_dim``.  A
    train-only 0.5/99.5 percentile affine map puts ``A(2x-1)`` into the SLM
    phase interval. Distillation retains the native coordinates and applies
    this exact projector inside the optimization, so synthetic supports remain
    on the fixed lifted-input manifold.
    """

    projection_dim = int(projection_dim)
    if projection_dim <= 0:
        return data
    raw_dim = int(data["input_dim"])
    generator = np.random.default_rng(int(projection_seed))
    matrix_np = generator.normal(
        0.0, 1.0 / math.sqrt(raw_dim), size=(projection_dim, raw_dim)
    ).astype(np.float32)
    matrix = torch.from_numpy(matrix_np).to(data["x_train"].device)

    latent_keys = ("x_train", "x_val", "x_test", "x_val_select")
    for key in latent_keys:
        data[f"{key}_latent"] = data[key]

    train_projected = (2.0 * data["x_train"].float() - 1.0) @ matrix.T
    train_np = train_projected.detach().cpu().numpy()
    lower_np = np.quantile(train_np, 0.005, axis=0).astype(np.float32)
    upper_np = np.quantile(train_np, 0.995, axis=0).astype(np.float32)
    scale_np = np.maximum(upper_np - lower_np, 1e-6).astype(np.float32)
    lower = torch.from_numpy(lower_np).to(matrix.device)
    scale = torch.from_numpy(scale_np).to(matrix.device)

    data.update(
        {
            "projection_matrix": matrix,
            "projection_lower": lower,
            "projection_scale": scale,
            "projection_seed": int(projection_seed),
            "projection_dim": projection_dim,
            "raw_input_dim": raw_dim,
        }
    )
    for key in latent_keys:
        data[key] = project_latent_inputs(data[f"{key}_latent"], data)
    data["input_dim"] = projection_dim
    data["feature_names"] = np.asarray(
        [f"A_x_{index:03d}" for index in range(projection_dim)], dtype=str
    )
    data["input_encoding"] = (
        f"train-scaled native x -> fixed seeded Gaussian A*x ({raw_dim} to "
        f"{projection_dim}) -> train-only affine SLM scaling; A is label-independent"
    )
    return data


def read_canonical_ccpp_sheet(workbook_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the first worksheet only and remove exact duplicate observations."""

    path = Path(workbook_path)
    if not path.is_file():
        raise FileNotFoundError(f"canonical CCPP workbook does not exist: {path}")
    workbook = pd.ExcelFile(path)
    if not workbook.sheet_names:
        raise ValueError(f"canonical CCPP workbook has no worksheets: {path}")
    selected_sheet = str(workbook.sheet_names[0])
    frame = pd.read_excel(workbook, sheet_name=selected_sheet)
    frame = frame.rename(columns={column: str(column).strip().upper() for column in frame})
    if "EP" in frame.columns and TARGET_NAME not in frame.columns:
        frame = frame.rename(columns={"EP": TARGET_NAME})
    required = list(RAW_FEATURE_NAMES) + [TARGET_NAME]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"canonical CCPP sheet is missing columns: {missing}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(np.float64)
    ).all():
        raise ValueError("canonical CCPP sheet contains missing/non-finite numeric values")
    input_rows = len(numeric)
    deduplicated = numeric.drop_duplicates(keep="first").reset_index(drop=True)
    if len(deduplicated) < 20:
        raise ValueError("canonical CCPP sheet is unexpectedly small after deduplication")
    metadata = {
        "workbook_path": str(path),
        "available_sheet_names": [str(value) for value in workbook.sheet_names],
        "selected_sheet_name": selected_sheet,
        "selected_sheet_index": 0,
        "worksheets_read_count": 1,
        "input_row_count": int(input_rows),
        "deduplicated_row_count": int(len(deduplicated)),
        "exact_duplicate_rows_removed": int(input_rows - len(deduplicated)),
        "deduplication_columns": required,
    }
    return deduplicated, metadata


def _fixed_split_indices(
    row_count: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/sealed-test indices without consulting values."""

    indices = np.arange(int(row_count), dtype=np.int64)
    first = np.random.default_rng(int(split_seed)).permutation(indices)
    test_count = int(math.ceil(0.20 * len(first)))
    sealed_test = np.sort(first[:test_count])
    development = first[test_count:]
    second = np.random.default_rng(int(split_seed) + 1).permutation(development)
    validation_count = int(math.ceil(0.20 * len(second)))
    validation = np.sort(second[:validation_count])
    train = np.sort(second[validation_count:])
    if any(
        len(np.intersect1d(left, right))
        for left, right in ((train, validation), (train, sealed_test), (validation, sealed_test))
    ):
        raise AssertionError("CCPP split indices overlap")
    if not np.array_equal(
        np.sort(np.concatenate([train, validation, sealed_test])), indices
    ):
        raise AssertionError("CCPP split does not cover deduplicated rows")
    return train, validation, sealed_test


def preprocess_ccpp(workbook_path, *, device, split_seed, projection_seed=20330821, validation_limit=1024):
    """Use the first canonical sheet, deterministic splits and the native 4-to-64 lift."""
    frame, metadata = read_canonical_ccpp_sheet(workbook_path)
    x = frame[list(RAW_FEATURE_NAMES)].to_numpy(np.float64)
    y = frame[TARGET_NAME].to_numpy(np.float64)[:, None]
    train, validation, test = _fixed_split_indices(len(frame), split_seed)
    data = _scale_arrays(x[train], x[validation], x[test], y[train], y[validation], y[test],
                         device=device, validation_limit=validation_limit)
    data = add_fixed_linear_projection(data, projection_dim=64, projection_seed=projection_seed)
    data.update(train_indices=train, validation_indices=validation, test_indices=test,
                y_train_raw=y[train].astype(np.float32), y_val_raw=y[validation].astype(np.float32),
                canonical_workbook=metadata, raw_feature_names=np.asarray(RAW_FEATURE_NAMES))
    return data


def raw_regression_metrics(prediction_per_tm, data):
    """Compute raw-scale metrics before averaging across independent test TMs."""
    predictions = np.asarray(prediction_per_tm, dtype=np.float64)
    predictions = predictions * np.asarray(data['y_std'], dtype=np.float64) + np.asarray(data['y_mean'], dtype=np.float64)
    truth = np.asarray(data['y_test_raw'], dtype=np.float64)
    values = [regression_metrics(torch.from_numpy(p), torch.from_numpy(truth)) for p in predictions]
    return {key: np.asarray([item[key] for item in values]) for key in ('r2', 'mse', 'rmse', 'mae')}
