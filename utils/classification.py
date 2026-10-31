"""Validation-selected classification distillation and ridge readout evaluation."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from .datasets import class_balanced_indices, random_coreset
from .reservoir_onn import fit_ridge, reservoir_forward, tm_ensemble
from .preprocessing import project_timeseries_inputs


@dataclass(frozen=True)
class ClassificationResult:
    x_init: torch.Tensor
    x_coreset: torch.Tensor
    y_coreset: torch.Tensor
    hard_labels: torch.Tensor
    source_indices: torch.Tensor
    loss_history: np.ndarray
    validation_steps: np.ndarray
    validation_accuracy: np.ndarray
    best_step: int
    best_validation_accuracy: float
    final_temperature: float


def dual_ridge_logits(
    support_features: torch.Tensor,
    support_targets: torch.Tensor,
    query_features: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Return ridge logits through the exact dual of the Exp1/Exp2 readout."""

    support_features = support_features.float()
    support_targets = support_targets.float()
    query_features = query_features.float()
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


def ridge_logits(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    tm: torch.Tensor,
    ridge_lambda: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    combined = reservoir_forward(torch.cat([x_support, x_query]), tm, alpha)
    support_size = len(x_support)
    return dual_ridge_logits(
        combined[:support_size], y_support, combined[support_size:], ridge_lambda
    )


@torch.no_grad()
def evaluate_support(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    tms: Sequence[torch.Tensor],
    ridge_lambda: float,
    alpha: float = 1.0,
    *,
    confusion: bool = False,
) -> dict[str, np.ndarray | float]:
    accuracies: list[float] = []
    num_classes = int(y_support.shape[1])
    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    true = y_query.detach().cpu().numpy()
    for tm in tms:
        prediction = ridge_logits(
            x_support, y_support, x_query, tm, ridge_lambda, alpha
        ).argmax(dim=1)
        prediction_np = prediction.cpu().numpy()
        accuracies.append(float((prediction == y_query).float().mean().item()))
        if confusion:
            np.add.at(counts, (true, prediction_np), 1.0)
    values = np.asarray(accuracies, dtype=np.float64)
    result: dict[str, np.ndarray | float] = {
        "accuracy_per_tm": values,
        "accuracy_mean": float(values.mean()),
        "accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }
    if confusion:
        denominators = counts.sum(axis=1, keepdims=True)
        result["counts"] = counts
        result["percentages"] = 100.0 * counts / np.maximum(denominators, 1.0)
    return result


def _balanced_query_indices(
    labels: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    exact_batch_size: bool = False,
) -> torch.Tensor:
    num_classes = int(labels.max().item()) + 1
    per_class = max(1, int(batch_size) // num_classes)
    selected = []
    for class_index in range(num_classes):
        candidates = torch.nonzero(labels == class_index, as_tuple=True)[0]
        draw_count = (int(batch_size) // num_classes + int(class_index < int(batch_size) % num_classes)
                      if exact_batch_size else per_class)
        draw = torch.randint(
            0,
            len(candidates),
            (draw_count,),
            generator=generator,
            device=labels.device,
        )
        selected.append(candidates[draw])
    return torch.cat(selected)


def distill_classification(
    data: dict[str, Any],
    *,
    samples_per_class: int,
    ridge_lambda: float,
    train_tms: Sequence[torch.Tensor],
    validation_tms: Sequence[torch.Tensor],
    steps: int,
    batch_size: int,
    tms_per_step: int,
    lr_x: float,
    lr_y: float,
    temperature_init: float,
    temperature_lr: float,
    gradient_clip: float,
    seed: int,
    alpha: float = 1.0,
    eval_every: int = 50,
) -> ClassificationResult:
    """Distill X+Y and retain the validation-selected checkpoint."""

    x_train, y_train = data["x_train"], data["y_train"]
    num_classes = int(data["num_classes"])
    source_indices = class_balanced_indices(
        y_train, samples_per_class, num_classes, seed
    )
    x_init = x_train[source_indices].detach().clone()
    hard_labels = y_train[source_indices].detach().clone()
    x_clipped = x_init.clamp(1e-4, 1.0 - 1e-4)
    x_parameter = torch.log(x_clipped / (1.0 - x_clipped)).requires_grad_(True)
    y_parameter = torch.full(
        (len(hard_labels), num_classes),
        -2.0,
        device=x_train.device,
    )
    y_parameter.scatter_(1, hard_labels[:, None], 2.0)
    y_parameter.requires_grad_(True)
    log_temperature = torch.tensor(
        math.log(float(temperature_init)),
        device=x_train.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [x_parameter], "lr": float(lr_x)},
            {"params": [y_parameter], "lr": float(lr_y)},
            {"params": [log_temperature], "lr": float(temperature_lr)},
        ]
    )
    generator = torch.Generator(device=x_train.device).manual_seed(int(seed) + 7_000)
    losses: list[float] = []
    validation_steps: list[int] = []
    validation_accuracy: list[float] = []
    best_score = -math.inf
    best_step = 0
    best_x = None
    best_y = None
    best_temperature = float(temperature_init)

    def current() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.sigmoid(x_parameter), torch.softmax(y_parameter, dim=1), torch.exp(log_temperature)

    def select(step: int) -> None:
        nonlocal best_score, best_step, best_x, best_y, best_temperature
        x_current, y_current, temperature = current()
        evaluation = evaluate_support(
            x_current.detach(),
            y_current.detach(),
            data["x_val"],
            data["y_val"],
            validation_tms,
            ridge_lambda,
            alpha,
        )
        score = float(evaluation["accuracy_mean"])
        validation_steps.append(int(step))
        validation_accuracy.append(score)
        if score > best_score:
            best_score = score
            best_step = int(step)
            best_x = x_current.detach().clone()
            best_y = y_current.detach().clone()
            best_temperature = float(temperature.detach().item())

    select(0)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        query_indices = _balanced_query_indices(y_train, batch_size, generator)
        x_query = x_train[query_indices]
        y_query = y_train[query_indices]
        x_current, y_current, temperature = current()
        tm_indices = torch.randint(
            0,
            len(train_tms),
            (int(tms_per_step),),
            generator=generator,
            device=x_train.device,
        ).tolist()
        loss = torch.zeros((), device=x_train.device)
        for tm_index in tm_indices:
            logits = ridge_logits(
                x_current,
                y_current,
                x_query,
                train_tms[tm_index],
                ridge_lambda,
                alpha,
            )
            loss = loss + F.cross_entropy(logits / temperature, y_query)
        loss = loss / len(tm_indices)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite graph distillation loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [x_parameter, y_parameter, log_temperature], float(gradient_clip)
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if step % int(eval_every) == 0 or step == int(steps):
            select(step)

    assert best_x is not None and best_y is not None
    return ClassificationResult(
        x_init=x_init.detach(),
        x_coreset=best_x,
        y_coreset=best_y,
        hard_labels=hard_labels,
        source_indices=source_indices,
        loss_history=np.asarray(losses, dtype=np.float64),
        validation_steps=np.asarray(validation_steps, dtype=np.int64),
        validation_accuracy=np.asarray(validation_accuracy, dtype=np.float64),
        best_step=best_step,
        best_validation_accuracy=best_score,
        final_temperature=best_temperature,
    )


def search_random_lambda(
    data: dict[str, Any],
    ipc_values: Sequence[int],
    lambda_grid: Sequence[float],
    validation_tms: Sequence[torch.Tensor],
    *,
    random_draws: int,
    seed: int,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Select one ridge_lambda per support size using random supports and validation only."""

    beta_values = np.asarray(lambda_grid, dtype=np.float64)
    scores = np.zeros((len(ipc_values), len(beta_values)), dtype=np.float64)
    for ipc_index, ipc in enumerate(ipc_values):
        for draw in range(int(random_draws)):
            x_support, y_support, _, _ = random_coreset(
                data["x_train"],
                data["y_train"],
                int(ipc),
                int(seed) + 60_000 + 100 * int(ipc) + draw,
                int(data["num_classes"]),
            )
            for beta_index, ridge_lambda in enumerate(beta_values):
                result = evaluate_support(
                    x_support,
                    y_support,
                    data["x_val"],
                    data["y_val"],
                    validation_tms,
                    float(ridge_lambda),
                    alpha,
                )
                scores[ipc_index, beta_index] += float(result["accuracy_mean"])
        scores[ipc_index] /= int(random_draws)
    best_indices = scores.argmax(axis=1)
    return {
        "validation_accuracy": scores,
        "lambda_grid": beta_values,
        "selected_lambda": beta_values[best_indices],
        "lambda_by_size": {
            int(ipc): float(beta_values[index])
            for ipc, index in zip(ipc_values, best_indices)
        },
    }


@torch.no_grad()
def evaluate_full(
    data: dict[str, Any],
    tms: Sequence[torch.Tensor],
    ridge_lambda: float,
    *,
    split: str,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Evaluate full training data with the Exp1/Exp2 primal ridge readout."""

    x_query = data[f"x_{split}"]
    y_query = data[f"y_{split}"]
    one_hot = F.one_hot(data["y_train"], int(data["num_classes"])).float()
    accuracy = []
    for tm in tms:
        features = reservoir_forward(torch.cat([data["x_train"], x_query]), tm, alpha)
        train_size = len(data["x_train"])
        weights = fit_ridge(features[:train_size], one_hot, ridge_lambda)
        prediction = (features[train_size:] @ weights).argmax(dim=1)
        accuracy.append(float((prediction == y_query).float().mean().item()))
    values = np.asarray(accuracy, dtype=np.float64)
    return {
        "accuracy_per_tm": values,
        "accuracy_mean": float(values.mean()),
        "accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def search_full_lambda(
    data: dict[str, Any],
    lambda_grid: Sequence[float],
    validation_tms: Sequence[torch.Tensor],
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    scores = np.asarray(
        [
            float(
                evaluate_full(
                    data, validation_tms, float(ridge_lambda), split="val", alpha=alpha
                )["accuracy_mean"]
            )
            for ridge_lambda in lambda_grid
        ],
        dtype=np.float64,
    )
    best = int(scores.argmax())
    return {
        "validation_accuracy": scores,
        "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
        "selected_lambda": float(lambda_grid[best]),
    }

@dataclass
class TimeSeriesResult:
    x_init: torch.Tensor
    x_coreset: torch.Tensor
    latent_x_init: torch.Tensor
    latent_x_coreset: torch.Tensor
    y_coreset: torch.Tensor
    hard_labels: torch.Tensor
    source_indices: torch.Tensor
    loss_history: np.ndarray
    validation_steps: np.ndarray
    validation_accuracy: np.ndarray
    best_step: int
    best_validation_accuracy: float
    temperature: float


def distill_timeseries(
    data: dict[str, Any],
    *,
    samples_per_class: int,
    ridge_lambda: float,
    train_tms: Sequence[torch.Tensor],
    validation_tms: Sequence[torch.Tensor],
    steps: int,
    batch_size: int,
    tms_per_step: int,
    lr_x: float,
    lr_y: float,
    temperature_init: float,
    temperature_lr: float,
    gradient_clip: float,
    seed: int,
    alpha: float = 1.0,
    eval_every: int = 50,
) -> TimeSeriesResult:
    """Distill native scaled windows through a fixed optical projection."""

    x_train, y_train = data["x_train"], data["y_train"]
    class_count = int(data["num_classes"])
    source_indices = class_balanced_indices(
        y_train, int(samples_per_class), class_count, int(seed)
    )
    x_init = x_train[source_indices].detach().clone()
    latent_train = data.get("x_train_latent", x_train)
    latent_x_init = latent_train[source_indices].detach().clone()
    hard_labels = y_train[source_indices].detach().clone()
    clipped = latent_x_init.clamp(1e-4, 1.0 - 1e-4)
    x_parameter = torch.log(clipped / (1.0 - clipped)).requires_grad_(True)
    y_parameter = torch.full(
        (len(hard_labels), class_count), -2.0, device=x_train.device
    )
    y_parameter.scatter_(1, hard_labels[:, None], 2.0)
    y_parameter.requires_grad_(True)
    log_temperature = torch.tensor(
        math.log(float(temperature_init)), device=x_train.device, requires_grad=True
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [x_parameter], "lr": float(lr_x)},
            {"params": [y_parameter], "lr": float(lr_y)},
            {"params": [log_temperature], "lr": float(temperature_lr)},
        ]
    )
    generator = torch.Generator(device=x_train.device).manual_seed(int(seed) + 7_000)
    losses: list[float] = []
    validation_steps: list[int] = []
    validation_accuracy: list[float] = []
    best_score = -math.inf
    best_step = 0
    best_x: torch.Tensor | None = None
    best_latent_x: torch.Tensor | None = None
    best_y: torch.Tensor | None = None
    best_temperature = float(temperature_init)

    def current() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_x = torch.sigmoid(x_parameter)
        return (
            project_timeseries_inputs(latent_x, data),
            latent_x,
            torch.softmax(y_parameter, dim=1),
            torch.exp(log_temperature).clamp(0.03, 3.0),
        )

    def select(step: int) -> None:
        nonlocal best_score, best_step, best_x, best_latent_x, best_y, best_temperature
        x_current, latent_current, y_current, temperature = current()
        evaluation = evaluate_support(
            x_current.detach(),
            y_current.detach(),
            data["x_val"],
            data["y_val"],
            validation_tms,
            ridge_lambda,
            alpha,
        )
        score = float(evaluation["accuracy_mean"])
        validation_steps.append(int(step))
        validation_accuracy.append(score)
        if score > best_score:
            best_score = score
            best_step = int(step)
            best_x = x_current.detach().clone()
            best_latent_x = latent_current.detach().clone()
            best_y = y_current.detach().clone()
            best_temperature = float(temperature.detach().item())

    select(0)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        query_indices = _balanced_query_indices(y_train, batch_size, generator, exact_batch_size=True)
        x_query = x_train[query_indices]
        y_query = y_train[query_indices]
        x_current, _, y_current, temperature = current()
        tm_indices = torch.randint(
            0,
            len(train_tms),
            (int(tms_per_step),),
            generator=generator,
            device=x_train.device,
        ).tolist()
        loss = torch.zeros((), device=x_train.device)
        for tm_index in tm_indices:
            logits = ridge_logits(
                x_current,
                y_current,
                x_query,
                train_tms[tm_index],
                ridge_lambda,
                alpha,
            )
            loss = loss + F.cross_entropy(logits / temperature, y_query)
        loss = loss / len(tm_indices)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite time-series loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [x_parameter, y_parameter, log_temperature], float(gradient_clip)
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if step % int(eval_every) == 0 or step == int(steps):
            select(step)

    if best_x is None or best_latent_x is None or best_y is None:
        raise RuntimeError("validation checkpoint selection failed")
    return TimeSeriesResult(
        x_init=x_init,
        x_coreset=best_x,
        latent_x_init=latent_x_init,
        latent_x_coreset=best_latent_x,
        y_coreset=best_y,
        hard_labels=hard_labels,
        source_indices=source_indices,
        loss_history=np.asarray(losses, dtype=np.float64),
        validation_steps=np.asarray(validation_steps, dtype=np.int64),
        validation_accuracy=np.asarray(validation_accuracy, dtype=np.float64),
        best_step=best_step,
        best_validation_accuracy=best_score,
        temperature=best_temperature,
    )

@torch.no_grad()
def search_text_lambda(
    data: dict,
    ipc_values: Sequence[int],
    lambda_grid: Sequence[float],
    output_dim: int = 512,
    tm_count: int = 10,
    random_draws: int = 5,
    base_seed: int = 2026,
    device: str = "cuda:0",
    alpha: float = 1.0,
) -> dict[str, np.ndarray | dict[int, float]]:
    """Tune one ridge ridge_lambda per coreset size on a held-out validation split."""

    if "x_val" not in data or "y_val" not in data:
        raise KeyError("data must contain x_val and y_val for ridge_lambda search")
    ipc_values = [int(value) for value in ipc_values]
    lambda_grid = [float(value) for value in lambda_grid]
    tms = tm_ensemble(
        data["input_dim"],
        int(output_dim),
        range(base_seed + 50_000, base_seed + 50_000 + int(tm_count)),
        device,
    )
    scores = np.zeros((len(ipc_values), len(lambda_grid)), dtype=np.float64)
    num_classes = int(data["num_classes"])
    for ipc_index, ipc in enumerate(ipc_values):
        totals = np.zeros(len(lambda_grid), dtype=np.float64)
        count = 0
        for draw in range(int(random_draws)):
            x_coreset, y_coreset, _, _ = random_coreset(
                data["x_train"],
                data["y_train"],
                ipc,
                base_seed + 60_000 + 100 * ipc + draw,
                num_classes,
            )
            coreset_size = int(x_coreset.shape[0])
            for tm in tms:
                features = reservoir_forward(
                    torch.cat([x_coreset, data["x_val"]]), tm, alpha
                )
                for beta_index, ridge_lambda in enumerate(lambda_grid):
                    weight = fit_ridge(
                        features[:coreset_size], y_coreset, float(ridge_lambda)
                    )
                    prediction = torch.argmax(
                        features[coreset_size:] @ weight, dim=1
                    )
                    totals[beta_index] += float(
                        (prediction == data["y_val"]).float().mean().item()
                    )
                count += 1
        scores[ipc_index] = totals / count
        best_index = int(np.argmax(scores[ipc_index]))
        print(
            f"ridge_lambda search IPC={ipc:3d}: ridge_lambda={lambda_grid[best_index]:g}, "
            f"validation={scores[ipc_index, best_index]:.6f}",
            flush=True,
        )
    best_indices = np.argmax(scores, axis=1)
    return {
        "ipc_values": np.asarray(ipc_values, dtype=np.int32),
        "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
        "validation_accuracy": scores,
        "selected_lambda": np.asarray(
            [lambda_grid[index] for index in best_indices], dtype=np.float64
        ),
        "lambda_by_size": {
            ipc: float(lambda_grid[index])
            for ipc, index in zip(ipc_values, best_indices)
        },
        "tm_seeds": np.arange(
            base_seed + 50_000,
            base_seed + 50_000 + int(tm_count),
            dtype=np.int64,
        ),
    }


@torch.no_grad()
def search_text_full_lambda(
    data: dict,
    lambda_grid: Sequence[float],
    output_dim: int = 512,
    tm_count: int = 10,
    base_seed: int = 2026,
    device: str = "cuda:0",
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Tune the full-training-data ridge ridge_lambda without using the test split."""

    if "x_val" not in data or "y_val" not in data:
        raise KeyError("data must contain x_val and y_val for ridge_lambda search")
    lambda_grid = [float(value) for value in lambda_grid]
    tms = tm_ensemble(
        data["input_dim"],
        int(output_dim),
        range(base_seed + 70_000, base_seed + 70_000 + int(tm_count)),
        device,
    )
    y_train = F.one_hot(
        data["y_train"], int(data["num_classes"])
    ).float()
    scores = np.zeros(len(lambda_grid), dtype=np.float64)
    train_size = int(data["x_train"].shape[0])
    for tm in tms:
        features = reservoir_forward(
            torch.cat([data["x_train"], data["x_val"]]), tm, alpha
        )
        for beta_index, ridge_lambda in enumerate(lambda_grid):
            weight = fit_ridge(features[:train_size], y_train, ridge_lambda)
            prediction = torch.argmax(features[train_size:] @ weight, dim=1)
            scores[beta_index] += float(
                (prediction == data["y_val"]).float().mean().item()
            )
    scores /= len(tms)
    best_index = int(np.argmax(scores))
    print(
        f"full-data ridge_lambda search: ridge_lambda={lambda_grid[best_index]:g}, "
        f"validation={scores[best_index]:.6f}",
        flush=True,
    )
    return {
        "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
        "validation_accuracy": scores,
        "selected_lambda": float(lambda_grid[best_index]),
        "tm_seeds": np.arange(
            base_seed + 70_000,
            base_seed + 70_000 + int(tm_count),
            dtype=np.int64,
        ),
    }
