"""Differentiable optimization of continuous inputs and target vectors."""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from .reservoir_onn import fit_ridge, reservoir_forward

from dataclasses import dataclass
from .datasets import class_balanced_indices
from .evaluation import evaluate_coreset

@dataclass(frozen=True)
class DistillationConfig:
    num_classes: int = 10
    images_per_class: int = 1
    ridge_lambda: float = 0.1
    steps: int = 700
    target_batch_size: int = 1024
    tms_per_step: int = 4
    lr_x: float = 1e-2
    lr_y: float = 1e-2
    loss: str = "cross_entropy"
    optimize_x: bool = True
    optimize_y: bool = True
    temperature: float = 1.0
    learn_temperature: bool = True
    temperature_lr: float = 1e-2
    alpha: float = 1.0
    coreset_seed: int = 2025
    batch_seed: int = 9025
    eval_every: int = 0
    grad_clip: float | None = None
    mask_parameterization: str = "clamp"
    label_parameterization: str = "free"
    tm_sampling: str = "cyclic"
    ridge_solver: str = "primal"
    batched_cached_tms: bool = False


@dataclass
class DistillationResult:
    x_coreset_init: torch.Tensor
    x_coreset: torch.Tensor
    y_coreset: torch.Tensor
    coreset_labels: torch.Tensor
    coreset_indices: torch.Tensor
    loss_history: np.ndarray
    step_history: np.ndarray
    accuracy_history: np.ndarray
    final_temperature: Optional[float]


def distill_coreset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_tms: Sequence[torch.Tensor],
    config: DistillationConfig,
    x_eval: Optional[torch.Tensor] = None,
    y_eval: Optional[torch.Tensor] = None,
    eval_tms: Optional[Sequence[torch.Tensor]] = None,
    precomputed_query_intensity: Optional[Sequence[torch.Tensor]] = None,
) -> DistillationResult:
    """Optimize continuous masks and soft labels with the original Exp1 loop."""

    device = x_train.device
    if precomputed_query_intensity is not None:
        if len(precomputed_query_intensity) != len(train_tms):
            raise ValueError('One raw query-intensity cache is required per TM')
        for raw, tm in zip(precomputed_query_intensity, train_tms):
            if raw.shape != (len(x_train), tm.shape[1]) or raw.requires_grad:
                raise ValueError('Query caches must be fixed raw intensities in training-image order')
    indices = class_balanced_indices(
        y_train, config.images_per_class, config.num_classes, config.coreset_seed
    )
    x_init = x_train[indices].clone().detach()
    hard_labels = y_train[indices].clone().detach()
    if config.mask_parameterization == "sigmoid":
        clipped = x_init.clamp(1e-4, 1.0 - 1e-4)
        x_parameter = torch.log(clipped / (1.0 - clipped)).requires_grad_(
            config.optimize_x
        )
    elif config.mask_parameterization == "clamp":
        x_parameter = x_init.clone().requires_grad_(config.optimize_x)
    else:
        raise ValueError(
            f"unknown mask_parameterization: {config.mask_parameterization}"
        )

    one_hot = F.one_hot(hard_labels, config.num_classes).float()
    if config.label_parameterization == "softmax":
        y_parameter = torch.full(
            one_hot.shape, -2.0, dtype=one_hot.dtype, device=device
        )
        y_parameter.scatter_(1, hard_labels[:, None], 2.0)
        y_parameter.requires_grad_(config.optimize_y)
    elif config.label_parameterization == "free":
        y_parameter = one_hot.clone().requires_grad_(config.optimize_y)
    else:
        raise ValueError(
            f"unknown label_parameterization: {config.label_parameterization}"
        )

    def current_coreset() -> tuple[torch.Tensor, torch.Tensor]:
        x_current = (
            torch.sigmoid(x_parameter)
            if config.mask_parameterization == "sigmoid"
            else x_parameter
        )
        y_current = (
            torch.softmax(y_parameter, dim=1)
            if config.label_parameterization == "softmax"
            else y_parameter
        )
        return x_current, y_current

    parameters = []
    if config.optimize_x:
        parameters.append({"params": x_parameter, "lr": config.lr_x})
    if config.optimize_y:
        parameters.append({"params": y_parameter, "lr": config.lr_y})
    log_temperature = None
    if config.loss in {"ce", "cross_entropy"} and config.learn_temperature:
        log_temperature = torch.tensor(
            float(np.log(config.temperature)), device=device, requires_grad=True
        )
        parameters.append({"params": [log_temperature], "lr": config.temperature_lr})
    optimizer = torch.optim.Adam(parameters)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.batch_seed))

    losses, recorded_steps, recorded_accuracy = [], [], []
    batched_tms = None
    if config.batched_cached_tms and precomputed_query_intensity is not None:
        if config.ridge_solver != 'dual':
            raise ValueError('Batched cached training uses the dual ridge solver')
        batched_tms = torch.stack(list(train_tms))

    def record(step: int) -> None:
        if x_eval is None or y_eval is None or eval_tms is None:
            return
        x_coreset, y_coreset = current_coreset()
        evaluation = evaluate_coreset(
            x_coreset.detach(),
            y_coreset.detach(),
            x_eval,
            y_eval,
            eval_tms,
            config.ridge_lambda,
            config.alpha,
        )
        recorded_steps.append(step)
        recorded_accuracy.append(evaluation["accuracy_mean"])

    if config.eval_every > 0:
        record(0)

    coreset_size = x_parameter.shape[0]
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        query_indices = torch.randint(
            0,
            x_train.shape[0],
            (config.target_batch_size,),
            generator=generator,
            device=device,
        )
        x_query = x_train[query_indices]
        y_query = y_train[query_indices]
        y_query_onehot = F.one_hot(y_query, config.num_classes).float()
        x_coreset, y_coreset = current_coreset()
        total_loss = 0.0
        if config.tm_sampling == "random":
            tm_indices = torch.randint(
                0,
                len(train_tms),
                (config.tms_per_step,),
                generator=generator,
                device=device,
            ).tolist()
        elif config.tm_sampling == "cyclic":
            tm_indices = [
                (step * config.tms_per_step + offset) % len(train_tms)
                for offset in range(config.tms_per_step)
            ]
        else:
            raise ValueError(f"unknown tm_sampling: {config.tm_sampling}")
        if batched_tms is not None:
            # Batch the same per-TM objectives to avoid four small CUDA solver
            # synchronizations per update. No TM or query sampling is changed.
            chosen_tms = batched_tms[tm_indices]
            support_raw = torch.abs(torch.matmul(torch.exp(
                1j * float(config.alpha) * torch.pi * x_coreset.float()
            ), chosen_tms)) ** 2
            if torch.is_tensor(precomputed_query_intensity):
                tm_ids = torch.tensor(tm_indices, device=device)
                query_raw = precomputed_query_intensity[tm_ids[:,None],query_indices[None,:],:]
            else:
                query_raw = torch.stack([precomputed_query_intensity[i][query_indices] for i in tm_indices])
            raw = torch.cat([support_raw, query_raw], dim=1)
            features = raw / torch.clamp(raw.amax(dim=(1,2),keepdim=True),min=1e-12)
            z_support, z_query = features[:,:coreset_size], features[:,coreset_size:]
            gram = z_support @ z_support.transpose(1,2)
            identity = torch.eye(coreset_size,device=device,dtype=gram.dtype)
            dual = torch.linalg.solve_ex(gram + float(config.ridge_lambda)*identity,
                y_coreset.expand(len(tm_indices),-1,-1),check_errors=False).result
            logits = z_query @ (z_support.transpose(1,2) @ dual)
            if config.loss == 'mse':
                step_loss = F.mse_loss(logits,y_query_onehot.expand(len(tm_indices),-1,-1))
            else:
                temperature = torch.exp(log_temperature) if log_temperature is not None else config.temperature
                step_loss = F.cross_entropy((logits/temperature).reshape(-1,config.num_classes),
                                            y_query.repeat(len(tm_indices)))
            total_loss = step_loss * config.tms_per_step
        for tm_index in ([] if batched_tms is not None else tm_indices):
            if precomputed_query_intensity is None:
                features = reservoir_forward(
                    torch.cat([x_coreset, x_query]), train_tms[tm_index], config.alpha
                )
            else:
                # Queries are fixed training images. Cache only their raw forward
                # pass; retain the original support/query joint normalization
                # and its gradient through the learned support intensities.
                support_raw = torch.abs(torch.exp(
                    1j * float(config.alpha) * torch.pi * x_coreset.float()
                ) @ train_tms[tm_index]) ** 2
                query_raw = precomputed_query_intensity[tm_index][query_indices]
                features = torch.cat([support_raw, query_raw])
                features = features / torch.clamp(torch.amax(features), min=1e-12)
            weight = fit_ridge(features[:coreset_size], y_coreset,
                               config.ridge_lambda, solver=config.ridge_solver)
            logits = features[coreset_size:] @ weight
            if config.loss == "mse":
                step_loss = F.mse_loss(logits, y_query_onehot)
            else:
                temperature = (
                    torch.exp(log_temperature)
                    if log_temperature is not None
                    else config.temperature
                )
                step_loss = F.cross_entropy(logits / temperature, y_query)
            total_loss = total_loss + step_loss
        loss = total_loss / config.tms_per_step
        loss.backward()
        if config.grad_clip is not None:
            clip_parameters = []
            if config.optimize_x:
                clip_parameters.append(x_parameter)
            if config.optimize_y:
                clip_parameters.append(y_parameter)
            if log_temperature is not None:
                clip_parameters.append(log_temperature)
            torch.nn.utils.clip_grad_norm_(clip_parameters, float(config.grad_clip))
        optimizer.step()
        with torch.no_grad():
            if config.optimize_x and config.mask_parameterization == "clamp":
                x_parameter.clamp_(0.0, 1.0)
        losses.append(float(loss.item()))
        if config.eval_every > 0 and step % config.eval_every == 0:
            record(step)

    final_temperature = (
        None
        if log_temperature is None
        else float(torch.exp(log_temperature.detach()).item())
    )
    x_coreset, y_coreset = current_coreset()
    return DistillationResult(
        x_coreset_init=x_init,
        x_coreset=x_coreset.detach(),
        y_coreset=y_coreset.detach(),
        coreset_labels=hard_labels,
        coreset_indices=indices,
        loss_history=np.asarray(losses, dtype=np.float64),
        step_history=np.asarray(recorded_steps, dtype=np.int32),
        accuracy_history=np.asarray(recorded_accuracy, dtype=np.float64),
        final_temperature=final_temperature,
    )
