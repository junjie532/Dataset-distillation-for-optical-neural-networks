"""Input-space compression, RBF-KIP and optical distribution matching.

Numerical objectives, initialization, normalization and ridge selection follow
the archived Fashion-MNIST experiments. No saved supports or scores are required.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .reservoir_onn import set_seed

NUM_CLASSES = 10


@torch.no_grad()
def classwise_herding(
    x_train: torch.Tensor, y_train: torch.Tensor, ipc: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select distinct real images by greedy class-wise pixel-mean matching.

    This is the historical additional-baseline algorithm: at selection t,
    minimize ||class_mean - (selected_sum + candidate) / t||^2.
    Inputs are never optimized and labels remain fixed one-hot vectors.
    """
    selected_global = []
    for class_index in range(NUM_CLASSES):
        global_indices = torch.nonzero(y_train == class_index, as_tuple=True)[0]
        points = x_train[global_indices]
        if not 1 <= int(ipc) <= len(points):
            raise ValueError('IPC must be between one and the class size')
        target_mean = points.mean(dim=0)
        running_sum = torch.zeros_like(target_mean)
        chosen_local = []
        available = torch.ones(len(points), dtype=torch.bool, device=points.device)
        for step in range(int(ipc)):
            candidate_means = (running_sum[None, :] + points) / float(step + 1)
            error = ((candidate_means - target_mean[None, :]) ** 2).sum(dim=1)
            error[~available] = torch.inf
            next_index = int(torch.argmin(error).item())
            chosen_local.append(next_index)
            available[next_index] = False
            running_sum = running_sum + points[next_index]
        selected_global.append(global_indices[torch.tensor(chosen_local, device=points.device)])
    indices = torch.cat(selected_global)
    return x_train[indices].clone(), F.one_hot(y_train[indices], NUM_CLASSES).float(), indices


def squared_distances(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance, clipped for roundoff."""
    return torch.clamp(
        (x * x).sum(dim=1, keepdim=True)
        + (centers * centers).sum(dim=1).unsqueeze(0)
        - 2.0 * (x @ centers.T),
        min=0.0,
    )


def _sample_kmeans_plus_plus(
    points: torch.Tensor, k: int, generator: torch.Generator
) -> torch.Tensor:
    first = int(
        torch.randint(
            points.shape[0], (1,), generator=generator, device=points.device
        ).item()
    )
    indices = [first]
    min_distance = squared_distances(points, points[first : first + 1])[:, 0]
    for _ in range(1, int(k)):
        probabilities = min_distance / torch.clamp(min_distance.sum(), min=1e-12)
        next_index = int(
            torch.multinomial(probabilities, 1, generator=generator).item()
        )
        indices.append(next_index)
        min_distance = torch.minimum(
            min_distance,
            squared_distances(points, points[next_index : next_index + 1])[:, 0],
        )
    return points[torch.as_tensor(indices, device=points.device)].clone()


@torch.no_grad()
def _torch_kmeans(
    points: torch.Tensor, k: int, seed: int, max_iter: int, tolerance: float = 1e-05
) -> tuple[torch.Tensor, int]:
    if int(k) == 1:
        return (points.mean(dim=0, keepdim=True), 1)
    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))
    centers = _sample_kmeans_plus_plus(points, int(k), generator)
    completed = 0
    for iteration in range(int(max_iter)):
        assignment = torch.argmin(squared_distances(points, centers), dim=1)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, assignment, points)
        counts = torch.bincount(assignment, minlength=int(k)).to(points.dtype)
        empty = torch.nonzero(counts == 0, as_tuple=True)[0]
        if empty.numel():
            nearest = squared_distances(points, centers).min(dim=1).values
            replacement_order = torch.argsort(nearest, descending=True)
            for (offset, cluster_index) in enumerate(empty.tolist()):
                new_centers[cluster_index] = points[replacement_order[offset]]
                counts[cluster_index] = 1.0
        new_centers = new_centers / counts[:, None]
        shift = torch.max(torch.linalg.vector_norm(new_centers - centers, dim=1))
        centers = new_centers
        completed = iteration + 1
        if float(shift.item()) <= float(tolerance):
            break
    return (centers.clamp_(0.0, 1.0), completed)


@torch.no_grad()
def classwise_kmeans_prototypes(
    x_train: torch.Tensor, y_train: torch.Tensor, ipc: int, seed: int, max_iter: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    (centers, iterations) = ([], [])
    for class_index in range(NUM_CLASSES):
        points = x_train[y_train == class_index]
        (class_centers, completed) = _torch_kmeans(
            points, int(ipc), int(seed) + 1009 * class_index, int(max_iter)
        )
        centers.append(class_centers)
        iterations.append(completed)
    masks = torch.cat(centers)
    hard_labels = torch.arange(NUM_CLASSES, device=x_train.device).repeat_interleave(
        int(ipc)
    )
    targets = F.one_hot(hard_labels, NUM_CLASSES).float()
    return (masks, targets, np.asarray(iterations, dtype=np.int32))


def estimate_rbf_scale(x_train: torch.Tensor, seed: int, pairs: int = 4096) -> float:
    generator = torch.Generator(device=x_train.device)
    generator.manual_seed(int(seed))
    left = torch.randint(
        x_train.shape[0], (int(pairs),), generator=generator, device=x_train.device
    )
    right = torch.randint(
        x_train.shape[0], (int(pairs),), generator=generator, device=x_train.device
    )
    distance = ((x_train[left] - x_train[right]) ** 2).sum(dim=1)
    positive = distance[distance > 0]
    return float(torch.median(positive).item())


def rbf_kernel(
    left: torch.Tensor, right: torch.Tensor, median_squared_distance: float
) -> torch.Tensor:
    return torch.exp(-squared_distances(left, right) / median_squared_distance)


def optimize_rbf_kip(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ipc: int,
    seed: int,
    steps: int,
    batch_size: int,
    lr_x: float,
    lr_y: float,
    kip_ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray | float]]:
    """Optimize inducing inputs/labels for a fixed digital RBF KRR model."""
    set_seed(seed)
    generator = torch.Generator(device=x_train.device)
    generator.manual_seed(int(seed) + 7000)
    initial_indices = []
    for class_index in range(NUM_CLASSES):
        candidates = torch.nonzero(y_train == class_index, as_tuple=True)[0]
        order = torch.randperm(
            candidates.numel(), generator=generator, device=x_train.device
        )
        initial_indices.append(candidates[order[: int(ipc)]])
    initial_indices_tensor = torch.cat(initial_indices)
    hard_labels = y_train[initial_indices_tensor]
    x_synthetic = x_train[initial_indices_tensor].clone().detach().requires_grad_(True)
    y_synthetic = (
        F.one_hot(hard_labels, NUM_CLASSES)
        .float()
        .clone()
        .detach()
        .requires_grad_(True)
    )
    median_distance = estimate_rbf_scale(x_train, seed + 17000)
    optimizer = torch.optim.Adam(
        [
            {"params": [x_synthetic], "lr": float(lr_x)},
            {"params": [y_synthetic], "lr": float(lr_y)},
        ]
    )
    loss_history = []
    identity = torch.eye(
        x_synthetic.shape[0], device=x_train.device, dtype=torch.float32
    )
    for step in range(1, int(steps) + 1):
        query_indices = torch.randint(
            x_train.shape[0],
            (int(batch_size),),
            generator=generator,
            device=x_train.device,
        )
        x_query = x_train[query_indices]
        y_query = F.one_hot(y_train[query_indices], NUM_CLASSES).float()
        optimizer.zero_grad(set_to_none=True)
        k_ss = rbf_kernel(x_synthetic, x_synthetic, median_distance)
        coefficients = torch.linalg.solve(
            k_ss + float(kip_ridge) * identity, y_synthetic
        )
        prediction = rbf_kernel(x_query, x_synthetic, median_distance) @ coefficients
        loss = F.mse_loss(prediction, y_query)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x_synthetic.clamp_(0.0, 1.0)
        loss_history.append(float(loss.item()))
    diagnostics = {
        "loss_history": np.asarray(loss_history, dtype=np.float64),
        "median_squared_distance": float(median_distance),
        "source_indices": initial_indices_tensor.detach().cpu().numpy(),
    }
    return (x_synthetic.detach(), y_synthetic.detach(), diagnostics)


@torch.no_grad()
def raw_optical_intensity(
    x: torch.Tensor, tm: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    chunks = []
    for start in range(0, x.shape[0], int(chunk_size)):
        masks = x[start : start + int(chunk_size)].float()
        field = torch.exp(1j * torch.pi * masks)
        chunks.append((torch.abs(field @ tm) ** 2).float())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def ridge_accuracy_from_raw(
    support_raw: torch.Tensor,
    targets: torch.Tensor,
    query_raw: torch.Tensor,
    query_labels: torch.Tensor,
    beta: float,
) -> float:
    scale = torch.maximum(support_raw.max(), query_raw.max()).clamp_min(1e-12)
    z_support = support_raw / scale
    z_query = query_raw / scale
    sample_count = z_support.shape[0]
    dual = torch.linalg.solve(
        z_support @ z_support.T
        + float(beta)
        * torch.eye(sample_count, device=z_support.device, dtype=z_support.dtype),
        targets.float(),
    )
    weight = z_support.T @ dual
    prediction = torch.argmax(z_query @ weight, dim=1)
    return float((prediction == query_labels).float().mean().item())


def balanced_validation_indices(
    labels: torch.Tensor, per_class: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=labels.device)
    generator.manual_seed(int(seed))
    chosen = []
    for class_index in range(NUM_CLASSES):
        candidates = torch.nonzero(labels == class_index, as_tuple=True)[0]
        order = torch.randperm(
            candidates.numel(), generator=generator, device=labels.device
        )
        chosen.append(candidates[order[: int(per_class)]])
    return torch.cat(chosen)


def class_balanced_initial_support(
    x_train: torch.Tensor, y_train: torch.Tensor, ipc: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=x_train.device)
    generator.manual_seed(int(seed))
    indices = []
    for class_index in range(NUM_CLASSES):
        candidates = torch.nonzero(y_train == class_index, as_tuple=True)[0]
        order = torch.randperm(
            candidates.numel(), generator=generator, device=x_train.device
        )
        indices.append(candidates[order[: int(ipc)]])
    index = torch.cat(indices)
    return (x_train[index].clone(), index)


def optimize_optical_dm(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ipc: int,
    seed: int,
    tms: list[torch.Tensor],
    target_means: torch.Tensor,
    target_scales: torch.Tensor,
    steps: int,
    tms_per_step: int,
    learning_rate: float,
    grad_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    """Match exact per-class optical feature means with fixed hard labels."""
    (x_initial, source_indices) = class_balanced_initial_support(
        x_train, y_train, ipc, seed
    )
    x_synthetic = x_initial.detach().clone().requires_grad_(True)
    hard_labels = torch.arange(NUM_CLASSES, device=x_train.device).repeat_interleave(
        int(ipc)
    )
    targets = F.one_hot(hard_labels, NUM_CLASSES).float()
    optimizer = torch.optim.Adam([x_synthetic], lr=float(learning_rate))
    losses = []
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for offset in range(int(tms_per_step)):
            tm_index = ((step - 1) * int(tms_per_step) + offset) % len(tms)
            raw = (
                torch.abs(
                    torch.exp(1j * torch.pi * x_synthetic.float()) @ tms[tm_index]
                )
                ** 2
            )
            features = raw.float() / target_scales[tm_index].clamp_min(1e-12)
            synthetic_means = features.reshape(
                NUM_CLASSES, int(ipc), features.shape[1]
            ).mean(dim=1)
            total_loss = total_loss + F.mse_loss(
                synthetic_means, target_means[tm_index]
            )
        loss = total_loss / float(tms_per_step)
        loss.backward()
        if float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_([x_synthetic], float(grad_clip))
        optimizer.step()
        with torch.no_grad():
            x_synthetic.clamp_(0.0, 1.0)
        losses.append(float(loss.item()))
    diagnostics = {
        "loss_history": np.asarray(losses, dtype=np.float64),
        "source_indices": source_indices.detach().cpu().numpy(),
        "x_initial": x_initial.detach().cpu().numpy().astype(np.float32),
    }
    return (x_synthetic.detach(), targets, diagnostics)
