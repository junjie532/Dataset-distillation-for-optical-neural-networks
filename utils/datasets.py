"""Fashion-MNIST data, balanced supports and numerical result I/O."""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


def flat_result_path(path):
    """Store results directly in results/, using __ to separate filename prefixes."""
    path = Path(path)
    if 'results' not in path.parts:
        return path
    index = path.parts.index('results')
    root = Path(*path.parts[:index + 1])
    return root / '__'.join(path.parts[index + 1:])


def flat_result_directory(path):
    """Return results/ for a result prefix, or the original non-result directory."""
    path = Path(path)
    if 'results' not in path.parts:
        return path
    return Path(*path.parts[:path.parts.index('results') + 1])


def _dataset_to_tensors(dataset, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.utils.data import DataLoader

    images, labels = [], []
    for image, label in DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0):
        images.append(image.reshape(image.shape[0], -1))
        labels.append(torch.as_tensor(label, dtype=torch.long))
    return torch.cat(images).float().to(device), torch.cat(labels).long().to(device)


def load_fmnist(
    data_root: str | Path,
    device: str | torch.device = "cpu",
    download: bool = True,
) -> dict[str, Any]:
    """Load the official Fashion-MNIST split as flattened masks in ``[0, 1]``."""

    from torchvision import datasets, transforms

    root = Path(data_root)
    transform = transforms.ToTensor()
    train = datasets.FashionMNIST(root, train=True, download=download, transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=download, transform=transform)
    x_train, y_train = _dataset_to_tensors(train, device)
    x_test, y_test = _dataset_to_tensors(test, device)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "input_dim": 784,
        "num_classes": 10,
        "image_shape": (28, 28),
        "class_names": np.asarray(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        ),
    }


def class_balanced_indices(
    labels: torch.Tensor,
    images_per_class: int,
    num_classes: int = 10,
    seed: int = 0,
) -> torch.Tensor:
    """Select a reproducible balanced subset ordered by class."""

    generator = torch.Generator(device=labels.device)
    generator.manual_seed(int(seed))
    selected = []
    for class_index in range(int(num_classes)):
        candidates = torch.nonzero(labels == class_index, as_tuple=True)[0]
        permutation = torch.randperm(
            candidates.numel(), generator=generator, device=labels.device
        )
        selected.append(candidates[permutation[: int(images_per_class)]])
    return torch.cat(selected)


def random_coreset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    images_per_class: int,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return balanced real images, one-hot labels, hard labels and indices."""

    indices = class_balanced_indices(y_train, images_per_class, num_classes, seed)
    hard_labels = y_train[indices]
    soft_labels = F.one_hot(hard_labels, num_classes).float()
    return x_train[indices], soft_labels, hard_labels, indices


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_npz(
    path: str | Path,
    arrays: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Path:
    """Save numeric results; embed the config JSON so the NPZ is standalone."""

    path = flat_result_path(Path(path))
    flat_result_directory(path.parent).mkdir(parents=True, exist_ok=True)
    output = {key: _to_numpy(value) for key, value in arrays.items()}
    if config is not None:
        output["config_json"] = np.asarray(json.dumps(_jsonable(config)))
    np.savez_compressed(flat_result_path(path), **output)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(flat_result_path(Path(path)), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_distillation(path, result, config, train_tm_seeds, test_tm_seeds=None,
                      evaluation=None, metadata=None):
    """Export optimized inputs, targets, training history and reproduction seeds."""
    arrays = {
        'initial_images': result.x_coreset_init,
        'synthetic_images': result.x_coreset,
        'synthetic_labels': result.y_coreset,
        'hard_labels': result.coreset_labels,
        'source_indices': result.coreset_indices,
        'loss': result.loss_history,
        'accuracy_steps': result.step_history,
        'accuracy_history': result.accuracy_history,
        'train_tm_seeds': np.asarray(train_tm_seeds),
    }
    if result.final_temperature is not None:
        arrays['final_temperature'] = result.final_temperature
    if test_tm_seeds is not None:
        arrays['test_tm_seeds'] = np.asarray(test_tm_seeds)
    if evaluation is not None:
        arrays['accuracy_per_tm'] = evaluation['accuracy_per_tm']
    settings = {'distillation': config}
    if metadata is not None:
        settings.update(metadata)
    return save_npz(path, arrays, settings)


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value


def save_checkpoint(path, checkpoint):
    path = flat_result_path(Path(path))
    flat_result_directory(path.parent).mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(to_cpu(checkpoint), flat_result_path(temporary))
    flat_result_path(temporary).replace(flat_result_path(path))


def save_json(path, content):
    path = flat_result_path(Path(path))
    flat_result_directory(path.parent).mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    flat_result_path(temporary).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    flat_result_path(temporary).replace(flat_result_path(path))
