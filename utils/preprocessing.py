"""Task-specific input representations and recorded train/validation/test splits."""
from __future__ import annotations
from .datasets import flat_result_path, flat_result_directory
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from .datasets import save_npz


HAR_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
HAR_CLASS_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)

def _read_agnews_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read the original AG News CSV format and remove exact duplicates."""

    texts: list[str] = []
    labels: list[int] = []
    seen: set[tuple[str, int]] = set()
    with flat_result_path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            text = f"{row[1]} {row[2]}".strip()
            label = int(row[0]) - 1
            key = (text, label)
            if key not in seen:
                seen.add(key)
                texts.append(text)
                labels.append(label)
    return texts, np.asarray(labels, dtype=np.int64)


def preprocess_agnews(
    data_root: str | Path,
    device: str | torch.device = "cpu",
    cache_path: str | Path | None = None,
    seed: int = 2026,
    svd_dim: int = 100,
    max_features: int = 20_000,
    min_df: int = 2,
    val_per_class: int = 1_000,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Load AG News as reproducible TF-IDF/SVD phase masks in ``[0, 1]``.

    The vectorizer and SVD are fitted only on the training portion after a
    stratified validation split. Each SVD coordinate is scaled with the 1st
    and 99th training percentiles, matching the original text experiment.
    A numeric cache avoids refitting TF-IDF/SVD on later notebook executions.
    """

    root = Path(data_root)
    if (root / "AGNews").is_dir():
        root = root / "AGNews"
    cache = None if cache_path is None else Path(cache_path)

    specification = {
        "seed": int(seed), "svd_dim": int(svd_dim), "max_features": int(max_features),
        "min_df": int(min_df), "val_per_class": int(val_per_class),
    }
    metadata = None if cache is None else cache.with_suffix(".json")
    cached_config = (json.loads(flat_result_path(metadata).read_text())
                     if metadata is not None and flat_result_path(metadata).exists() else {})
    valid_cache = (cache is not None and flat_result_path(cache).exists()
                   and all(cached_config.get(key) == value for key, value in specification.items()))
    if valid_cache and not rebuild:
        with np.load(flat_result_path(cache), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    else:
        train_path, test_path = root / "train.csv", root / "test.csv"
        if not flat_result_path(train_path).exists() or not flat_result_path(test_path).exists():
            raise FileNotFoundError(
                f"AG News train.csv/test.csv were not found under {root}"
            )

        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        train_text, train_labels = _read_agnews_csv(train_path)
        test_text, test_labels = _read_agnews_csv(test_path)
        train_text_set = set(train_text)
        keep_test = np.asarray(
            [text not in train_text_set for text in test_text], dtype=bool
        )
        test_text = [text for text, keep in zip(test_text, keep_test) if keep]
        test_labels = test_labels[keep_test]

        num_classes = int(train_labels.max()) + 1
        generator = np.random.default_rng(int(seed))
        val_indices = np.concatenate(
            [
                generator.choice(
                    np.flatnonzero(train_labels == class_index),
                    int(val_per_class),
                    replace=False,
                )
                for class_index in range(num_classes)
            ]
        )
        is_val = np.zeros(len(train_labels), dtype=bool)
        is_val[val_indices] = True
        train_fit_text = [text for index, text in enumerate(train_text) if not is_val[index]]
        val_text = [train_text[index] for index in val_indices]
        y_train = train_labels[~is_val]
        y_val = train_labels[val_indices]

        vectorizer = TfidfVectorizer(
            max_features=int(max_features), min_df=int(min_df), sublinear_tf=True
        )
        train_tfidf = vectorizer.fit_transform(train_fit_text)
        reducer = TruncatedSVD(n_components=int(svd_dim), random_state=int(seed))
        train_svd = reducer.fit_transform(train_tfidf)
        val_svd = reducer.transform(vectorizer.transform(val_text))
        test_svd = reducer.transform(vectorizer.transform(test_text))

        lower = np.quantile(train_svd, 0.01, axis=0)
        upper = np.quantile(train_svd, 0.99, axis=0)
        scale = np.maximum(upper - lower, 1e-6)

        def phase_scale(values: np.ndarray) -> np.ndarray:
            return np.clip((values - lower) / scale, 0.0, 1.0).astype(np.float32)

        arrays = {
            "train_indices": np.flatnonzero(~is_val),
            "validation_indices": val_indices,
            "tfidf_vocabulary": np.asarray(sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)),
            "tfidf_idf": vectorizer.idf_,
            "x_train": phase_scale(train_svd),
            "y_train": y_train.astype(np.int64),
            "x_val": phase_scale(val_svd),
            "y_val": y_val.astype(np.int64),
            "x_test": phase_scale(test_svd),
            "y_test": test_labels.astype(np.int64),
            "lower": lower.astype(np.float32),
            "scale": scale.astype(np.float32),
            "svd_components": reducer.components_.astype(np.float32),
            "svd_explained_variance_ratio": reducer.explained_variance_ratio_.astype(np.float32),
            "class_names": np.asarray(["World", "Sports", "Business", "Sci/Tech"]),
        }
        if cache is not None:
            flat_result_directory(cache.parent).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(flat_result_path(cache), **arrays)
            flat_result_path(metadata).write_text(json.dumps(specification, indent=2))

    class_names = np.asarray(arrays["class_names"]).astype(str)
    input_dim = int(arrays["x_train"].shape[1])
    side = int(round(input_dim**0.5))
    image_shape = (side, side) if side * side == input_dim else (input_dim,)
    return {
        "x_train": torch.from_numpy(arrays["x_train"]).float().to(device),
        "y_train": torch.from_numpy(arrays["y_train"]).long().to(device),
        "x_val": torch.from_numpy(arrays["x_val"]).float().to(device),
        "y_val": torch.from_numpy(arrays["y_val"]).long().to(device),
        "x_test": torch.from_numpy(arrays["x_test"]).float().to(device),
        "y_test": torch.from_numpy(arrays["y_test"]).long().to(device),
        "input_dim": input_dim,
        "num_classes": int(len(class_names)),
        "image_shape": image_shape,
        "class_names": class_names,
        "train_indices": arrays["train_indices"],
        "validation_indices": arrays["validation_indices"],
        "svd_lower": arrays["lower"],
        "svd_scale": arrays["scale"],
        "svd_components": arrays["svd_components"],
        "svd_explained_variance_ratio": arrays["svd_explained_variance_ratio"],
    }


def preprocess_cora(data_root, device="cpu", cache_path=None, rebuild=False,
                    seed=20260812, hops=5, input_dim=100):
    """Apply SGC on the unlabeled full graph, then training-fitted phase scaling.

    SVD is transductive and uses all unlabeled nodes. Labels are used only to
    reproduce the fixed stratified 60/20/20 split. The split indices and fitted
    SVD/scaling parameters are exported with the processed features.
    """
    from torch_geometric.io import read_planetoid_data
    from torch_geometric.utils import to_scipy_sparse_matrix
    root = Path(data_root)
    cache = None if cache_path is None else Path(cache_path)
    raw = root / "raw"
    specification = {"seed": int(seed), "hops": int(hops), "input_dim": int(input_dim)}
    if not any(raw.glob("ind.cora.*")):
        raise FileNotFoundError(f"Cora raw files are missing under {raw}")
    if cache is not None and flat_result_path(cache).exists() and not rebuild:
        with np.load(flat_result_path(cache), allow_pickle=False) as saved:
            cached_config = json.loads(str(saved["config_json"]))
            if any(cached_config.get(key) != value for key, value in specification.items()):
                raise ValueError("Cora cache does not match the requested preprocessing")
            arrays = {k: saved[k] for k in saved.files}
    else:
        graph = read_planetoid_data(str(raw), "Cora")
        labels = graph.y.cpu().numpy().astype(np.int64)
        indices = np.arange(graph.num_nodes)
        development, test = train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)
        train, validation = train_test_split(development, test_size=0.25,
                                             stratify=labels[development], random_state=seed+1)
        train, validation, test = map(np.sort, (train, validation, test))
        adjacency = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes).astype(np.float64)
        adjacency = adjacency + sp.eye(graph.num_nodes, dtype=np.float64)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse_sqrt = sp.diags(np.power(np.maximum(degree, 1e-12), -0.5))
        propagation = inverse_sqrt @ adjacency @ inverse_sqrt
        features = normalize(sp.csr_matrix(graph.x.cpu().numpy().astype(np.float64)), norm="l1", axis=1)
        for _ in range(int(hops)):
            features = propagation @ features
        reducer = TruncatedSVD(n_components=input_dim, n_iter=7, random_state=seed+100*hops+input_dim)
        projected = reducer.fit_transform(features.tocsr())
        lower = np.percentile(projected[train], 1.0, axis=0, keepdims=True)
        upper = np.percentile(projected[train], 99.0, axis=0, keepdims=True)
        scale = np.maximum(upper-lower, 1e-8)
        arrays = {"x": np.clip((projected-lower)/scale, 0, 1).astype(np.float32), "labels": labels,
                  "train_indices": train, "validation_indices": validation, "test_indices": test,
                  "lower": lower.squeeze(0).astype(np.float32), "scale": scale.squeeze(0).astype(np.float32),
                  "svd_components": reducer.components_.astype(np.float32),
                  "explained_variance": np.asarray(reducer.explained_variance_ratio_.sum()),
                  "hops": np.asarray(hops), "dimension": np.asarray(input_dim)}
        if cache is not None:
            save_npz(cache, arrays, specification)
    data = {"input_dim": int(input_dim), "num_classes": 7, "image_shape": (10, 10),
            "class_names": np.asarray(["Case_Based", "Genetic_Algorithms", "Neural_Networks",
                                        "Probabilistic_Methods", "Reinforcement_Learning", "Rule_Learning", "Theory"])}
    for split, key in [("train", "train_indices"), ("val", "validation_indices"), ("test", "test_indices")]:
        data["x_"+split] = torch.as_tensor(arrays["x"][arrays[key]], device=device)
        data["y_"+split] = torch.as_tensor(arrays["labels"][arrays[key]], device=device).long()
        data[key] = arrays[key]
    data["phase_lower"], data["phase_scale"] = arrays["lower"], arrays["scale"]
    return data


def preprocessing_table(data):
    """Report sample counts and class balance for each retained split."""
    import pandas as pd
    rows = []
    for split in ["train", "val", "test"]:
        counts = np.bincount(data["y_"+split].cpu().numpy(), minlength=data["num_classes"])
        rows.append({"Split": split, "Samples": int(counts.sum()),
                     "Input dimensions": data["input_dim"], "Samples per class": counts.tolist()})
    return pd.DataFrame(rows)


def _load_signal_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    signals = [
        np.loadtxt(root / split / "Inertial Signals" / f"{name}_{split}.txt")
        for name in HAR_CHANNELS
    ]
    x = np.stack(signals, axis=1).astype(np.float32)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    return x, y


def _choose_validation_subjects(
    labels: np.ndarray,
    subjects: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Choose a deterministic subject subset with size and class balance near target."""

    unique = np.unique(subjects)
    count = max(1, min(len(unique) - 1, int(round(len(unique) * float(fraction)))))
    global_hist = np.bincount(labels, minlength=len(HAR_CLASS_NAMES)).astype(float)
    global_hist /= global_hist.sum()
    target_size = len(labels) * float(fraction)
    best_score = math.inf
    best: tuple[int, ...] | None = None
    for candidate in itertools.combinations(unique.tolist(), count):
        mask = np.isin(subjects, candidate)
        hist = np.bincount(labels[mask], minlength=len(HAR_CLASS_NAMES)).astype(float)
        if np.any(hist == 0):
            continue
        hist /= hist.sum()
        size_error = abs(mask.sum() - target_size) / len(labels)
        balance_error = np.abs(hist - global_hist).mean()
        score = size_error + balance_error
        if score < best_score:
            best_score, best = score, tuple(int(v) for v in candidate)
    if best is None:
        raise RuntimeError("could not construct an all-class subject-level validation split")
    return np.asarray(best, dtype=np.int64)


def project_timeseries_inputs(
    x_latent: torch.Tensor, data: dict[str, Any]
) -> torch.Tensor:
    """Map native train-scaled windows to continuous optical phase masks."""

    if "projection_matrix" not in data:
        return x_latent
    centered = 2.0 * x_latent.float() - 1.0
    projected = centered @ data["projection_matrix"].T
    scaled = (projected - data["projection_lower"]) / data["projection_scale"]
    return scaled.clamp(0.0, 1.0)


def preprocess_uci_har(
    root: str | Path,
    *,
    device: str | torch.device,
    cache_path: str | Path | None = None,
    validation_fraction: float = 0.2,
    validation_subjects: Sequence[int] | None = None,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    projection_dim: int = 1024,
    projection_seed: int = 20260818,
) -> dict[str, Any]:
    """Load UCI-HAR with subject-disjoint train/validation/official-test splits.

    Both channel scaling and post-projection phase scaling are fit only on the
    final training subjects.  Native channel-major windows remain available as
    latent variables; their optical representation has ``projection_dim``
    continuous values in [0, 1].
    """

    root = Path(root)
    cache = None if cache_path is None else Path(cache_path)
    if cache is not None and flat_result_path(cache).exists():
        with np.load(flat_result_path(cache), allow_pickle=False) as archive:
            x_development = archive["Xtr"].astype(np.float32)
            y_development = archive["ytr"].astype(np.int64)
            x_test_raw = archive["Xte"].astype(np.float32)
            y_test = archive["yte"].astype(np.int64)
    else:
        x_development, y_development = _load_signal_split(root, "train")
        x_test_raw, y_test = _load_signal_split(root, "test")
    subject_development = np.loadtxt(
        root / "train" / "subject_train.txt", dtype=np.int64
    )
    subject_test = np.loadtxt(root / "test" / "subject_test.txt", dtype=np.int64)

    expected_development = (len(y_development), len(HAR_CHANNELS), 128)
    expected_test = (len(y_test), len(HAR_CHANNELS), 128)
    if x_development.shape != expected_development or x_test_raw.shape != expected_test:
        raise ValueError(
            f"unexpected HAR signal shapes {x_development.shape}, {x_test_raw.shape}"
        )
    if validation_subjects is None:
        selected_subjects = _choose_validation_subjects(
            y_development, subject_development, validation_fraction
        )
    else:
        selected_subjects = np.asarray(validation_subjects, dtype=np.int64)
    validation_mask = np.isin(subject_development, selected_subjects)
    training_mask = ~validation_mask
    if not training_mask.any() or not validation_mask.any():
        raise ValueError("training and validation subject sets must both be non-empty")
    if np.intersect1d(subject_development[training_mask], selected_subjects).size:
        raise AssertionError("subject leakage between training and validation")
    if np.intersect1d(subject_development, subject_test).size:
        raise AssertionError("subject leakage into the official test split")

    x_train_raw = x_development[training_mask]
    x_val_raw = x_development[validation_mask]
    y_train = y_development[training_mask]
    y_val = y_development[validation_mask]
    lower = np.quantile(
        x_train_raw, float(lower_quantile), axis=(0, 2), keepdims=True
    ).astype(np.float32)
    upper = np.quantile(
        x_train_raw, float(upper_quantile), axis=(0, 2), keepdims=True
    ).astype(np.float32)
    scale = np.maximum(upper - lower, np.float32(1e-6))

    def transform(values: np.ndarray) -> np.ndarray:
        scaled = np.clip((values - lower) / scale, 0.0, 1.0).astype(np.float32)
        return scaled.reshape(len(scaled), -1)

    x_train_latent = transform(x_train_raw)
    x_val_latent = transform(x_val_raw)
    x_test_latent = transform(x_test_raw)

    native_input_dim = int(x_train_latent.shape[1])
    projection_dim = int(projection_dim)
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    generator = np.random.default_rng(int(projection_seed))
    projection_matrix = generator.normal(
        0.0,
        1.0 / math.sqrt(native_input_dim),
        size=(projection_dim, native_input_dim),
    ).astype(np.float32)

    def linear_project(values: np.ndarray) -> np.ndarray:
        return (2.0 * values - 1.0) @ projection_matrix.T

    projected_train = linear_project(x_train_latent)
    projected_lower = np.quantile(
        projected_train, float(lower_quantile), axis=0
    ).astype(np.float32)
    projected_upper = np.quantile(
        projected_train, float(upper_quantile), axis=0
    ).astype(np.float32)
    projected_scale = np.maximum(
        projected_upper - projected_lower, np.float32(1e-6)
    )

    def phase_project(values: np.ndarray) -> np.ndarray:
        projected = linear_project(values)
        return np.clip(
            (projected - projected_lower) / projected_scale, 0.0, 1.0
        ).astype(np.float32)

    x_train = phase_project(x_train_latent)
    x_val = phase_project(x_val_latent)
    x_test = phase_project(x_test_latent)
    for split_name, labels in (("train", y_train), ("val", y_val), ("test", y_test)):
        if set(np.unique(labels).tolist()) != set(range(len(HAR_CLASS_NAMES))):
            raise ValueError(f"{split_name} does not contain every HAR class")

    to_float = lambda value: torch.from_numpy(value).float().to(device)
    to_long = lambda value: torch.from_numpy(value).long().to(device)
    projection_side = math.isqrt(projection_dim)
    input_shape = (
        (projection_side, projection_side)
        if projection_side * projection_side == projection_dim
        else (projection_dim,)
    )
    return {
        "x_train": to_float(x_train),
        "x_train_latent": to_float(x_train_latent),
        "y_train": to_long(y_train),
        "x_val": to_float(x_val),
        "x_val_latent": to_float(x_val_latent),
        "y_val": to_long(y_val),
        "x_test": to_float(x_test),
        "x_test_latent": to_float(x_test_latent),
        "y_test": to_long(y_test),
        "input_dim": int(x_train.shape[1]),
        "input_shape": input_shape,
        "native_input_dim": native_input_dim,
        "native_input_shape": (len(HAR_CHANNELS), 128),
        "num_classes": len(HAR_CLASS_NAMES),
        "class_names": np.asarray(HAR_CLASS_NAMES),
        "channel_names": np.asarray(HAR_CHANNELS),
        "train_subjects": np.unique(subject_development[training_mask]),
        "validation_subjects": np.unique(subject_development[validation_mask]),
        "test_subjects": np.unique(subject_test),
        "train_development_indices": np.flatnonzero(training_mask),
        "validation_development_indices": np.flatnonzero(validation_mask),
        "scale_lower": lower.reshape(-1),
        "scale_upper": upper.reshape(-1),
        "scale_range": scale.reshape(-1),
        "projection_matrix": to_float(projection_matrix),
        "projection_lower": to_float(projected_lower),
        "projection_upper": to_float(projected_upper),
        "projection_scale": to_float(projected_scale),
        "projection_seed": int(projection_seed),
        "projection_dim": projection_dim,
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
    }


# MZI mesh: fixed, training-only amplitude encoding.
class PCAEncoder:
    """Fixed PCA64 encoding fitted on training pixels only.

    u = (x - mean) @ P.T * s; field = normalize(u + off), where
    off = mean @ P.T * s. All modes carry signed real amplitudes.
    The per-component scaling is std**(-power), followed by a global scale.
    Synthetic inputs are optimized in u space, without a dedicated bias mode.
    """
    def __init__(self, x_train, modes=64, power=.25):
        self.mean = x_train.mean(0)
        centered = x_train - self.mean
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
        order = torch.argsort(eigenvalues, descending=True)[:modes]
        self.P = eigenvectors[:, order].T.float().contiguous()
        self.std = eigenvalues[order].clamp_min(1e-12).sqrt().float()
        scale = self.std ** (-power)
        coefficients = (centered @ self.P.T) * scale
        self.s = scale / coefficients.norm(dim=1).mean()
        self.off = (self.mean @ self.P.T) * self.s
        self.k, self.bias, self.inp = modes, 0., 'uncent'

    def u(self, images):
        return ((images - self.mean) @ self.P.T) * self.s

    def field(self, coefficients):
        return torch.nn.functional.normalize(coefficients + self.off, dim=-1)

    def encode(self, images):
        return self.field(self.u(images))

    @classmethod
    def from_state(cls, state, device):
        encoder = cls.__new__(cls)
        for key, value in state.items():
            setattr(encoder, key, value.to(device) if torch.is_tensor(value) else value)
        if encoder.inp != 'uncent':
            raise ValueError('This publication experiment uses the uncentered PCA field encoder.')
        return encoder


def split_fmnist_for_mzi(data, seed=20260926, validation_per_class=500):
    """Class-wise 55k/5k split with the paper's CPU-generator ordering.

    The input is the dictionary returned by datasets.load_fmnist. PCA must be
    fitted after this split. Sampling on CPU preserves the archived indices.
    """
    x, y = data['x_train'], data['y_train']
    generator = torch.Generator().manual_seed(int(seed))
    train, validation = [], []
    for label in range(10):
        indices = torch.where(y.cpu() == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        validation.append(indices[:validation_per_class])
        train.append(indices[validation_per_class:])
    train, validation = torch.cat(train).to(x.device), torch.cat(validation).to(x.device)
    return (x[train], y[train]), (x[validation], y[validation]), (data['x_test'], data['y_test'])
