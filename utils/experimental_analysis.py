"""MATLAB recording readers, ridge algebra and shared result summaries."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.linalg import solve

FMNIST_IPC = np.asarray([1, 2, 5, 10, 20, 50, 100], dtype=np.int32)
FMNIST_RANDOM_EXTRA_IPC = np.asarray([600, 700], dtype=np.int32)
FMNIST_CLASSES = np.asarray([
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
])


def _matlab_text(array) -> str:
    return "".join(chr(int(value)) for value in np.asarray(array).reshape(-1, order="F"))


def _matlab_value(handle: h5py.File, reference):
    dataset = handle[reference]
    if dataset.attrs.get("MATLAB_class") == b"char":
        return _matlab_text(dataset)
    values = np.asarray(dataset).reshape(-1, order="F")
    if len(values) == 1:
        return values[0].item()
    return values


def _manifest(handle: h5py.File) -> list[dict]:
    columns = {}
    for name, dataset in handle["block_manifest"].items():
        values = np.asarray(dataset).reshape(-1, order="F")
        if dataset.dtype.kind == "O":
            columns[name] = [_matlab_value(handle, ref) if ref else "" for ref in values]
        else:
            columns[name] = values.tolist()
    count = len(next(iter(columns.values())))
    return [
        {name: values[index] for name, values in columns.items()}
        for index in range(count)
    ]


def _load_source(path: Path):
    with h5py.File(path, "r") as handle:
        soft = np.asarray(handle["all_label_soft"], dtype=np.float64).T
        hard = np.asarray(handle["all_label_hard"], dtype=np.int64).reshape(-1, order="F")
        rows = _manifest(handle)
        mapping = handle["random_subset_mapping"]
        ipcs = np.asarray(mapping["ipcs"], dtype=np.int32).reshape(-1, order="F")
        refs = np.asarray(mapping["positions_in_random100"]).reshape(-1, order="F")
        positions = {
            int(ipc): np.asarray(handle[ref], dtype=np.int64).reshape(-1, order="F")
            for ipc, ref in zip(ipcs, refs)
        }
    return soft, hard, rows, positions


def _load_features(path: Path, channel_indices=None) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        output = np.asarray(handle["output"], dtype=np.float64).transpose(2, 1, 0)
    output = output.reshape(len(output), -1)
    return output if channel_indices is None else output[:, channel_indices]


def _find(rows, *, variant=None, block_type=None) -> dict:
    matches = [
        row for row in rows
        if (variant is None or str(row["variant"]) == variant)
        and (block_type is None or str(row["block_type"]) == block_type)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one block, found {len(matches)}: {variant=}, {block_type=}")
    return matches[0]


def _indices(row) -> np.ndarray:
    return np.arange(int(row["start_idx"]) - 1, int(row["end_idx"]), dtype=np.int64)


def _normalise(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(values))
    if scale <= 0:
        raise ValueError("Feature maximum must be positive")
    return values / scale


def _ridge_accuracy(train, labels, test, truth, beta, *, confusion=False):
    train = _normalise(train)
    beta = float(beta)
    if len(train) <= train.shape[1]:
        dual = solve(
            train @ train.T + beta * np.eye(len(train)), labels,
            assume_a="pos", check_finite=False,
        )
        scores = (test @ train.T) @ dual
    else:
        weights = solve(
            train.T @ train + beta * np.eye(train.shape[1]), train.T @ labels,
            assume_a="pos", check_finite=False,
        )
        scores = test @ weights
    prediction = np.argmax(scores, axis=1)
    accuracy = float(np.mean(prediction == truth))
    counts = None
    if confusion:
        counts = np.zeros((10, 10), dtype=np.int64)
        np.add.at(counts, (truth, prediction), 1)
    return accuracy, counts


def _summary(result: dict, name: str, values: np.ndarray) -> None:
    result[f"{name}_accuracy"] = values
    result[f"{name}_mean"] = values.mean(axis=0)
    result[f"{name}_std"] = values.std(axis=0, ddof=1)
    result[f"{name}_min"] = values.min(axis=0)
    result[f"{name}_max"] = values.max(axis=0)


def measurement_summary(sizes, coreset, random, *, metric='accuracy'):
    """Compact mean ± sample SD across repetition/acquisition rows."""
    if metric not in {'accuracy', 'r2'}:
        raise ValueError("metric must be 'accuracy' or 'r2'")
    scale, digits = (100.0, 2) if metric == 'accuracy' else (1.0, 4)
    rows = {'Samples per class' if metric == 'accuracy' else 'Training samples': sizes}
    for label, values in [('Coreset', coreset), ('Random', random)]:
        values = scale * np.asarray(values)
        if values.ndim != 2 or values.shape[1] != len(sizes) or values.shape[0] < 2:
            raise ValueError('Expected repetitions × support sizes with at least two repetitions')
        rows[label + (' accuracy (%)' if metric == 'accuracy' else ' R²')] = [
            f'{mean:.{digits}f} ± {std:.{digits}f}'
            for mean, std in zip(values.mean(axis=0), values.std(axis=0, ddof=1))
        ]
    return pd.DataFrame(rows)


def _ridge_output(train, targets, test, beta):
    beta = float(beta)
    if len(train) <= train.shape[1]:
        dual = solve(
            train @ train.T + beta * np.eye(len(train)), targets,
            assume_a="pos", check_finite=False,
        )
        return (test @ train.T) @ dual
    weights = solve(
        train.T @ train + beta * np.eye(train.shape[1]), train.T @ targets,
        assume_a="pos", check_finite=False,
    )
    return test @ weights


def _scalar(handle, name):
    return np.asarray(handle[name]).reshape(-1, order="F")[0].item()


def sample_efficiency_table(result):
    """Summarize arithmetic means and sample SDs across the nine measured states."""
    ipc = result['coreset_ipc'][:4]
    matched = result['matched_random_ipc'][:, :4]
    return pd.DataFrame({'Coreset IPC': ipc,
                         'Matched random IPC (mean)': matched.mean(axis=0),
                         'Matched random IPC (s.d.)': matched.std(axis=0, ddof=1),
                         'Sample-efficiency gain (mean)': (matched / ipc).mean(axis=0),
                         'Sample-efficiency gain (s.d.)': (matched / ipc).std(axis=0, ddof=1)})
