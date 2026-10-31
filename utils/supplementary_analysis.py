"""Numerical summaries and measured-TM readouts for supplementary figures."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.linalg import solve

from .datasets import load_npz
from .experimental_analysis import (
    _find,
    _indices,
    _load_features,
    _load_source,
    _normalise,
    _ridge_accuracy,
)


def analyze_tm_specific_controls(root, channel_seed=17):
    """Recompute the recorded ensemble/random controls from 18 camera readouts.

    The original figure pooled two acquisitions of each of nine scattering
    states. Its SD is over the 18 readouts, without averaging acquisition pairs.
    The two 2025-channel recordings retain the shared 1024-channel subset.
    Support and test blocks retain their original separate maximum scales.
    """
    root = Path(root)
    channels = np.sort(
        np.random.default_rng(channel_seed).choice(2025, 1024, replace=False)
    )
    ensemble, random, files, states, acquisitions = [], [], [], [], []
    for acquisition in [1, 2]:
        for rep in range(1, 10):
            suffix = "_exp2" if acquisition == 2 else ""
            path = root / f"rep{rep}{suffix}.mat"
            subset = channels if acquisition == 2 and rep in [5, 8] else None
            output = _load_features(path, subset)
            soft, hard, rows, positions = _load_source(
                root / "sim_source_data" / f"rep{rep}.mat"
            )
            test_index = _indices(_find(rows, block_type="test"))
            query, truth = _normalise(output[test_index]), hard[test_index]
            support = _find(
                [row for row in rows if int(row["ipc"]) == 1], variant="x_plus_y"
            )
            index = _indices(support)
            score, _ = _ridge_accuracy(
                output[index], soft[index], query, truth, support["ridge_beta"]
            )
            ensemble.append(score)
            random_start = int(_find(rows, variant="random")["start_idx"]) - 1
            index = random_start + positions[1] - 1
            score, _ = _ridge_accuracy(output[index], soft[index], query, truth, 1.0)
            random.append(score)
            files.append(path.name)
            states.append(rep)
            acquisitions.append(acquisition)
    result = {
        "source_files": np.asarray(files),
        "source_reps": np.asarray(states),
        "acquisition_ids": np.asarray(acquisitions),
        "channel_seed": channel_seed,
    }
    for name, values in [("tm_ensemble", ensemble), ("random", random)]:
        values = np.asarray(values, dtype=np.float64)
        result[name] = {
            "accuracy": values,
            "mean_percent": 100 * values.mean(),
            "sd_percent": 100 * values.std(ddof=1),
        }
    return result


def analyze_tm_specific_experiment(path, include_simulation=True):
    """Recompute the five-seed strict no-bias experiment from camera intensities.

    The fixed feature scale and ridge lambda come from the acquisition record.
    All seeds share one physical acquisition and the same 10,000 test speckles.
    This variation measures optimization-seed sensitivity within that acquisition.
    Set include_simulation=False when the simulation is trained again separately.
    """
    fields = [
        "speckle_raw",
        "support_rows",
        "support_targets",
        "test_rows",
        "test_labels",
        "ridge_beta",
        "feature_scale",
        "training_seed",
        "experimental_accuracy_saved",
        "predictions_saved",
        "bias_used_in_training",
        "bias_used_in_deployment",
    ]
    if include_simulation:
        fields += [
            "simulation_accuracy",
            "snapshot_steps",
            "simulation_accuracy_history",
        ]
    raw = loadmat(Path(path), variable_names=fields, squeeze_me=True)
    if bool(raw["bias_used_in_training"]) or bool(raw["bias_used_in_deployment"]):
        raise ValueError("The archived experiment requires the strict no-bias model")
    features = np.asarray(raw["speckle_raw"], dtype=np.float64) / float(
        raw["feature_scale"]
    )
    support_rows = np.asarray(raw["support_rows"], dtype=np.int64) - 1
    test_rows = np.asarray(raw["test_rows"], dtype=np.int64).reshape(-1) - 1
    truth = np.asarray(raw["test_labels"], dtype=np.int64).reshape(-1)
    targets = np.asarray(raw["support_targets"], dtype=np.float64)
    predictions, accuracy = [], []
    for rows, labels in zip(support_rows, targets):
        support = features[rows]
        coefficients = solve(
            support @ support.T + float(raw["ridge_beta"]) * np.eye(len(rows)),
            labels,
            assume_a="pos",
            check_finite=False,
        )
        prediction = (features[test_rows] @ support.T @ coefficients).argmax(axis=1)
        predictions.append(prediction)
        accuracy.append(np.mean(prediction == truth))
    predictions, accuracy = np.asarray(predictions), np.asarray(accuracy)
    result = {
        "training_seeds": np.asarray(raw["training_seed"], dtype=np.int64),
        "experimental_accuracy": accuracy,
        "predictions": predictions,
        "ridge_lambda": np.asarray(raw["ridge_beta"]),
        "feature_scale": np.asarray(raw["feature_scale"]),
        "shared_physical_acquisition": np.asarray(True),
    }
    if include_simulation:
        trajectory = np.asarray(raw["simulation_accuracy_history"], dtype=np.float64)
        result.update(
            snapshot_steps=np.asarray(raw["snapshot_steps"], dtype=np.int64),
            simulation_accuracy_history=trajectory,
            simulation_accuracy=np.asarray(
                raw["simulation_accuracy"], dtype=np.float64
            ),
        )
    return result


def training_tm_summary(path):
    """Average the 100 unseen TMs within each repetition before computing CIs."""
    result = load_npz(path)
    accuracy = np.asarray(result["accuracy_per_tm"], dtype=np.float64)
    counts = result["train_tm_counts"].astype(int)
    if (
        accuracy.ndim != 3
        or accuracy.shape[0] != len(counts)
        or accuracy.shape[1] < 2
        or accuracy.shape[2] != 100
        or np.any(counts < 1)
        or len(np.unique(counts)) != len(counts)
        or not np.isfinite(accuracy).all()
    ):
        raise ValueError("Incomplete training-TM sweep")
    for training, unseen in zip(result["train_tm_seeds"], result["unseen_tm_seeds"]):
        if np.intersect1d(training, unseen).size:
            raise ValueError("Training and evaluation TM seeds overlap")
    means = accuracy.mean(axis=2)
    std = means.std(axis=1, ddof=1)
    return {
        "train_tm_counts": counts,
        "accuracy_per_repetition": means,
        "mean": means.mean(axis=1),
        "std": std,
        "ci95": 1.96 * std / np.sqrt(means.shape[1]),
    }
