"""Recalculate optical characterization from measured transmission data."""


import numpy as np
from scipy.io import loadmat
from scipy.stats import norm


def pearson_rows(predicted, measured):
    """Pearson correlation across output channels, independently for each input."""
    predicted = np.asarray(predicted, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if predicted.shape != measured.shape or predicted.ndim != 2:
        raise ValueError("Expected matching input-by-channel intensity arrays.")
    predicted = predicted - predicted.mean(axis=1, keepdims=True)
    measured = measured - measured.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(measured, axis=1)
    if np.any(denominator == 0):
        raise ValueError("Pearson correlation is undefined for constant speckles.")
    return np.sum(predicted * measured, axis=1) / denominator


def analyze_tm_distribution(tm_file, bins=55):
    """Fit real and imaginary Gaussian marginals to all measured TM elements."""
    data = loadmat(tm_file, variable_names=["T", "strict_no_b"])
    tm = np.asarray(data["T"])
    if tm.ndim != 2 or not np.iscomplexobj(tm):
        raise ValueError("The measured TM must be a complex matrix.")
    real = tm.real.ravel().astype(np.float64)
    imag = tm.imag.ravel().astype(np.float64)
    limit = float(np.quantile(np.abs(np.concatenate([real, imag])), 0.999))
    edges = np.linspace(-limit, limit, bins + 1)
    result = {"tm_shape": np.asarray(tm.shape), "bin_edges": edges}
    for name, values in [("real", real), ("imag", imag)]:
        mu, sigma = norm.fit(values)
        result[f"{name}_counts"] = np.histogram(values, bins=edges)[0]
        result[f"{name}_mu"] = float(mu)
        result[f"{name}_sigma"] = float(sigma)
        result[f"{name}_sample_count"] = values.size
    return result


