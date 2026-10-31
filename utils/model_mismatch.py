"""Optical errors for evaluating frozen synthetic datasets.

Figure 8 perturbation strengths are set explicitly in SI.ipynb.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MISMATCH_NAMES = (
    "zernike_wavefront_aberration",
    "spatial_nonlinear_phase_response",
    "pixel_crosstalk",
    "mode_amplitude_nonuniformity",
)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    values = values - values.mean()
    return values / values.square().mean().sqrt().clamp_min(1e-08)


def smooth_error_map(seed: int, device: torch.device) -> torch.Tensor:
    """Make a low-spatial-frequency 28x28 calibration map with unit RMS."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    coarse = torch.randn(1, 1, 7, 7, generator=generator, device=device)
    smooth = F.interpolate(coarse, size=(28, 28), mode="bicubic", align_corners=False)[
        0, 0
    ]
    return _standardize(smooth).reshape(-1)


def mixed_pixel_error_map(
    seed: int, device: torch.device, pixel_fraction: float = 0.3
) -> torch.Tensor:
    """Unit-RMS map containing both smooth calibration drift and pixel residuals."""
    smooth = smooth_error_map(seed, device).reshape(28, 28)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1000000)
    pixel = _standardize(torch.randn(28, 28, generator=generator, device=device))
    mixed = np.sqrt(1.0 - pixel_fraction**2) * smooth + pixel_fraction * pixel
    return _standardize(mixed).reshape(-1)


def zernike_error_map(repetition: int, device: torch.device) -> torch.Tensor:
    """Unit-RMS mixture of Noll Z4, Z5, Z6, Z7, Z8, and Z11.

    The square 28x28 controlled region is inscribed in the unit disk, so every
    controlled mode receives a well-defined phase and no extra pupil mask is
    introduced into the ideal model.
    """
    coordinate = torch.linspace(
        -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 28, device=device
    )
    (yy, xx) = torch.meshgrid(coordinate, coordinate, indexing="ij")
    rho = torch.sqrt(xx.square() + yy.square())
    theta = torch.atan2(yy, xx)
    radial_coma = 3.0 * rho.pow(3) - 2.0 * rho
    modes = torch.stack(
        [
            np.sqrt(3.0) * (2.0 * rho.square() - 1.0),
            np.sqrt(6.0) * rho.square() * torch.sin(2.0 * theta),
            np.sqrt(6.0) * rho.square() * torch.cos(2.0 * theta),
            np.sqrt(8.0) * radial_coma * torch.sin(theta),
            np.sqrt(8.0) * radial_coma * torch.cos(theta),
            np.sqrt(5.0) * (6.0 * rho.pow(4) - 6.0 * rho.square() + 1.0),
        ]
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(610000 + int(repetition))
    coefficients = torch.randn(len(modes), generator=generator, device=device)
    mixture = torch.einsum("m,mhw->hw", coefficients, modes)
    return _standardize(mixture).reshape(-1)


def amplitude_gain_for_target_cv(
    base_map: torch.Tensor, target_cv: float
) -> torch.Tensor:
    """Positive mean-one field gain whose spatial CV matches target_cv."""
    if target_cv <= 0:
        return torch.ones_like(base_map)
    (lower, upper) = (0.0, 2.0)
    for _ in range(32):
        tau = 0.5 * (lower + upper)
        gain = torch.exp(tau * base_map)
        gain = gain / gain.mean()
        cv = gain.std(unbiased=False) / gain.mean()
        if float(cv) < float(target_cv):
            lower = tau
        else:
            upper = tau
    gain = torch.exp(0.5 * (lower + upper) * base_map)
    return gain / gain.mean()


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, 1, 1, 1, device=device)
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    one_dimensional = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = one_dimensional[:, None] * one_dimensional[None, :]
    return kernel[None, None]


def optical_error_maps(
    repetition: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "zernike": zernike_error_map(repetition, device),
        "phase_gain": mixed_pixel_error_map(620000 + repetition, device),
        "phase_nonlinearity": mixed_pixel_error_map(630000 + repetition, device),
        "amplitude": mixed_pixel_error_map(640000 + repetition, device),
    }


@torch.inference_mode()
def mismatch_intensity(
    x_numpy: np.ndarray,
    tm: torch.Tensor,
    maps: dict[str, torch.Tensor],
    mismatch_index: int,
    strength: float,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    pi = float(np.pi)
    crosstalk_kernel = (
        gaussian_kernel(strength, device) if mismatch_index == 2 else None
    )
    crosstalk_radius = (
        (crosstalk_kernel.shape[-1] - 1) // 2 if crosstalk_kernel is not None else 0
    )
    amplitude_gain = (
        amplitude_gain_for_target_cv(maps["amplitude"], strength)
        if mismatch_index == 3
        else None
    )
    for start in range(0, len(x_numpy), batch_size):
        x = torch.as_tensor(
            x_numpy[start : start + batch_size], dtype=torch.float32, device=device
        )
        if mismatch_index == 0:
            phase = pi * x + float(strength) * maps["zernike"]
            field = torch.exp(1j * phase)
        elif mismatch_index == 1:
            phase = pi * (
                (1.0 + float(strength) * maps["phase_gain"]) * x
                + float(strength) * maps["phase_nonlinearity"] * x * (1.0 - x)
            )
            field = torch.exp(1j * phase)
        elif mismatch_index == 2:
            phase_image = (pi * x).reshape(-1, 1, 28, 28)
            if crosstalk_radius > 0:
                phase_image = F.pad(
                    phase_image, (crosstalk_radius,) * 4, mode="reflect"
                )
                phase_image = F.conv2d(phase_image, crosstalk_kernel)
            field = torch.exp(1j * phase_image.reshape(-1, 784))
        elif mismatch_index == 3:
            field = amplitude_gain * torch.exp(1j * pi * x)
        else:
            raise ValueError(f"Unknown mismatch index: {mismatch_index}")
        outputs.append((field @ tm).abs().square().float())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def feature_deformation(
    ideal: torch.Tensor, perturbed: torch.Tensor
) -> tuple[float, float]:
    ideal_centered = ideal - ideal.mean(dim=1, keepdim=True)
    perturbed_centered = perturbed - perturbed.mean(dim=1, keepdim=True)
    correlation = (ideal_centered * perturbed_centered).sum(dim=1) / (
        ideal_centered.square().sum(dim=1).sqrt()
        * perturbed_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)
    best_scale = (ideal * perturbed).sum(dim=1, keepdim=True) / perturbed.square().sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    aligned = best_scale * perturbed
    relative_l2 = (aligned - ideal).square().sum(dim=1).sqrt() / ideal.square().sum(
        dim=1
    ).sqrt().clamp_min(1e-12)
    return (float(correlation.mean()), float(relative_l2.mean()))
