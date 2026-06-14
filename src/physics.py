from __future__ import annotations

import torch


CU_K_ALPHA_WAVELENGTH = 1.5406


def bragg_2theta_from_lattice(
    lattice_params: torch.Tensor,
    peak_hkls: torch.Tensor,
    wavelength: float = CU_K_ALPHA_WAVELENGTH,
) -> torch.Tensor:
    """Compute 2theta positions from lattice parameters and hkl indices."""
    eps = 1e-6
    a = torch.clamp(lattice_params[:, 0], min=eps)
    b = torch.clamp(lattice_params[:, 1], min=eps)
    c = torch.clamp(lattice_params[:, 2], min=eps)
    alpha = torch.deg2rad(lattice_params[:, 3])
    beta = torch.deg2rad(lattice_params[:, 4])
    gamma = torch.deg2rad(lattice_params[:, 5])

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)

    batch_size = lattice_params.shape[0]
    metric = torch.zeros((batch_size, 3, 3), dtype=lattice_params.dtype, device=lattice_params.device)
    metric[:, 0, 0] = a * a
    metric[:, 1, 1] = b * b
    metric[:, 2, 2] = c * c
    metric[:, 0, 1] = metric[:, 1, 0] = a * b * cos_gamma
    metric[:, 0, 2] = metric[:, 2, 0] = a * c * cos_beta
    metric[:, 1, 2] = metric[:, 2, 1] = b * c * cos_alpha

    identity = torch.eye(3, dtype=lattice_params.dtype, device=lattice_params.device).unsqueeze(0)
    reciprocal_metric = torch.linalg.inv(metric + identity * 1e-5)

    hkls = peak_hkls.to(dtype=lattice_params.dtype)
    inv_d2 = torch.einsum("bki,bij,bkj->bk", hkls, reciprocal_metric, hkls)
    inv_d2 = torch.clamp(inv_d2, min=eps)
    d_hkl = torch.rsqrt(inv_d2)

    sin_theta = torch.clamp(wavelength / (2.0 * d_hkl), min=-0.999999, max=0.999999)
    return 2.0 * torch.rad2deg(torch.asin(sin_theta))


def masked_peak_mae(
    lattice_params: torch.Tensor,
    peak_hkls: torch.Tensor,
    peak_2theta: torch.Tensor,
    peak_mask: torch.Tensor,
) -> torch.Tensor:
    calc_2theta = bragg_2theta_from_lattice(lattice_params, peak_hkls)
    abs_error = torch.abs(calc_2theta - peak_2theta) * peak_mask
    return abs_error.sum() / torch.clamp(peak_mask.sum(), min=1.0)


def masked_peak_mse(
    lattice_params: torch.Tensor,
    peak_hkls: torch.Tensor,
    peak_2theta: torch.Tensor,
    peak_mask: torch.Tensor,
) -> torch.Tensor:
    calc_2theta = bragg_2theta_from_lattice(lattice_params, peak_hkls)
    squared_error = (calc_2theta - peak_2theta).pow(2) * peak_mask
    return squared_error.sum() / torch.clamp(peak_mask.sum(), min=1.0)
