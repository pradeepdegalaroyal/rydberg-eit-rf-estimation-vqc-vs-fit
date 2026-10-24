"""Four-level Rydberg-EIT/Autler-Townes model with Doppler averaging (steady-state Lindblad).

Levels: |g>, |e>, |r>, |r'>. Probe couples g-e (Rabi Omega_p), coupling laser e-r (Omega_c),
RF field r-r' (Omega_rf, resonant). All rates in MHz (angular units, hbar = 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N = 4


@dataclass(frozen=True)
class Params:
    """Nominal experiment parameters (MHz)."""

    omega_p: float = 1.0
    omega_c: float = 8.0
    gamma_e: float = 6.0  # decay e -> g
    gamma_r: float = 0.3  # effective decay r, r' -> e
    doppler_sigma: float = 6.0  # Doppler width seen by the probe
    k_ratio: float = -0.6  # coupling/probe Doppler shift ratio (counter-propagating, wavelength mismatch)
    n_vel: int = 9
    det_max: float = 20.0
    n_det: int = 32


def _liouvillian(h: np.ndarray, gam_e: float, gam_r: float) -> np.ndarray:
    """Liouvillian (row-major vec) for a stack of Hamiltonians h[..., 4, 4]."""
    eye = np.eye(N)
    lead = h.shape[:-2]
    comm = np.einsum("...ij,kl->...ikjl", h, eye) - np.einsum("ij,...lk->...ikjl", eye, h)
    L = -1j * comm.reshape(*lead, N * N, N * N)
    for to, frm, g in [(0, 1, gam_e), (1, 2, gam_r), (1, 3, gam_r)]:
        c = np.zeros((N, N))
        c[to, frm] = 1.0
        cdc = c.T @ c
        L = L + g * (np.kron(c, c) - 0.5 * np.kron(cdc, eye) - 0.5 * np.kron(eye, cdc.T))
    return L


def absorption(omega_rf: np.ndarray, omega_c: np.ndarray | float, p: Params = Params()) -> np.ndarray:
    """Doppler-averaged probe absorption Im(rho_ge) vs probe detuning; returns shape (B, n_det)."""
    omega_rf = np.atleast_1d(omega_rf).astype(float)
    omega_c = np.broadcast_to(np.atleast_1d(omega_c), omega_rf.shape).astype(float)
    B = omega_rf.size
    dets = np.linspace(-p.det_max, p.det_max, p.n_det)
    v = np.linspace(-2.5, 2.5, p.n_vel) * p.doppler_sigma
    w = np.exp(-((v / p.doppler_sigma) ** 2))
    w /= w.sum()
    dp = np.broadcast_to(dets[None, :, None] + v[None, None, :], (B, p.n_det, p.n_vel))
    dc = np.broadcast_to(p.k_ratio * v[None, None, :], dp.shape)
    H = np.zeros(dp.shape + (N, N), dtype=complex)
    H[..., 0, 1] = H[..., 1, 0] = p.omega_p / 2
    H[..., 1, 2] = H[..., 2, 1] = omega_c[:, None, None] / 2
    H[..., 2, 3] = H[..., 3, 2] = omega_rf[:, None, None] / 2
    H[..., 1, 1] = -dp
    H[..., 2, 2] = -(dp + dc)
    H[..., 3, 3] = -(dp + dc)
    A = _liouvillian(H, p.gamma_e, p.gamma_r)
    A[..., 0, :] = 0.0  # replace first equation by Tr(rho) = 1
    A[..., 0, [0, 5, 10, 15]] = 1.0
    rhs = np.zeros(dp.shape + (N * N, 1), dtype=complex)
    rhs[..., 0, 0] = 1.0
    rho = np.linalg.solve(A, rhs)[..., 0]
    return (rho[..., 1].imag * w[None, None, :]).sum(-1)  # rho_{ge}


# Fixed normalisation so that clean nominal spectra have peak ~1.
SCALE = float(absorption(np.array([6.0]), 8.0).max())


def crb_std(omega_rf: float, sigma: float, p: Params = Params(), h: float = 0.05) -> float:
    """Cramer-Rao std of omega_rf for iid Gaussian noise `sigma` on the normalised nominal spectrum."""
    s = absorption(np.array([omega_rf + h, omega_rf - h]), 8.0, p)
    d = (s[0] - s[1]) / (2 * h) / SCALE
    return float(sigma / np.sqrt((d**2).sum()))
