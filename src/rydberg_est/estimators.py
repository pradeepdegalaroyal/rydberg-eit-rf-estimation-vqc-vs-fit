"""Estimators of the RF Rabi frequency: model-based fit, ridge, MLP, and a variational quantum circuit."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .physics import SCALE, Params, absorption

OMEGA_LO, OMEGA_HI = 3.0, 15.0


def make_data(n: int, rng: np.random.Generator, drift: float, p: Params = Params()) -> tuple[np.ndarray, np.ndarray]:
    """Clean normalised spectra and labels; `drift` = relative std of the unmodelled coupling Rabi frequency."""
    y = rng.uniform(OMEGA_LO, OMEGA_HI, n)
    oc = 8.0 * (1 + drift * rng.standard_normal(n))
    return absorption(y, oc, p) / SCALE, y


def add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian detection noise (std `sigma` in units of the nominal peak signal)."""
    return x + sigma * rng.standard_normal(x.shape)


class ModelFit:
    """Maximum-likelihood fit with the nominal physical model (grid search + parabolic refinement)."""

    def __init__(self, p: Params = Params(), n_grid: int = 241):
        self.grid = np.linspace(OMEGA_LO - 0.5, OMEGA_HI + 0.5, n_grid)
        self.lib = absorption(self.grid, 8.0, p) / SCALE

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = ((x[:, None, :] - self.lib[None]) ** 2).sum(-1)
        i = np.clip(d.argmin(1), 1, len(self.grid) - 2)
        r = np.arange(len(x))
        a, b, c = d[r, i - 1], d[r, i], d[r, i + 1]
        den = a - 2 * b + c
        ok = den > 1e-12
        off = np.where(ok, 0.5 * (a - c) / np.where(ok, den, 1.0), 0.0)
        return self.grid[i] + np.clip(off, -1, 1) * (self.grid[1] - self.grid[0])


class PCA:
    """Whitened PCA features."""

    def fit(self, x: np.ndarray, k: int) -> "PCA":
        self.mu = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.mu, full_matrices=False)
        self.w = vt[:k].T
        self.sd = ((x - self.mu) @ self.w).std(0) + 1e-9
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) @ self.w / self.sd


def _norm_y(y: np.ndarray) -> np.ndarray:
    return (y - OMEGA_LO) / (OMEGA_HI - OMEGA_LO)


def _denorm_y(y: np.ndarray) -> np.ndarray:
    return y * (OMEGA_HI - OMEGA_LO) + OMEGA_LO


def fit_ridge(xtr: np.ndarray, ytr: np.ndarray, lam: float = 1e-2) -> Callable[[np.ndarray], np.ndarray]:
    """Linear ridge regression on the features."""
    mu = xtr.mean(0)
    A = np.c_[xtr - mu, np.ones(len(xtr))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return lambda x: np.c_[x - mu, np.ones(len(x))] @ w


def fit_mlp(
    xtr: np.ndarray, ytr: np.ndarray, hidden: int, seed: int, epochs: int = 400
) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    """Two-hidden-layer tanh MLP trained with Adam; returns (predict, n_params)."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )
    X = torch.tensor(xtr, dtype=torch.float32)
    Y = torch.tensor(_norm_y(ytr), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(X)[:, 0] - Y) ** 2).mean().backward()
        opt.step()

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return _denorm_y(net(torch.tensor(x, dtype=torch.float32))[:, 0].numpy())

    return predict, sum(p.numel() for p in net.parameters())


class VQC:
    """Data re-uploading variational circuit regressor (exact statevector simulation in torch).

    Per layer: RY(s_l,i * x_i) encoding (every layer if `reupload`, else layer 0 only), trainable
    RY-RZ on each qubit, then a ring of CZ gates. Output is <Z_0> mapped affinely to [0, 1].
    With `shots` set, <Z_0> is replaced by a binomial estimate (finite-sampling readout noise).
    """

    def __init__(self, n_qubits: int = 4, layers: int = 3, reupload: bool = True, seed: int = 0):
        self.n, self.L, self.reupload = n_qubits, layers, reupload
        g = torch.Generator().manual_seed(seed)
        self.theta = (0.3 * torch.randn(layers, n_qubits, 2, generator=g)).requires_grad_(True)
        self.scale = torch.ones(layers if reupload else 1, n_qubits).requires_grad_(True)
        self.out_scale = torch.tensor(1.0, requires_grad=True)
        self.n_params = self.theta.numel() + self.scale.numel() + 1
        idx = torch.arange(2**n_qubits)
        self.bits = (idx[:, None] >> (n_qubits - 1 - torch.arange(n_qubits))) & 1
        self.z0 = 1.0 - 2.0 * self.bits[:, 0]
        cz = torch.ones(2**n_qubits)
        for q in range(n_qubits):
            both = self.bits[:, q] & self.bits[:, (q + 1) % n_qubits]
            cz = cz * (1.0 - 2.0 * both)
        self.cz_phase = cz  # a CZ ring is diagonal in the computational basis

    def _ry(self, psi: torch.Tensor, q: int, ang: torch.Tensor) -> torch.Tensor:
        B = psi.shape[0]
        psi = psi.reshape(B, 2**q, 2, 2 ** (self.n - q - 1))
        c = torch.cos(ang / 2).reshape(-1, 1, 1)
        s = torch.sin(ang / 2).reshape(-1, 1, 1)
        a0, a1 = psi[:, :, 0], psi[:, :, 1]
        return torch.stack([c * a0 - s * a1, s * a0 + c * a1], dim=2).reshape(B, -1)

    def _rz(self, psi: torch.Tensor, q: int, ang: torch.Tensor) -> torch.Tensor:
        return psi * torch.exp(-0.5j * ang * (1.0 - 2.0 * self.bits[:, q]))

    def expval(self, x: torch.Tensor) -> torch.Tensor:
        """Exact <Z_0> for features x of shape (B, n_qubits)."""
        B = x.shape[0]
        psi = torch.zeros(B, 2**self.n, dtype=torch.complex64)
        psi[:, 0] = 1.0
        for l in range(self.L):
            if l == 0 or self.reupload:
                s = self.scale[l if self.reupload else 0]
                for q in range(self.n):
                    psi = self._ry(psi, q, s[q] * x[:, q])
            for q in range(self.n):
                psi = self._ry(psi, q, self.theta[l, q, 0].expand(B))
                psi = self._rz(psi, q, self.theta[l, q, 1])
            psi = psi * self.cz_phase
        return (psi.abs() ** 2 * self.z0).sum(1)

    def forward(self, x: torch.Tensor, shots: int | None = None, gen: torch.Generator | None = None) -> torch.Tensor:
        z = self.expval(x)
        if shots is not None:
            p0 = ((1 - z.detach()) / 2).clamp(0, 1)  # P(bit 0 = 1)
            k = torch.binomial(torch.full_like(p0, float(shots)), p0, generator=gen)
            z = 1 - 2 * k / shots
        return self.out_scale * (z + 1) / 2

    @staticmethod
    def _angles(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.clip(x, -2.5, 2.5) * (np.pi / 2.5), dtype=torch.float32)

    def fit(self, xtr: np.ndarray, ytr: np.ndarray, steps: int = 250, lr: float = 0.08) -> "VQC":
        X, Y = self._angles(xtr), torch.tensor(_norm_y(ytr), dtype=torch.float32)
        opt = torch.optim.Adam([self.theta, self.scale, self.out_scale], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            ((self.forward(X) - Y) ** 2).mean().backward()
            opt.step()
        return self

    def predict(self, x: np.ndarray, shots: int | None = None, seed: int = 0) -> np.ndarray:
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            return _denorm_y(self.forward(self._angles(x), shots, gen).numpy())
