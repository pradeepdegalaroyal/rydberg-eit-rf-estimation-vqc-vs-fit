import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rydberg_est.estimators import PCA, VQC, ModelFit, add_noise, make_data  # noqa: E402
from rydberg_est.physics import SCALE, absorption, crb_std  # noqa: E402


def test_absorption_shape_and_positive():
    a = absorption(np.array([4.0, 9.0]), 8.0)
    assert a.shape == (2, 32) and np.all(a > 0)
    assert SCALE > 0


def test_spectrum_depends_on_rf():
    a = absorption(np.array([4.0, 12.0]), 8.0)
    assert np.abs(a[0] - a[1]).max() > 0.01 * a.max()


def test_modelfit_noiseless_recovers():
    rng = np.random.default_rng(0)
    x, y = make_data(10, rng, 0.0)
    assert np.abs(ModelFit(n_grid=61).predict(x) - y).max() < 0.5


def test_crb_scales_with_sigma():
    assert abs(crb_std(8, 0.04) / crb_std(8, 0.02) - 2) < 1e-6


def test_vqc_range_and_training():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((40, 4))
    y = 9 + 3 * np.tanh(f[:, 0])
    v = VQC(4, 2, True, 0)
    e0 = np.mean((v.predict(f) - y) ** 2)
    v.fit(f, y, steps=40)
    assert np.mean((v.predict(f) - y) ** 2) < e0
    z = v.expval(torch.zeros(3, 4))
    assert ((z >= -1 - 1e-5) & (z <= 1 + 1e-5)).all()


def test_pca_whitening_and_noise():
    x = np.random.default_rng(0).standard_normal((50, 8))
    z = PCA().fit(x, 3).transform(x)
    assert z.shape == (50, 3) and np.allclose(z.std(0), 1, atol=1e-6)
    assert add_noise(x, 0.0, np.random.default_rng(0)).shape == x.shape
