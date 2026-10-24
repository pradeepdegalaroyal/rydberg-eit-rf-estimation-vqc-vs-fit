"""Reproduce every result and figure.  `python experiments/run_all.py [--quick]`."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rydberg_est.estimators import (PCA, VQC, ModelFit, add_noise, fit_mlp, fit_ridge, make_data)
from rydberg_est.physics import SCALE, crb_std

torch.set_num_threads(6)
ROOT = Path(__file__).resolve().parents[1]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_cell(fit: ModelFit, seed: int, sigma: float, drift: float, n_train: int, n_test: int,
             vq_steps: int, mlp_epochs: int, extra: bool = False) -> list[dict]:
    """Train/evaluate all estimators for one (seed, sigma, drift, n_train) cell."""
    rng = np.random.default_rng(1000 * seed + 7)
    xtr, ytr = make_data(n_train, rng, drift)
    xte, yte = make_data(n_test, rng, drift)
    xtr, xte = add_noise(xtr, sigma, rng), add_noise(xte, sigma, rng)
    base = dict(seed=seed, sigma=sigma, drift=drift, n_train=n_train)
    rows: list[dict] = []

    def add(method: str, pred: np.ndarray, n_par: int = 0, **kw):
        rows.append({**base, "method": method, "rmse": rmse(pred, yte), "n_params": n_par, **kw})

    add("model_fit", fit.predict(xte))
    pca = PCA().fit(xtr, 4)
    ftr, fte = pca.transform(xtr), pca.transform(xte)
    add("ridge_pca4", fit_ridge(ftr, ytr)(fte), 5)
    pred, npar = fit_mlp(xtr, ytr, 64, seed, mlp_epochs)
    add("mlp_raw", pred(xte), npar)
    pred, npar = fit_mlp(ftr, ytr, 5, seed, mlp_epochs)
    add("mlp_pca4_small", pred(fte), npar)
    vqc = VQC(4, 3, True, seed).fit(ftr, ytr, vq_steps)
    add("vqc_L3", vqc.predict(fte), vqc.n_params)
    if extra:
        for L, re in [(1, True), (2, True), (3, False)]:
            v = VQC(4, L, re, seed).fit(ftr, ytr, vq_steps)
            add(f"vqc_L{L}" + ("" if re else "_noreupload"), v.predict(fte), v.n_params)
        for shots in (1000, 100):
            rows.append({**base, "method": f"vqc_L3_shots{shots}", "n_params": vqc.n_params,
                         "rmse": rmse(vqc.predict(fte, shots, seed), yte)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    seeds = [0] if q else [0, 1, 2]
    n_test = 100 if q else 200
    vq_steps, ep = (30, 60) if q else (120, 200)
    sigmas = [0.02] if q else [0.01, 0.02, 0.05]
    sizes = [64] if q else [32, 128, 256]
    n_main = 64 if q else 256
    out = ROOT / ("results_quick" if q else "results")
    figs = ROOT / ("figures_quick" if q else "figures")
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    t0 = time.time()
    fit = ModelFit()

    rows: list[dict] = []
    for drift in (0.0, 0.1):                       # E1: noise sweep, with/without unmodelled drift
        for s in sigmas:
            for sd in seeds:
                rows += run_cell(fit, sd, s, drift, n_main, n_test, vq_steps, ep, extra=(s == 0.02 and drift == 0.1))
                print(f"E1 drift={drift} sigma={s} seed={sd} t={time.time()-t0:.0f}s", flush=True)
    for n in sizes:                                # E2: training-set size at fixed noise, with drift
        for sd in seeds:
            rows += run_cell(fit, sd, 0.02, 0.1, n, n_test, vq_steps, ep)
            print(f"E2 n={n} seed={sd} t={time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False)

    crb = pd.DataFrame([{"sigma": s, "crb_rmse": float(np.sqrt(np.mean([crb_std(w, s) ** 2 for w in np.linspace(3, 15, 25)])))}
                        for s in sigmas])
    crb.to_csv(out / "crb.csv", index=False)
    g = df.groupby(["drift", "sigma", "n_train", "method"])["rmse"].agg(["mean", "std", "count"]).reset_index()
    g.to_csv(out / "summary.csv", index=False)
    (out / "meta.json").write_text(f'{{"runtime_s": {time.time()-t0:.1f}, "signal_scale": {SCALE:.6g}, "seeds": {len(seeds)}}}')
    if not q:
        from make_figures import make_figures
        make_figures(out, figs)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
