"""Figures from results/*.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rydberg_est.estimators import OMEGA_HI, OMEGA_LO
from rydberg_est.physics import SCALE, Params, absorption

STY = {"model_fit": ("k", "o", "Model fit (nominal physics)"), "ridge_pca4": ("tab:gray", "s", "Ridge (PCA-4)"),
       "mlp_raw": ("tab:blue", "^", "MLP (raw, 6.4k params)"), "mlp_pca4_small": ("tab:green", "v", "MLP (PCA-4, 5 hidden)"),
       "vqc_L3": ("tab:red", "D", "VQC (4 qubits, 3 layers)")}


def make_figures(res: Path, figs: Path) -> None:
    g = pd.read_csv(res / "summary.csv")
    crb = pd.read_csv(res / "crb.csv")
    # Fig 1: spectra
    fig, ax = plt.subplots(figsize=(5, 3.4))
    dets = np.linspace(-20, 20, Params().n_det)
    for w in (3, 6, 10, 14):
        ax.plot(dets, absorption(np.array([w]), 8.0)[0] / SCALE, label=rf"$\Omega_{{RF}}$={w} MHz")
    ax.set_xlabel("Probe detuning (MHz)")
    ax.set_ylabel("Normalised probe absorption")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "fig1_spectra.png", dpi=200)
    plt.close(fig)
    # Fig 2: noise sweep, two drift panels
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, d in zip(axs, (0.0, 0.1)):
        for m, (c, mk, lab) in STY.items():
            s = g[(g.drift == d) & (g.method == m) & (g.n_train == g[g.drift == d].n_train.max()) & (g.sigma.isin(crb.sigma))]
            s = s[s["n_train"] == s["n_train"].max()].sort_values("sigma")
            ax.errorbar(s.sigma, s["mean"], s["std"].fillna(0), color=c, marker=mk, label=lab, capsize=2, ms=4)
        ax.plot(crb.sigma, crb.crb_rmse, "k--", lw=1, label="Cramér–Rao (nominal model)")
        ax.set(xscale="log", yscale="log", xlabel="Detection noise σ (frac. of peak)", title=f"Coupling drift = {int(d*100)}%")
    axs[0].set_ylabel(r"RMSE of $\Omega_{RF}$ (MHz)")
    axs[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "fig2_noise_sweep.png", dpi=200)
    plt.close(fig)
    # Fig 3: training size
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for m, (c, mk, lab) in STY.items():
        s = g[(g.drift == 0.1) & (g.sigma == 0.02) & (g.method == m)].sort_values("n_train")
        ax.errorbar(s.n_train, s["mean"], s["std"].fillna(0), color=c, marker=mk, label=lab, capsize=2, ms=4)
    ax.set(xscale="log", yscale="log", xlabel="Training spectra", ylabel=r"RMSE of $\Omega_{RF}$ (MHz)",
           title="σ=0.02, 10% coupling drift")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "fig3_train_size.png", dpi=200)
    plt.close(fig)
    # Fig 4: ablation
    names = ["vqc_L1", "vqc_L2", "vqc_L3_noreupload", "vqc_L3", "vqc_L3_shots1000", "vqc_L3_shots100"]
    s = g[(g.drift == 0.1) & (g.sigma == 0.02) & (g.n_train == g.n_train.max()) & g.method.isin(names)]
    s = s.set_index("method").reindex([n for n in names if n in s.method.values])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(range(len(s)), s["mean"], yerr=s["std"].fillna(0), color="tab:red", capsize=3)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([n.replace("vqc_", "").replace("_", "\n") for n in s.index], fontsize=7)
    ax.set_ylabel(r"RMSE of $\Omega_{RF}$ (MHz)")
    ax.set_title("VQC ablation (σ=0.02, 10% drift)")
    fig.tight_layout()
    fig.savefig(figs / "fig4_vqc_ablation.png", dpi=200)
    plt.close(fig)
