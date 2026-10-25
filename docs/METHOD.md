# Method

**Forward model.** Four-level ladder |g>,|e>,|r>,|r'> (probe, coupling, resonant radio-frequency (RF) on the Rydberg pair), steady-state Lindblad solution, Doppler-averaged over 9 velocity classes, 32 probe detunings in ±20 MHz (`physics.py`). Output: Im ρ_ge, normalised to peak 1 at nominal parameters. RF Rabi frequency Ω_RF ~ U(3, 15) MHz. Nominal Ω_c = 8 MHz; per-spectrum "drift" multiplies Ω_c by (1 + d·N(0,1)), d ∈ {0, 0.1}.

**Noise.** i.i.d. Gaussian detection noise σ (fraction of peak) added to every point: σ ∈ {0.01, 0.02, 0.05}.

**Estimators.**
- *Model fit* (classical, physics-based baseline): maximum likelihood (ML) grid search over the nominal model plus parabolic refinement; assumes d = 0.
- *Ridge* on 4 whitened principal component analysis (PCA) features (linear baseline).
- *multilayer perceptron (MLP) raw* (32→64→64→1, ~6.4k params) and *MLP PCA-4 small* (4→5→5→1, 61 params, parameter-matched to the variational quantum circuit (VQC)).
- *VQC* (standard quantum baseline: data re-uploading variational regressor): 4 qubits, 3 layers of [single-qubit Y rotation (RY)(s·x) encoding, trainable RY–single-qubit Z rotation (RZ), controlled-Z (CZ) ring], read out ⟨Z0⟩; 4 PCA features; Adam, 120 steps; exact statevector.
- *Reference*: Cramér–Rao root-mean-square error (RMSE) of the nominal model, averaged over Ω_RF.

**Metrics.** RMSE (MHz) on 200 held-out noisy spectra; 3 seeds (each seed redraws train/test data, noise, and initialisation).

**Experiments.** E1 noise sweep × drift {0, 0.1} (256 training spectra). E2 training-set size {32, 128, 256} at σ = 0.02, d = 0.1. Ablation (σ = 0.02, d = 0.1): VQC depth 1/2/3, no data re-uploading. Robustness: VQC readout with finite shots (1000, 100) on ⟨Z0⟩.
