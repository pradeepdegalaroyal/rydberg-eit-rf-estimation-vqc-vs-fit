# Research gap

## State of the art (from the supplied literature)
- Reviews of quantum machine learning (QML) report mostly promising but simulation-scale results and stress that present devices lack the qubit count and fidelity to deliver advantage (Peral-García et al., 2024, doi:10.1016/j.cosrev.2024.100619; Yang et al., 2023, doi:10.1109/COMST.2023.3254481).
- Variational quantum circuits (VQCs) are the standard near-term model, but there is a stated lack of "guiding principles for the construction of suitable parametrizations" (Meyer et al., 2023, doi:10.1103/PRXQuantum.4.010328), and quantum-kernel methods can suffer exponential concentration (Thanasilp et al., 2024, doi:10.1038/s41467-024-49287-w).
- It is "poorly understood" why variational quantum circuit (VQC) agents often fail to beat classical methods (doi:10.1016/j.ins.2026.124009).
- Most application papers benchmark on images, healthcare or generic tabular data; physical-sensor inference tasks with a known forward model and a known statistical bound are rare.

## Gap
Quantum radio-frequency (RF) sensing with Rydberg-atom electrometers reads out an RF field amplitude from a probe-laser electromagnetically induced transparency (EIT)/Autler–Townes spectrum. The physics gives (i) a forward model, hence a maximum-likelihood (ML) estimator, and (ii) a Cramér–Rao bound (CRB). This makes it a rare QML task where "does the learned model help?" has a sharp answer. Open question: when the forward model is *mismatched* (e.g. unmodelled coupling-laser power drift), can a small parameter-efficient VQC regressor recover the loss of the nominal-model fit, and how does it compare with parameter-matched classical learners and with the CRB? All results here are noiseless-statevector simulation of the circuit and a Lindblad steady-state model of the atoms, on a laptop.

## Hypothesis (falsifiable)
H1: with no model mismatch, the nominal-model ML fit is within ~1.5x of the CRB and no learned estimator beats it.
H2: with 10% coupling-Rabi-frequency drift, learned estimators trained on drifted data beat the nominal-model fit at low noise, because the fit is biased.
H3: a 4-qubit, 3-layer VQC (37 parameters) matches a parameter-matched classical multilayer perceptron (MLP) (61 parameters) on the same 4 principal component analysis (PCA) features within one standard deviation over seeds.
Success metric: root-mean-square error (RMSE) of the RF Rabi frequency (MHz) on held-out spectra, mean ± std over 3 seeds. H1–H3 are each reported as supported or refuted by `results/`.
