# notebooks/2_de — Differential Evolution Fitting Analysis

This folder contains the analysis of thin-film transmittance fitting results obtained via **Differential Evolution (DE)**, a global optimization algorithm.

## Contents

### Notebook
- **`0_comparison_dist.ipynb`** — Analyzes the fitting results for 145 AZO (Aluminum-doped Zinc Oxide) thin-film samples fitted using SciPy's Differential Evolution. It loads the optimized parameters and experimentally measured spectra, then generates extensive visualizations to evaluate fitting quality.

### What the notebook does

1. **Loads DE fitting results** from `../../results/SciPy_IIM/differential_evolution_NonLinear_145F.npy` — a (145, 14) array containing: thickness, roughness σ₁, roughness σ₂, 5 Sellmeier coefficients (A, B, C, D, E), 3 absorption coefficients (α, β, γ), electron concentration nₑ, and MSE.

2. **Loads experimental data** from `../../results/dataframe_spectrum_thickness_145_final.pkl` — contains sample names, thickness estimates, errors, and measured spectra (911 wavelength points from 190–1100 nm).

3. **Transmittance model** — Implements the full physical model with:
   - **Sellmeier equation** for bandgap refractive index contribution (5 parameters: A, B, C, D, E)
   - **Drude model** for free-electron effects (nₑ, mobility μ)
   - **Absorption** via Urbach tail (α, β, γ)
   - **Surface roughness** via scalar scattering theory (σ₁, σ₂)
   - **Multi-layer interference** (film on glass substrate, including glass transmittance from `TexpglassO.txt`)

4. **Visualizations** — Generates multiple families of plots to assess fitting quality:

   | Output Directory | Description |
   |---|---|
   | `images/comparison/` | 145 experimental vs. fitted transmittance spectra |
   | `images/comparison9/` | 16 grid overviews of fitted spectra |
   | `images/dist/` | KDE distributions of thickness, error, ECM |
   | `images/dist_f/` | Distributions of absorption, Sellmeier, and roughness parameters |
   | `images/ridge_plot/` | Ridge plots per parameter (Sellmeier A–E, σ₁, σ₂, α, β, γ, nₑ, λ threshold) |
   | `combined_absorcion.png` | Combined absorption parameter distributions |
   | `combined_rugosidades.png` | Combined roughness parameter distributions |
   | `combined_sellmeier.png` | Combined Sellmeier coefficient distributions |

5. **Binned analysis** — `bins/` directory contains pre-computed statistics (all_metrics, bin_index, min_max_mean_std, region) for 5 spectral wavelength regions, enabling region-wise evaluation of fitting performance.

### Key Libraries
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scipy`
