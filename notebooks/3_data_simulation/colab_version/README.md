# 3. Data Simulation — Colab Version

This section contains the pipeline for generating synthetic transmittance spectra of Al-doped Zinc Oxide (AzO) thin films using Gaussian Mixture Models (GMM). The goal is data augmentation: from 145 experimental spectra, we generate 1.2 million synthetic spectra to train deep learning models.

---

## Directory Structure

```
colab_version/
├── 1_DataSimulationForEachBin_0_Colab.ipynb   # Region 1 (100–250 nm): 5 samples
├── 1_DataSimulationForEachBin_1_Colab.ipynb   # Region 2 (250–600 nm): 46 samples
├── 1_DataSimulationForEachBin_2_Colab.ipynb   # Region 3 (600–950 nm): 49 samples
├── 1_DataSimulationForEachBin_3_Colab.ipynb   # Region 4 (950–1100 nm): 12 samples
├── 1_DataSimulationForEachBin_4_Colab.ipynb   # Region 5 (1100–1500 nm): 33 samples
├── Copia de 2_Figures.ipynb                   # Visualization (ridge plots, KDE, composite images)
├── Copia de 3_join_split_merge.ipynb          # Merge, shuffle, split into Train/Val/Test
├── metodologia.tex                             # Methodology chapter (LaTeX)
├── data/
│   ├── R1/   # GMM-generated distributions for Region 1 (12 .npy files)
│   ├── R2/   # GMM-generated distributions for Region 2 (12 .npy files)
│   ├── R3/   # GMM-generated distributions for Region 3 (12 .npy files)
│   ├── R4/   # GMM-generated distributions for Region 4 (12 .npy files)
│   └── R5/   # GMM-generated distributions for Region 5 (12 .npy files)
└── images/
    └── ridger_plot/   # Ridge/KDE plots for each parameter across regions
```

---

## Pipeline Overview

### 1. Input Data
Fitted parameters from Differential Evolution optimization of each experimental spectrum. Each sample yields 12 parameters:

| Index | Parameter | Description | Units |
|-------|-----------|-------------|-------|
| 0 | $r$ | Film thickness | nm |
| 1 | $\sigma_1$ | Surface roughness (air–film) | nm |
| 2 | $\sigma_2$ | Surface roughness (film–substrate) | nm |
| 3 | $A$ | Sellmeier coefficient | — |
| 4 | $B$ | Sellmeier coefficient | — |
| 5 | $C$ | Sellmeier coefficient | nm |
| 6 | $D$ | Sellmeier coefficient | — |
| 7 | $E$ | Sellmeier coefficient | nm |
| 8 | $\alpha_0$ | Absorption parameter | nm⁻¹ |
| 9 | $\beta$ | Absorption parameter | eV⁻¹ |
| 10 | $\lambda_g$ | Absorption parameter (bandgap wavelength) | nm |
| 11 | $n_e$ | Carrier density | nm⁻³ |

Parameters are organized by thickness region (5 dominant regions), loaded from `.npy` files.

### 2. GMM Fitting (`GMM` function)
Each parameter column is modeled independently via a 1D Gaussian Mixture Model using `sklearn.mixture.GaussianMixture` with `covariance_type='full'`.

- **Maximum gaussians: 3** (BIC-based selection among 1–3 components)
- The Bayesian Information Criterion (BIC) selects the optimal number of gaussians per parameter
- Output per parameter: `[centroids μ, std devs σ, weights P, min_val, max_val, gmm_object]`

For parameters with low sample count (Region 1 has only 5 samples), the GMM fitting is limited to 1 component to avoid overfitting.

### 3. Synthetic Distribution Generation (`GMM_Model`)
For each parameter, `N = 5,000` synthetic values are sampled:

- **Single gaussian**: Draw `N` samples from $\mathcal{N}(\mu, \sigma)$
- **Multiple gaussians**: For each component $i$, draw `P_i × N` samples from $\mathcal{N}(\mu_i, \sigma_i)$; concatenate all
- All values are clipped to `[min(ρ_original), max(ρ_original)]`

Thickness is additionally clipped to region boundaries: `[100, 250, 600, 950, 1100, 1500]`.

### 4. Sellmeier Constraint Filtering
For Sellmeier coefficients $A, B, C, D, E$, 3,000 valid combinations are selected among random draws from the synthetic distributions. A combination is valid only if the resulting refractive index satisfies:

$$1.8 \leq n(\lambda) \leq 2.1 \quad \forall \lambda \in [190, 1100] \text{ nm}$$

### 5. Spectrum Generation (240,000 per region)
For each of the 240,000 synthetic samples per region:
1. Randomly select one value from each parameter's synthetic distribution
2. Evaluate the optical transmittance model:
   `modelo_transmitancia(λ, d, σ₁, σ₂, A, B, C, D, E, α₀, β, λg, ne)`
3. Skip NaN/invalid results
4. Store: spectrum (911 points), thickness, roughnesses, Sellmeier coefficients, absorption parameters, carrier density

**Total: 5 regions × 240,000 = 1.2 M samples**

Each spectrum spans 190–1100 nm at 1 nm resolution (911 points).

### 6. Output Format
Each notebook saves a `bin_{i}.parquet` file with columns:
- `Espectro`: 911-point transmittance spectrum array
- `Espesor`: thickness (nm)
- `R1`, `R2`: roughness values
- `Sellmeier`: list `[A, B, C, D, E]`
- `Absorcion`: list `[α₀, β, λg]`
- `ne`: carrier density

### 7. Merge, Shuffle, Split (`Copia de 3_join_split_merge.ipynb`)
- Merges all 5 parquet files into a single DataFrame (1.2 M rows)
- Randomly shuffles with `df.sample(frac=1)`
- Splits into Train (80%), Test (10%), Validation (10%)
- Converts spectra to shape `(n, 911, 1)` and saves as HDF5

### 8. Visualization (`Copia de 2_Figures.ipynb`)
Generates ridge/KDE plots comparing original and synthetic distributions across regions, composite image grids for roughnesses, Sellmeier, absorption, and spectrum examples.

---

## Region Summary

| Region | Bin | Range [nm] | Experimental Samples |
|--------|-----|------------|---------------------|
| $R_1$ | 0 | 100–250 | 5 |
| $R_2$ | 1 | 250–600 | 46 |
| $R_3$ | 2 | 600–950 | 49 |
| $R_4$ | 3 | 950–1100 | 12 |
| $R_5$ | 4 | 1100–1500 | 33 |

---

## Related Files
- GMM parameter tables (TeX): `tablas_gmm_max3.tex`
- Methodology chapter: `metodologia.tex`
- Pre-computed GMM distributions: `data/R{1-5}/*.npy`
