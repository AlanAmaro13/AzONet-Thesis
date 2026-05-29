# Parameters Estimation — Alonso's Transmittance Model Fitting

This folder contains the fitting of experimental transmittance spectra to the theoretical model proposed by Alonso et al. (IIM — Interference Induced Method) using three global optimization algorithms from SciPy.

## Context

In the previous folder (`0_data_processing`), a DataFrame was created containing experimental spectra and their thicknesses. The goal here is to fit each experimental transmittance curve with the theoretical model by minimizing the Mean Square Error (MSE).

## Transmittance Model

The Alonso model describes transmittance as a function of **13 parameters**:

| Parameter | Description |
|-----------|-------------|
| `x` | Wavelength [nm] |
| `d` | Film thickness [nm] |
| `R1`, `R2` | Surface roughness parameters |
| `A`, `B`, `C`, `D`, `E` | Sellmeier coefficients (background refractive index) |
| `alpha`, `beta`, `gamma` | Absorption coefficients |
| `ne` | Free carrier concentration |

The model incorporates:
- **Sellmeier equation** for the background dielectric constant
- **Drude model** for free-carrier contribution (depends on `ne`)
- **Absorption model** with Urbach tail
- **Roughness corrections** for interfaces (via scalar scattering theory)
- **Glass substrate** transmittance (`TexpglassO.txt`) to compute `ng`

The refractive index is constrained to the **physically meaningful range**: `1.8 ≤ n(λ) ≤ 2.1`.

## Data

- **Source**: `../../results/dataframe_spectrum_thickness_145_final.pkl`
- **Samples**: 145 experimental thin-film samples (AZO — Aluminium-doped Zinc Oxide)
- **Spectrum range**: 190–1100 nm (911 points per spectrum)
- **Glass transmittance**: `../../experimental_samples/Background_data/TexpglassO.txt`

## Optimization Algorithms

Three global optimization algorithms were tested:

### 1. Basin-Hopping (`basin_hopping`)
- **File**: `1_SciPy_AllSamples_BH_NonLinear_Porcentual_145F.ipynb`
- **Local optimizer**: SLSQP
- **Constraint**: `NonlinearConstraint` on refractive index
- **Initial guess**: Mean values from prior EGP-GA results
- **Bounds**: Allowed ±50% variation around seed values, ± measurement error for thickness

### 2. Differential Evolution (`differential_evolution`)
- **File**: `2_SciPy_AllSamples_DE_NonLinear_Porcentual_145F.ipynb`
- **Constraint**: `NonlinearConstraint` on refractive index
- **Parallelization**: `workers=-1` (uses all available CPUs)
- **Stochastic**: Random seed per run

### 3. Direct (`direct`)
- **File**: `3_SciPy_AllSamples_Direct_NonLinear_Porcentual_145F.ipynb`
- **Constraint**: Penalty method (Direct does not natively support `NonlinearConstraint`)
- **Fastest** but weakest results

## Results

Results are saved as NumPy arrays in `../../results/SciPy_IIM/`:

| File | Algorithm |
|------|-----------|
| `basin_hopping_NonLinear_145F.npy` | Basin-Hopping |
| `differential_evolution_NonLinear_145F.npy` | Differential Evolution |
| `direct_NonLinear_145F.npy` | Direct |

Each array has shape `(145, 14)` with columns:

```
[SampleIndex, Thickness, R1, R2, A, B, C, D, E, alpha, beta, gamma, ne, MSE]
```

## Comparison & Conclusions

The notebook `4_Comparison.ipynb` (which also generates `comparativa_optimizadores.pdf`) compares the three optimizers:

| Optimizer | Mean MSE | Execution Time | Quality |
|-----------|----------|----------------|---------|
| **Differential Evolution** | ~0 (best) | ~4 hours | Excellent |
| **Basin-Hopping** | ~5 | ~12 hours | Moderate |
| **Direct** | ~8–30 | ~5 minutes | Poor |

**Key findings**:

- **Differential Evolution** is the best optimizer for this problem — it achieves the lowest MSE consistently across all samples. Its KDE density distribution is tightly centered near zero.
- **Basin-Hopping** produces a normal-like distribution centered around MSE ≈ 5, with significantly longer runtime.
- **Direct** is fast but produces large errors, with a multi-modal error distribution. Not recommended for this problem.

The comparison visualizations are saved in the `images/` folder:
- `comparative.png` — Scatter plot of MSE per sample for all three algorithms
- `kde_optimizers.png` — Kernel Density Estimation of the MSE distributions
- `final_comparative.png` — Bar chart of mean MSE comparison

## Notebooks

| Notebook | Description |
|----------|-------------|
| `1_SciPy_AllSamples_BH_NonLinear_Porcentual_145F.ipynb` | Basin-Hopping fitting (interpreted) |
| `2_SciPy_AllSamples_DE_NonLinear_Porcentual_145F.ipynb` | Differential Evolution fitting (no outputs) |
| `3_SciPy_AllSamples_Direct_NonLinear_Porcentual_145F.ipynb` | Direct fitting (no outputs) |
| `4_Comparison.ipynb` | Results comparison, KDE, bar charts, and PDF report generation |
| `Python_Notebooks/*.py` | Python script versions using the `AmaroX` library (same logic, alternative imports) |

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- `fpdf` (for PDF report generation in `4_Comparison.ipynb`) 
- `AmaroX` (custom library, used only in `Python_Notebooks/*.py`)
