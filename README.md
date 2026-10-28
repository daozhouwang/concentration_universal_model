# Halo Concentration Universal Model (Paper I & Paper II)

A Python package for computing halo mass variance, formation-time proxies, and halo concentration predictions using the universal models calibrated in **Paper I** (population-level statistics: `peak`, `mean`, `median`) and **Paper II** (individual halo concentrations from assembly histories and concentration distributions: `relaxed`, `all_halos`, `relaxed_einasto`).

This repository implements a unified two-step workflow:

1. compute halo mass variance from an input halo file (`input_masses.dat`);
2. predict halo formation-time quantities and halo concentrations using the universal models of Paper I and Paper II.

The package supports both standard CDM-like cosmologies and $\Lambda$WDM cosmologies with an analytic half-mode mass ($M_{\rm hm}$) correction. To keep outputs cleanly separated:
- **Paper I outputs** (statistical $c$–$M$ relations) are saved in `./output_halo_properties/`
- **Paper II outputs** (individual halo $c$ & $c$ distributions) are saved in `./output_individual_halo_properties/`

---

## Features

- Compute halo mass variance $\sigma(M)$ and linear growth factor $D(z)$ from input halo masses
- **Paper I (Population Statistics)**:
  - Predict characteristic halo formation-time proxy $D(z_{\rm f})$
  - Predict statistical halo concentrations (`peak`, `mean`, `median`) under `vir` and `200c` definitions
- **Paper II (Individual Halos & Concentration Distributions)**:
  - **Case A — Known Formation Time (`Mass  z_f` in `input_masses.dat`)**: Given individual halo mass $M$ and half-mass assembly redshift $z_{\rm f}$, compute each halo's scaled formation variable $\omega_{\rm f}$ and directly predict its concentration $c_\Delta$ via the empirical multi-linear regression model (Equation 5 for NFW; Equation 8 for Einasto $\alpha=0.18$).
  - **Case B — Unknown Formation Time, Only $M$ and $z$ Given (`Mass` in `input_masses.dat`)**: When only halo mass $M$ is provided, automatically draw $\omega_{\rm f}$ via Monte Carlo sampling from the simulation-calibrated Generalized Gamma distribution $P(\omega_{\rm f})$ (Equation 6, Figure 3) and theoretical EPS distribution $p_{\rm EPS}(\omega_{\rm f})$, and combine with the intrinsic Gaussian residual scatter $\mathcal{N}(0, \sigma_{\rm res}^2)$ to predict the full halo concentration distribution $c(M)$ (Figure 4).
- Support for multiple cosmologies (`cosmology_params.py`):
  - `ΛCDM` (`planck18`, `planck15`, `WMAP1`–`WMAP9`, etc.)
  - `OCDM`, `wCDM`, `SCDM`
  - `LWDM` (with automatic half-mode mass $M_{\rm hm}$ calculation and sharp-$k$ filter)
  - scale-free / EdS-like setups (`self_similar`)

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── input_masses.dat
├── cosmology_params.py
├── GrowthFactor.py
├── VarianceCalculator.py
├── halo_variance.py
├── halo_model.py
├── halo_pipeline.py
├── output_halo_properties/             # Paper I outputs (statistical c: peak, mean, median)
└── output_individual_halo_properties/  # Paper II outputs (individual halo c & c distribution)
```

## Main Files

- `cosmology_params.py`  
  Cosmological parameter configurations. Users may also define additional cosmologies of interest by extending this file.

- `GrowthFactor.py`  
  Linear growth factor $D(z)$ calculation.

- `VarianceCalculator.py`  
  Halo mass variance $\sigma(M)$ calculation tools (real-space top-hat filter for CDM, sharp-$k$ filter for WDM).

- `halo_variance.py`  
  Script to compute halo variance (and $D(z_{\rm f})$ when a second column $z_{\rm f}$ is provided in `input_masses.dat`).

- `halo_model.py`  
  Universal halo concentration models for both Paper I and Paper II.

- `halo_pipeline.py`  
  Unified end-to-end pipeline for variance and concentration prediction.

---

## Quick Start

### 1. Prepare `input_masses.dat`

- **1 column (`Mass`)**: Provide halo masses in $[h^{-1}M_\odot]$.
  - Used by **Paper I** to compute statistical concentrations (`peak`, `mean`, `median`).
  - Used by **Paper II** when halo formation times are unknown: the code automatically performs Monte Carlo sampling of $\omega_{\rm f}$ from $P(\omega_{\rm f})$ and adds the Gaussian residual scatter $\mathcal{N}(0, \sigma_{\rm res}^2)$ to predict the concentration distribution $c(M)$ (Paper II Figure 4).
- **2 columns (`Mass  z_f`)**: Provide halo mass $M$ $[h^{-1}M_\odot]$ in column 1 and half-mass assembly redshift $z_{\rm f}$ in column 2.
  - Used by **Paper II** when individual halo formation times are known: the code computes the exact $D(z_{\rm f})$ and $\omega_{\rm f}$ for each halo and predicts its individual concentration $c_\Delta$.

### 2. Run the Pipeline

```bash
# =========================================================================
# Paper I: Statistical concentrations -> saved to ./output_halo_properties/
# =========================================================================
python halo_pipeline.py planck18 0.0          # default: peak (200c & vir)
python halo_pipeline.py planck18 0.0 median   # median (200c & vir)
python halo_pipeline.py planck18 0.0 mean     # mean (200c & vir)
python halo_pipeline.py planck18 0.0 all      # peak, mean, and median

# =========================================================================
# Paper II: Individual halo c & c(M) distribution
#           -> saved to ./output_individual_halo_properties/
# =========================================================================
python halo_pipeline.py planck18 0.0 individual       # runs relaxed, all_halos, and relaxed_einasto
python halo_pipeline.py planck18 0.0 relaxed          # NFW relaxed halos (200c & vir)
python halo_pipeline.py planck18 0.0 all_halos        # NFW all halos (200c & vir)
python halo_pipeline.py planck18 0.0 relaxed_einasto  # Einasto (alpha=0.18) relaxed halos (200c)
```

If you only want to compute the halo mass variance table (`output_variance.dat`):

```bash
python halo_variance.py input_masses.dat planck18 0.0
```

---

## Model Description

### Part 1: Paper I — Statistical Concentration Model (`peak`, `mean`, `median`)

The universal characteristic formation-time model is

```math
D(z_{\rm f})_{\rm CDM}
=
D(z)\left[
a_1 \left(\frac{M}{10^{12}\,h^{-1}M_\odot}\right)^{b_1}
+
a_2 D(z)^{b_2}
\right].
```

For $\Lambda$WDM cosmologies, a half-mode mass correction is applied:

```math
D(z_{\rm f})
=
D(z_{\rm f})_{\rm CDM}
\left[
1 + \eta \left(\frac{M_{\rm hm}}{M}\right)^\mu
\right].
```

The population concentration (`peak`, `mean`, or `median`) is

```math
c = \alpha \nu_{\rm eff}^{\beta} + \gamma,
\qquad
\nu_{\rm eff} = \nu_{\rm peak}\left(1 + A \frac{M_{\rm hm}}{M}\right),
\qquad
\nu_{\rm peak} = \frac{\delta_c}{\sigma(M)\,D(z)\,D(z_{\rm f})}.
```

---

### Part 2: Paper II — Individual Halo Concentration & Distribution Model (`relaxed`, `all_halos`, `relaxed_einasto`)

To parameterize the individual mass accretion history (MAH) of a halo, Paper II uses the EPS-scaled formation variable:

```math
\omega_{\rm f} = \frac{\delta_{\rm sc}(z_{\rm f}) - \delta_{\rm sc}(z)}{\sqrt{\sigma^2(0.5M) - \sigma^2(M)}},
\qquad
\delta_{\rm sc}(z) = \frac{1.686}{D(z)}.
```

The concentration of an individual halo is modeled via the universal regression relation (Paper II Equation 5 for NFW; Equation 8 for Einasto $\alpha=0.18$):

```math
\log_{10} c_{\Delta,{\rm eq5}}
=
A + \alpha_D \log_{10} D(z) + \alpha_\sigma \log_{10} \sigma(M) + \beta \log_{10}(1 + \omega_{\rm f}) + \gamma \log_{10} A_{\rm hm},
```

where the WDM small-scale damping factor is

```math
A_{\rm hm} = 1 + a \frac{M_{\rm hm}}{M_\Delta},
```

and the calibrated parameters $(A, \alpha_D, \alpha_\sigma, \beta, \gamma, a, \sigma_{\rm res})$ are taken from Table 2 and Equation (8) of Paper II.

#### Two Prediction Regimes in Paper II:

1. **Predicting Individual Halo $c$ from Known Formation Time (`input_masses.dat` has 2 columns: `Mass  z_f`)**:
   - When the half-mass formation redshift $z_{\rm f}$ of each halo is provided in the input file, `halo_variance.py` computes the exact $D(z_{\rm f})$ for each halo, and `halo_model.py` evaluates its exact $\omega_{\rm f}$ to directly predict the individual halo concentration $c_{\rm sim\_eq5}$ (Column 8) via Equation (5) or Equation (8).
   - A realization including the intrinsic logarithmic residual scatter $\mathcal{N}(0, \sigma_{\rm res}^2)$ is also provided in Column 9 (`c_sim_mc`).

2. **Predicting the Concentration Distribution $c(M)$ When Only Mass $M$ Is Given (`input_masses.dat` has 1 column: `Mass`)**:
   - When formation times are not provided in `input_masses.dat`, the code draws random realizations of $\omega_{\rm f}$ via Monte Carlo sampling from the simulation-calibrated Generalized Gamma PDF (Paper II Equation 6, Figure 3):
     ```math
     P(\omega_{\rm f}) = C_0 \left(\frac{\omega_{\rm f}}{\omega_0}\right)^k \exp\left[-\frac{k}{p}\left(\frac{\omega_{\rm f}}{\omega_0}\right)^p\right], \qquad (\omega_{\rm f} \ge 0),
     ```
     with $(\omega_0, k, p) = (1.25, 2.14, 2.04)$ for CDM and $(1.21, 1.82, 2.70)$ for WDM, as well as from the theoretical EPS distribution $p_{\rm EPS}(\omega_{\rm f}) = 2\omega_{\rm f}\,{\rm erfc}(\omega_{\rm f}/\sqrt{2})$.
   - Following Paper II (Figure 4), the predicted concentration distribution incorporates the Gaussian residual scatter $\mathcal{N}(0, \sigma_{\rm res}^2)$ around the regression relation:
     ```math
     \log_{10} c_{\rm MC} = \log_{10} c_{\Delta,{\rm eq5}}(\omega_{\rm f}) + \mathcal{N}(0, \sigma_{\rm res}^2),
     ```
     which is saved in Column 9 (`c_sim_mc`, using $\omega_{\rm f} \sim P(\omega_{\rm f})$) and Column 10 (`c_eps_mc`, using $\omega_{\rm f} \sim p_{\rm EPS}(\omega_{\rm f})$), while Column 8 (`c_sim_eq5`) stores the noiseless regression value $c_{\Delta,{\rm eq5}}$.

---

## Output Files & Column Descriptions

### 1. Paper I Outputs (`./output_halo_properties/halo_properties_{z}_{cosmology}_{200c|vir}_{peak|mean|median}.dat`)
- `0:Mvir[Msun/h]`: Halo mass $[h^{-1}M_\odot]$
- `1:sigma(M,0)`: Linear rms density fluctuation $\sigma(M, z=0)$
- `2:sigma(M_f,0)(f=0.50)`: Linear rms fluctuation at half mass $\sigma(0.5M, z=0)$
- `3:z`: Halo identification redshift $z$
- `4:D(z)`: Linear growth factor at redshift $z$
- `5:D(z_f)_simfit`: Characteristic formation growth factor $D(z_{\rm f})$ from Paper I universal fit
- `6:D(z_f)_eps(f=0.50)`: EPS formation growth factor for $f=0.50$
- `7:D(z_f)_eps(f=0.16)`: EPS formation growth factor for $f=0.16$
- `8:c_simfit`: Predicted statistical concentration (`peak`, `mean`, or `median`) from Paper I
- `9:c_eps(f=0.50)`: Concentration predicted using $D(z_{\rm f})_{\rm eps}(f=0.50)$
- `10:c_eps(f=0.16)`: Concentration predicted using $D(z_{\rm f})_{\rm eps}(f=0.16)$
- `11:M_hm[Msun/h]`: Half-mode mass $[h^{-1}M_\odot]$ ($0$ for CDM)

### 2. Paper II Outputs (`./output_individual_halo_properties/c_distribution_{z}_{cosmology}_{200c|vir}_{relaxed|all_halos|relaxed_einasto}.dat`)
- `0:Mvir[Msun/h]`: Halo mass $[h^{-1}M_\odot]$
- `1:sigma(M,0)`: Linear rms density fluctuation $\sigma(M, z=0)$
- `2:sigma(M_f,0)(f=0.50)`: Linear rms fluctuation at half mass $\sigma(0.5M, z=0)$
- `3:z`: Halo identification redshift $z$
- `4:D(z)`: Linear growth factor at redshift $z$
- `5:omega_f_sim`: Scaled formation variable $\omega_{\rm f}$ — computed from input $z_{\rm f}$ if 2 columns (`Mass  z_f`) are provided in `input_masses.dat`, or Monte Carlo sampled from Eq. (6) $P(\omega_{\rm f})$ if only 1 column (`Mass`) is provided
- `6:omega_f_eps`: $\omega_{\rm f}$ Monte Carlo sampled from the theoretical EPS distribution $p_{\rm EPS}(\omega_{\rm f})$
- `7:D(z_f)_sim`: Linear growth factor $D(z_{\rm f})$ corresponding to `omega_f_sim`
- `8:c_sim_eq5`: Concentration predicted directly by Eq. (5) / Eq. (8) without adding $\mathcal{N}(0, \sigma_{\rm res}^2)$ (exact regression prediction for individual halos when $z_{\rm f}$ is given)
- `9:c_sim_mc`: Predicted concentration including the intrinsic Gaussian scatter $\mathcal{N}(0, \sigma_{\rm res}^2)$ in $\log_{10} c$, i.e. $\log_{10} c_{\rm sim\_mc} = \log_{10} c_{\rm sim\_eq5} + \mathcal{N}(0, \sigma_{\rm res}^2)$ (reproduces the simulation-based $P(\omega_{\rm f})$ concentration distribution in Paper II Figure 4)
- `10:c_eps_mc`: Predicted concentration using EPS $\omega_{\rm f,eps}$ and including $\mathcal{N}(0, \sigma_{\rm res}^2)$ (reproduces the EPS-based concentration distribution in Paper II Figure 4)
- `11:M_hm[Msun/h]`: Half-mode mass $[h^{-1}M_\odot]$ ($0$ for CDM)

---

## Citation

If you use this code in scientific work, please cite:
- **Paper I**: Wang, D.-Z., Lin, W., & Luan, T.-C. 2026, *Universal Fitting Formulae for the Peak Concentration of Dark Matter Halos*, ApJ, 1006, 61, doi:10.3847/1538-4357/ae81a2
- **Paper II**: Wang, D.-Z., Lin, W., Zhu, W., & Luan, T.-C. 2026, *Predicting Individual Dark Matter Halo Concentrations from Their Assembly Histories*
