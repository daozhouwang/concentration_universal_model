"""
Calculate halo concentrations using the universal models from Paper I and Paper II:

1. Paper I (Statistical concentration: peak, mean, median) -> saved in ./output_halo_properties/
   D(z_f)_CDM = D(z) * [a1 * (M/1e12)^(b1) + a2 * D(z)^(b2)]
   D(z_f) = D(z_f)_CDM * [1 + eta * (M_hm/M)^mu]
   c = alpha * nu_eff^(beta) + gamma
   nu_eff = nu_peak * (1 + A * M_hm/M)
   nu_peak = delta_c / [sigma(M) * D(z) * D(z_f)]

2. Paper II (Individual halo c & c distribution: relaxed, all_halos, relaxed_einasto) -> saved in ./output_individual_halo_properties/
   log10(c_eq5) = A + alpha_D * log10(D(z)) + alpha_sigma * log10(sigma(M))
                  + beta * log10(1 + omega_f) + gamma * log10(A_hm)
   log10(c_mc)  = log10(c_eq5) + N(0, sigma_res^2)
   A_hm = 1 + a * M_hm / M
   omega_f = [delta_sc(z_f) - delta_sc(z)] / sqrt(sigma^2(0.5M) - sigma^2(M))

   Two input modes are supported automatically via input_masses.dat:
   - Case A (Known formation time z_f): 2 columns (Mass  z_f) in input_masses.dat
     -> Computes each halo's exact D(z_f) and omega_f from z_f, predicting individual
        halo concentration via Eq. (5) / Eq. (8) (8:c_sim_eq5) and with N(0, sigma_res^2) (9:c_sim_mc).
   - Case B (Unknown formation time, only Mass and z known): 1 column (Mass) in input_masses.dat
     -> Samples omega_f via Monte Carlo from the generalized gamma distribution P(omega_f)
        (Eq. 6, Fig. 3) and EPS distribution p_EPS(omega_f), and incorporates the Gaussian
        residual scatter N(0, sigma_res^2) to predict the c(M) distribution (9:c_sim_mc, 10:c_eps_mc, Fig. 4).
"""

import numpy as np
import sys
import os
from scipy.special import erfc, erf
from scipy.interpolate import interp1d

# =========================================================
# Import cosmology config
# =========================================================
try:
    from cosmology_params import cosmology_configs
except ImportError:
    print("Warning: cosmology_params module not found.")
    cosmology_configs = {}

# =========================================================
# Paper I: Statistical Concentration Model Parameters
# =========================================================
MODEL_PARAMS = {
    '200c': {
        'peak':   {'a1': 0.19, 'b1': 0.12, 'a2': 0.36, 'b2': -0.46, 'eta': 0.13, 'mu': 1.05, 'alpha': 8.81,  'beta': -0.95, 'gamma': 2.31, 'A': 3.06},
        'mean':   {'a1': 0.21, 'b1': 0.11, 'a2': 0.35, 'b2': -0.45, 'eta': 0.12, 'mu': 0.91, 'alpha': 9.44,  'beta': -1.12, 'gamma': 2.96, 'A': 1.73},
        'median': {'a1': 0.20, 'b1': 0.12, 'a2': 0.35, 'b2': -0.45, 'eta': 0.12, 'mu': 1.30, 'alpha': 9.16,  'beta': -1.05, 'gamma': 2.64, 'A': 2.50}
    },
    'vir': {
        'peak':   {'a1': 0.20, 'b1': 0.12, 'a2': 0.36, 'b2': -0.43, 'eta': 0.12, 'mu': 1.09, 'alpha': 12.03, 'beta': -1.09, 'gamma': 2.54, 'A': 3.98},
        'mean':   {'a1': 0.23, 'b1': 0.11, 'a2': 0.34, 'b2': -0.43, 'eta': 0.11, 'mu': 0.95, 'alpha': 12.36, 'beta': -1.22, 'gamma': 3.13, 'A': 2.36},
        'median': {'a1': 0.23, 'b1': 0.10, 'a2': 0.34, 'b2': -0.43, 'eta': 0.12, 'mu': 1.29, 'alpha': 12.07, 'beta': -1.14, 'gamma': 2.88, 'A': 3.43}
    }
}

# =========================================================
# Paper II: Individual Halo Concentration Model Parameters
# (Table 2 & Equation 8 of Paper II)
# =========================================================
INDIVIDUAL_PARAMS = {
    '200c': {
        'relaxed':         {'A': 0.46, 'alpha_D': 0.79, 'alpha_sigma': 0.64, 'beta': 0.67, 'gamma': -0.39, 'a': 5.59, 'sigma_res': 0.089},
        'all_halos':       {'A': 0.45, 'alpha_D': 0.82, 'alpha_sigma': 0.61, 'beta': 0.71, 'gamma': -0.48, 'a': 3.06, 'sigma_res': 0.096},
        'relaxed_einasto': {'A': 0.46, 'alpha_D': 0.89, 'alpha_sigma': 0.63, 'beta': 0.71, 'gamma': -0.44, 'a': 5.59, 'sigma_res': 0.098}
    },
    'vir': {
        'relaxed':         {'A': 0.61, 'alpha_D': 1.03, 'alpha_sigma': 0.66, 'beta': 0.63, 'gamma': -0.45, 'a': 4.77, 'sigma_res': 0.088},
        'all_halos':       {'A': 0.59, 'alpha_D': 1.05, 'alpha_sigma': 0.63, 'beta': 0.68, 'gamma': -0.43, 'a': 3.98, 'sigma_res': 0.094}
    }
}

# Generalized Gamma PDF parameters for omega_f (Paper II Eq. 6)
OMEGA_F_PDF_PARAMS = {
    'CDM':  {'omega_0': 1.25, 'k': 2.14, 'p': 2.04},
    'LWDM': {'omega_0': 1.21, 'k': 1.82, 'p': 2.70}
}

# =========================================================
# Physics Functions (Paper I & Shared)
# =========================================================
def universal_D_zf(D_z, Mvir, params, M_hm=0.0):
    D_z = np.asarray(D_z, dtype=float)
    Mvir = np.asarray(Mvir, dtype=float)
    M12 = Mvir / 1.0e12
    D_zf_cdm = D_z * (params['a1'] * M12**params['b1'] + params['a2'] * D_z**params['b2'])

    if np.any(np.asarray(M_hm) > 0):
        return D_zf_cdm * (1.0 + params['eta'] * (M_hm / Mvir)**params['mu'])
    return D_zf_cdm

def D_zf_eps_func(D_z, sigma_M, sigma_M_f, omega_f=0.75, delta_c=1.686):
    D_z = np.asarray(D_z, dtype=float)
    sigma_M = np.asarray(sigma_M, dtype=float)
    sigma_M_f = np.asarray(sigma_M_f, dtype=float)
    Delta_S2 = np.maximum(sigma_M_f**2 - sigma_M**2, 0.0)
    Delta_S = np.sqrt(Delta_S2)
    return D_z / (1.0 + (omega_f / delta_c) * D_z * Delta_S)

def universal_concentration(D_z, D_zf, sigma_M, Mvir, params, M_hm=0.0, delta_c=1.686):
    D_z = np.asarray(D_z, dtype=float)
    D_zf = np.asarray(D_zf, dtype=float)
    sigma_M = np.asarray(sigma_M, dtype=float)
    Mvir = np.asarray(Mvir, dtype=float)

    denom = sigma_M * D_z * D_zf
    nu_peak = delta_c / denom
    nu_eff = nu_peak * (1.0 + params['A'] * M_hm / Mvir)
    return params['alpha'] * nu_eff**params['beta'] + params['gamma']

def calculate_lwdm_half_mode_mass(cosmo_params, m_wdm_keV=1.5):
    h = cosmo_params.get('h', 0.6777)
    omega_m = cosmo_params.get('omega_m', cosmo_params.get('omega_0', 0.3071))
    omega_wdm = cosmo_params.get('omega_wdm', omega_m)
    mu = 1.12
    alpha = 0.049 * (m_wdm_keV / 1.0)**(-1.11) * (omega_wdm / 0.25)**(0.11) * (h / 0.7)**(1.22)
    T_hm = 0.5
    k_hm = (1.0 / alpha) * (T_hm**(-mu / 5.0) - 1.0)**(1.0 / (2.0 * mu))
    rho_crit = 2.775e11
    rho_m = rho_crit * omega_m
    R_hm = np.pi / k_hm
    M_hm = (4.0 * np.pi / 3.0) * rho_m * (R_hm**3)
    return M_hm, k_hm

# =========================================================
# Physics Functions (Paper II: Individual Halos & P(omega_f))
# =========================================================
def compute_omega_f(D_z, D_zf, sigma_M, sigma_M_f, delta_c=1.686):
    """Compute EPS-scaled formation variable omega_f from D(z) and D(z_f) (Paper II Eq. 3)."""
    Delta_S = np.sqrt(np.maximum(sigma_M_f**2 - sigma_M**2, 1e-30))
    omega_f = delta_c * (1.0 / D_zf - 1.0 / D_z) / Delta_S
    return np.maximum(omega_f, 0.0)

def sample_omega_f_sim(size, cosmology_type='CDM', rng=None):
    """Sample omega_f from the generalized gamma PDF P(omega_f) (Paper II Eq. 6)."""
    if rng is None:
        rng = np.random.default_rng(42)
    key = 'LWDM' if cosmology_type == 'LWDM' else 'CDM'
    p_pdf = OMEGA_F_PDF_PARAMS[key]
    w0, k, p = p_pdf['omega_0'], p_pdf['k'], p_pdf['p']
    u = rng.gamma(shape=(k + 1.0) / p, scale=1.0, size=size)
    return w0 * ((p / k) * u) ** (1.0 / p)

def sample_omega_f_eps(size, rng=None):
    """Sample omega_f from theoretical EPS p(omega_f) = 2 * omega_f * erfc(omega_f / sqrt(2))."""
    if rng is None:
        rng = np.random.default_rng(42)
    w = np.linspace(0.0, 8.0, 10000)
    s = w / np.sqrt(2.0)
    cdf = np.clip((w**2) * erfc(s) + erf(s) - np.sqrt(2.0 / np.pi) * w * np.exp(-0.5 * w**2), 0.0, 1.0)
    cdf[0], cdf[-1] = 0.0, 1.0
    inv_cdf = interp1d(cdf, w, kind='linear', bounds_error=False, fill_value=(0.0, 8.0))
    return inv_cdf(rng.uniform(0.0, 1.0, size=size))

def individual_concentration(D_z, sigma_M, omega_f, Mvir, params, M_hm=0.0):
    """Predict individual halo concentration using Paper II Equation (5) / Equation (8)."""
    D_z = np.asarray(D_z, dtype=float)
    sigma_M = np.asarray(sigma_M, dtype=float)
    omega_f = np.maximum(np.asarray(omega_f, dtype=float), 0.0)
    Mvir = np.asarray(Mvir, dtype=float)

    A_hm = 1.0 + params['a'] * (M_hm / Mvir)
    log10_c = (
        params['A']
        + params['alpha_D'] * np.log10(D_z)
        + params['alpha_sigma'] * np.log10(sigma_M)
        + params['beta'] * np.log10(1.0 + omega_f)
        + params['gamma'] * np.log10(A_hm)
    )
    return 10.0**log10_c

# =========================================================
# Main processing function
# =========================================================
def process_halo_data(input_file, cosmology_name, redshift, halo_def, relation):
    try:
        data = np.loadtxt(input_file, skiprows=1)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return None, None, None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    Mvir = data[:, 0]
    z_data = np.full_like(data[:, 1], redshift, dtype=float)
    D_z = data[:, 2]
    sigma_M = data[:, 3]
    sigma_M_f = data[:, 4]
    sigma_M_f2 = data[:, 5]

    cosmo_type = 'CDM'
    M_hm = 0.0
    if cosmology_name in cosmology_configs:
        cosmo_params = cosmology_configs[cosmology_name]
        cosmo_type = cosmo_params.get('cosmology_type', 'CDM')
        if cosmo_type == 'LWDM':
            m_keV = cosmo_params.get('m_keV', 1.5)
            M_hm, _ = calculate_lwdm_half_mode_mass(cosmo_params, m_wdm_keV=m_keV)

    # ---------------------------------------------------------
    # Paper II Branch: Individual halo c (from z_f) or c(M) distribution (via P(omega_f) MC)
    # ---------------------------------------------------------
    if relation in INDIVIDUAL_PARAMS[halo_def]:
        params = INDIVIDUAL_PARAMS[halo_def][relation]
        n_halos = len(Mvir)
        rng = np.random.default_rng(42)

        # Case A: User provided 2 columns (Mass  z_f) in input_masses.dat -> use exact D(z_f)
        if data.shape[1] >= 8:
            D_zf_sim = data[:, 7]
            omega_f_sim = compute_omega_f(D_z, D_zf_sim, sigma_M, sigma_M_f)
        # Case B: User provided only 1 column (Mass) -> Monte Carlo sample from P(omega_f)
        else:
            omega_f_sim = sample_omega_f_sim(n_halos, cosmology_type=cosmo_type, rng=rng)
            D_zf_sim = D_zf_eps_func(D_z, sigma_M, sigma_M_f, omega_f=omega_f_sim)

        omega_f_eps = sample_omega_f_eps(n_halos, rng=rng)

        # Direct Equation (5) / Equation (8) predictions
        c_sim_eq5 = individual_concentration(D_z, sigma_M, omega_f_sim, Mvir, params, M_hm=M_hm)
        c_eps_eq5 = individual_concentration(D_z, sigma_M, omega_f_eps, Mvir, params, M_hm=M_hm)

        # Monte Carlo realizations including intrinsic residual scatter sigma_res
        scatter_sim = rng.normal(0.0, params['sigma_res'], size=n_halos)
        scatter_eps = rng.normal(0.0, params['sigma_res'], size=n_halos)
        c_sim_mc = c_sim_eq5 * (10.0**scatter_sim)
        c_eps_mc = c_eps_eq5 * (10.0**scatter_eps)

        results = np.column_stack([
            Mvir,                       # 0: Mvir[Msun/h]
            sigma_M,                    # 1: sigma(M,0)
            sigma_M_f,                  # 2: sigma(M_f,0)(f=0.50)
            z_data,                     # 3: z
            D_z,                        # 4: D(z)
            omega_f_sim,                # 5: omega_f_sim (from input z_f or sampled from Eq. 6 P(omega_f))
            omega_f_eps,                # 6: omega_f_eps (sampled from EPS p(omega_f))
            D_zf_sim,                   # 7: D(z_f)_sim
            c_sim_eq5,                  # 8: c_sim_eq5 (direct Eq. 5 / Eq. 8 prediction)
            c_sim_mc,                   # 9: c_sim_mc (with intrinsic scatter sigma_res)
            c_eps_mc,                   # 10: c_eps_mc (EPS MC with sigma_res)
            np.full_like(Mvir, M_hm)    # 11: M_hm[Msun/h]
        ])
        return Mvir, D_z, results

    # ---------------------------------------------------------
    # Paper I Branch: Statistical concentration (peak, mean, median)
    # ---------------------------------------------------------
    results = np.zeros((len(Mvir), 12))
    params = MODEL_PARAMS[halo_def][relation]

    for i in range(len(Mvir)):
        Mi = Mvir[i]
        zi = z_data[i]
        Dz = D_z[i]
        sig = sigma_M[i]
        sig_f = sigma_M_f[i]
        sig_f2 = sigma_M_f2[i]

        D_zf_sim = universal_D_zf(Dz, Mi, params, M_hm=M_hm)
        D_zf_eps = D_zf_eps_func(Dz, sig, sig_f)
        D_zf_eps2 = D_zf_eps_func(Dz, sig, sig_f2)

        c_sim = universal_concentration(Dz, D_zf_sim, sig, Mi, params, M_hm=M_hm)
        c_eps = universal_concentration(Dz, D_zf_eps, sig, Mi, params, M_hm=M_hm)
        c_eps2 = universal_concentration(Dz, D_zf_eps2, sig, Mi, params, M_hm=M_hm)

        results[i] = [
            Mi,          # 0
            sig,         # 1
            sig_f,       # 2
            zi,          # 3
            Dz,          # 4
            D_zf_sim,    # 5
            D_zf_eps,    # 6
            D_zf_eps2,   # 7
            c_sim,       # 8
            c_eps,       # 9
            c_eps2,      # 10
            M_hm         # 11
        ]

    return Mvir, D_z, results

# =========================================================
# Main
# =========================================================
def main():
    if len(sys.argv) < 6:
        sys.exit(1)

    input_file = sys.argv[1]
    cosmology = sys.argv[2]
    redshift = float(sys.argv[3])
    halo_def = sys.argv[4].lower()
    relation = sys.argv[5].lower()

    Mvir, D_z, results = process_halo_data(input_file, cosmology, redshift, halo_def, relation)

    if results is None:
        sys.exit(1)

    # Separate output folders and clear filenames for Paper I vs Paper II
    if relation in INDIVIDUAL_PARAMS.get(halo_def, {}):
        os.makedirs("./output_individual_halo_properties", exist_ok=True)
        output_file = f"./output_individual_halo_properties/c_distribution_{redshift}_{cosmology}_{halo_def}_{relation}.dat"
        header = (
            "# 0:Mvir[Msun/h] "
            "1:sigma(M,0) "
            "2:sigma(M_f,0)(f=0.50) "
            "3:z "
            "4:D(z) "
            "5:omega_f_sim "
            "6:omega_f_eps "
            "7:D(z_f)_sim "
            "8:c_sim_eq5 "
            "9:c_sim_mc "
            "10:c_eps_mc "
            "11:M_hm[Msun/h]"
        )
        fmt = '%.6e %.6f %.6f %.3f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6e'

    else:
        os.makedirs("./output_halo_properties", exist_ok=True)
        output_file = f"./output_halo_properties/halo_properties_{redshift}_{cosmology}_{halo_def}_{relation}.dat"
        header = (
            "# 0:Mvir[Msun/h] "
            "1:sigma(M,0) "
            "2:sigma(M_f,0)(f=0.50) "
            "3:z "
            "4:D(z) "
            "5:D(z_f)_simfit "
            "6:D(z_f)_eps(f=0.50) "
            "7:D(z_f)_eps(f=0.16) "
            "8:c_simfit "
            "9:c_eps(f=0.50) "
            "10:c_eps(f=0.16) "
            "11:M_hm[Msun/h]"
        )
        fmt = '%.6e %.6f %.6f %.3f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6e'

    np.savetxt(
        output_file,
        results,
        fmt=fmt,
        header=header,
        comments=''
    )

if __name__ == '__main__':
    main()
