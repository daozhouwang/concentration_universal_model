"""
Halo Mass Variance Calculator with Adaptive Table Bounds
Usage: python halo_variance.py [mass_file.dat] [cosmology] [redshift]

Note:
- Case 1 (1 column in mass_file.dat: Mass):
  Computes standard 6-column variance table (used by Paper I and Paper II Monte Carlo P(omega_f) sampling).
- Case 2 (2 columns in mass_file.dat: Mass  z_f):
  Also computes D(z_f) for each halo's formation redshift z_f and saves columns 6 (z_f) and 7 (D(z_f))
  in output_variance.dat (used by Paper II to predict individual halo c from known z_f).
"""
import numpy as np
import sys
import os
from scipy.interpolate import interp1d
from VarianceCalculator import HaloMassVarianceCalculator
from GrowthFactor import linear_growth
from cosmology_params import cosmology_configs

def load_input_data(filename):
    """Load halo masses (and optional 2nd column: formation redshift z_f) from input file"""
    data = np.loadtxt(filename)
    if data.ndim == 0:
        return np.array([float(data)]), None
    if data.ndim == 1:
        return data, None
    return data[:, 0], data[:, 1]
    
def save_results(results, redshift, cosmology, z_f=None, D_zf=None, filename="output_variance.dat"):
    """Save results to file with mixed formatting"""

    n_halos = len(results['M'])
    redshifts_column = np.full(n_halos, redshift)
    
    if z_f is None:
        output_data = np.column_stack([
            results['M'], 
            redshifts_column,  
            results['D_z'], 
            results['sigma_M'], 
            results['sigma_M_f'], 
            results['sigma_M_f2']
        ])
        fmt = ['%.6e', '%.2f', '%.6f', '%.6f', '%.6f', '%.6f']
        header = 'Mass[M_sun/h] z D(z) sigma(M,0) sigma(0.5M,0) sigma(0.16M,0)'
    else:
        output_data = np.column_stack([
            results['M'], 
            redshifts_column,  
            results['D_z'], 
            results['sigma_M'], 
            results['sigma_M_f'], 
            results['sigma_M_f2'],
            z_f,
            D_zf
        ])
        fmt = ['%.6e', '%.2f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f']
        header = 'Mass[M_sun/h] z D(z) sigma(M,0) sigma(0.5M,0) sigma(0.16M,0) z_f D(z_f)'
    
    np.savetxt(
        filename,
        output_data,
        fmt=fmt,
        header=header,
        comments='')
    
# Parse command line arguments
mass_file = sys.argv[1] if len(sys.argv) > 1 else "input_masses.dat"
cosmology = sys.argv[2] if len(sys.argv) > 2 else 'LCDM'
redshift = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
# Validate inputs
if cosmology not in cosmology_configs:
    print(f"Error: Cosmology '{cosmology}' not supported")
    sys.exit(1)
if not os.path.exists(mass_file):
    print(f"Error: Mass file {mass_file} not found")
    sys.exit(1)

# Load halo masses (and optional 2nd column z_f)
masses, z_f = load_input_data(mass_file)
print(f"Loaded {len(masses)} halos from {mass_file}")
print(f"Mass range: {np.min(masses):.3e} - {np.max(masses):.3e} M_sun/h")
print(f"Cosmology: {cosmology}, Redshift: z = {redshift}")

# Initialize calculator with auto-adjustment
calculator = HaloMassVarianceCalculator(
    cosmology_name=cosmology,
    auto_adjust=True,   
    safety_factor=0.1,   
    table_size=1000    
)

# Calculate mass variance (will auto-initialize table)
results = calculator.calculate_variance(masses, redshift)

# If formation redshift z_f (2nd column) is provided, compute D(z_f) for each halo
D_zf = None
if z_f is not None:
    cosmo = cosmology_configs[cosmology]
    z_f_safe = np.clip(np.asarray(z_f, dtype=float), 0.0, 50.0)
    z_grid = np.linspace(0.0, max(float(np.max(z_f_safe)) + 0.5, 1.0), 500)
    D_grid, _ = linear_growth(
        z_grid,
        h=cosmo['h'],
        omega_0=cosmo['omega_m'],
        w_0=cosmo['w_0'],
        w_a=cosmo['w_a'],
        cosmology=cosmo['cosmology_type']
    )
    D_interp = interp1d(z_grid, D_grid, kind='cubic', bounds_error=False, fill_value='extrapolate')
    D_zf = D_interp(z_f_safe)

# Preview results
print("\nPreview (first 5 halos):")
print("Mass [M_sun/h]   z    D(z)  sigma(M,0)  sigma(0.5M,0) sigma(0.16M,0)")
print("-" * 50)
for i in range(min(5, len(masses))):
    print(f"{results['M'][i]:12.2e}  {redshift} {results['D_z'][i]:8.4f}  "
          f"{results['sigma_M'][i]:8.4f}  {results['sigma_M_f'][i]:8.4f} "
          f"{results['sigma_M_f2'][i]:8.4f}")
# Save full results
save_results(results, redshift, cosmology, z_f=z_f, D_zf=D_zf, filename="output_variance.dat")
print(f"\nFull results saved to output_variance.dat")
