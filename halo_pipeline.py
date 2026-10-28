"""
Halo Analysis Pipeline
======================

Unified interface for the two-step halo analysis pipeline (Paper I & Paper II).

Usage: python halo_pipeline.py [cosmology] [redshift] [mode]
Modes:
  - Paper I  (statistical c -> ./output_halo_properties/):
      peak, mean, median, or all (default: peak)
  - Paper II (individual halo c & c distribution -> ./output_individual_halo_properties/):
      relaxed, all_halos, relaxed_einasto, or individual
      * If input_masses.dat has 2 columns (Mass  z_f): computes exact omega_f and c for each halo.
      * If input_masses.dat has 1 column  (Mass):      Monte Carlo samples P(omega_f) to predict c(M) distribution.
Example:
  python halo_pipeline.py planck18 0.0
  python halo_pipeline.py planck18 0.0 individual
"""
import sys
import subprocess
import os
import time
from datetime import datetime

def main():

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(__doc__)
        sys.exit(1)
    
    # halo mass file
    mass_file = "input_masses.dat"
    
    cosmology = sys.argv[1]
    redshift = sys.argv[2]
    
    mode_input = sys.argv[3].lower() if len(sys.argv) == 4 else 'peak'
    
    paper1_modes = ['peak', 'mean', 'median']
    paper2_modes = ['relaxed', 'all_halos', 'relaxed_einasto']

    if mode_input == 'all':
        modes_to_run = paper1_modes
    elif mode_input == 'individual':
        modes_to_run = paper2_modes
    elif mode_input in paper1_modes or mode_input in paper2_modes:
        modes_to_run = [mode_input]
    else:
        print(
            f"ERROR: Invalid mode '{mode_input}'. "
            "Choose from: peak, mean, median, all, relaxed, all_halos, relaxed_einasto, individual"
        )
        sys.exit(1)
        
    # halo identification
    halo_defs = ['200c', 'vir']
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 60)
    print("HALO ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Start time: {start_datetime}")
    print(f"Input Data: {mass_file}")
    print(f"Cosmology:  {cosmology}")
    print(f"Redshift:   z = {redshift}")
    print(f"Modes:      {', '.join(modes_to_run)}")
    print(f"Halo Defs:  {', '.join(halo_defs)}")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # Step 1: Compute halo mass variance
    # ---------------------------------------------------------
    print("\n[STEP 1/2] Computing halo mass variance...")
    cmd1 = ["python", "halo_variance.py", mass_file, cosmology, redshift]
    print(f"Running: {' '.join(cmd1)}")
    
    step1_start = time.time()
    result1 = subprocess.run(cmd1)
    step1_time = time.time() - step1_start
    
    if result1.returncode != 0:
        print("ERROR: Step 1 failed")
        sys.exit(1)
    
    if not os.path.exists("output_variance.dat"):
        print("ERROR: output_variance.dat not generated")
        sys.exit(1)
    
    print(f"Step 1 completed ({step1_time:.1f} seconds)")
    
    # ---------------------------------------------------------
    # Step 2: Compute halo concentrations
    # ---------------------------------------------------------
    print("\n[STEP 2/2] Computing halo concentrations...")
    
    step2_start = time.time()
    generated_files = []
    
    for current_mode in modes_to_run:
        # relaxed_einasto is calibrated specifically under the 200c definition (Paper II Eq. 8)
        defs_for_mode = ['200c'] if current_mode == 'relaxed_einasto' else halo_defs
        for current_def in defs_for_mode:
            print(f"\n--- Processing: Mode = {current_mode.upper()}, Def = {current_def.upper()} ---")
            
            cmd2 = ["python", "halo_model.py", "output_variance.dat", cosmology, redshift, current_def, current_mode]
            print(f"Running: {' '.join(cmd2)}")
            
            result2 = subprocess.run(cmd2)
            
            if result2.returncode != 0:
                print(f"ERROR: Step 2 failed for {current_def} {current_mode}")
                sys.exit(1)
                
            if current_mode in paper2_modes:
                expected_output = f"./output_individual_halo_properties/c_distribution_{float(redshift)}_{cosmology}_{current_def}_{current_mode}.dat"
            else:
                expected_output = f"./output_halo_properties/halo_properties_{float(redshift)}_{cosmology}_{current_def}_{current_mode}.dat"

            if os.path.exists(expected_output):
                generated_files.append(expected_output)
            else:
                print(f"WARNING: Expected output {expected_output} not found.")
                
    step2_time = time.time() - step2_start
    total_time = time.time() - start_time
    
    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    if generated_files:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Results saved to:")
        for f in generated_files:
            file_size = os.path.getsize(f)
            print(f"  - {f} ({file_size:,} bytes)")
        
        print("\nTIMING SUMMARY:")
        print(f"Step 1 (variance):      {step1_time:.1f} seconds")
        print(f"Step 2 (concentration): {step2_time:.1f} seconds")
        print(f"Total time:             {total_time:.1f} seconds")
        print("=" * 60)
    else:
        print("ERROR: Could not find any output files")
        print(f"Total time elapsed: {total_time:.1f} seconds")
        sys.exit(1)

if __name__ == "__main__":
    main()
