#!/usr/bin/env python3
"""
Batch-run accuracy calculations for each combined `.cand` file.

This script scans `/files_processed/combined_cand` for files named like
`beam_<DATA_FILE>_frb_<DM_TOL>_<BOXCAR>.cand`, constructs the corresponding
benchmark paths, and invokes `python_scripts/accuracy_calc.py` to compute
accuracy metrics (Time, DM, SNR) per parameter combination.

For each input `.cand`, two outputs are written under
`/files_processed/accuracy_results/`:

- `<DATA_FILE>_frb_<DM_TOL>_<BOXCAR>_accuracy_results.txt`
- `<DATA_FILE>_frb_<DM_TOL>_<BOXCAR>_snr_accuracy_results.txt`

Inputs & Path Conventions
-------------------------
- Base directory: ``/files_processed``
- Combined candidates: ``/files_processed/combined_cand`` (input `.cand` files)
- Accuracy outputs: ``/files_processed/accuracy_results`` (created if missing)
- Benchmark CSV: derived as
  ``/simulated_frbs/<DATA_FILE (without '_inj_13_processed')>_inj_params.csv``
- SNR benchmark: fixed to the combined file with tolerance=``1.001`` and
  boxcar=``512``, i.e.:
  ``/files_processed/combined_cand/beam_<DATA_FILE>_frb_1.001_512.cand``

Behavior & Notes
----------------
- Files whose names do not match the expected ``..._frb_<TOL>_<BOXCAR>.cand``
  pattern are skipped with a message.
- The script calls: ``python3 python_scripts/accuracy_calc.py`` with arguments:
  ``-i <cand>``, ``-b <benchmark.csv>``, ``-o <accuracy.txt>``,
  ``-s <snr_benchmark.cand>``, ``-so <snr_accuracy.txt>``.
- Subprocess errors will raise ``CalledProcessError`` due to ``check=True``.

Example
-------
Run the script directly to process all combined candidates:

    python3 run_accuracy_batch.py
"""

import os
import subprocess

# Define base directories
base_dir = "/files_processed"
cand_dir = os.path.join(base_dir, "combined_cand")  # Directory containing .cand files
output_dir = os.path.join(base_dir, "accuracy_results")  # Directory for accuracy results

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each .cand file in combined_cand/
for cand_file in sorted(os.listdir(cand_dir)):  # Sorting for consistency
    if not cand_file.endswith(".cand"):
        continue  # Skip non-cand files

    # Extract the main file name and parameter combination
    parts = cand_file.replace("beam_", "").replace(".cand", "").split("_frb_")
    
    if len(parts) != 2:
        print(f"Skipping unexpected file format: {cand_file}")
        continue

    data_file, params = parts

    # Define full paths
    input_file = os.path.join(cand_dir, cand_file)
    benchmark_file = f"/simulated_frbs/{data_file.replace('_inj_13_processed', '')}_inj_params.csv"
    snr_benchmark_file = os.path.join(cand_dir, f"beam_{data_file}_frb_1.001_512.cand")  # SNR benchmark file
    output_file = os.path.join(output_dir, f"{data_file}_frb_{params}_accuracy_results.txt")
    snr_output_file = os.path.join(output_dir, f"{data_file}_frb_{params}_snr_accuracy_results.txt")

    # Check if the input cand file exists
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue

    # Run accuracy_calc.py
    command = [
        "python3", "python_scripts/accuracy_calc.py",
        "-i", input_file,
        "-b", benchmark_file,
        "-o", output_file,
        "-s", snr_benchmark_file,
        "-so", snr_output_file
    ]

    subprocess.run(command, check=True)

    # Print status update
    print(f"Processed {input_file} -> {output_file}")
