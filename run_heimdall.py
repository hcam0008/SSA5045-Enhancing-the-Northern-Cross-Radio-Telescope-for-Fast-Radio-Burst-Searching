#!/usr/bin/env python3
"""
Batch-run Heimdall over multiple .fil files and parameter grids, saving logs.

This script reads a list of CSV paths from
``/files_processed/latest_files_log.txt`` (one path per line). For each CSV, it
derives the corresponding filterbank file by replacing the suffix
``_params.csv`` with ``_13_processed.fil`` and then runs ``heimdall`` across a
grid of dispersion-measure tolerances (``tols``) and maximum boxcar sizes
(``boxcars``). For every (tol, boxcar) pair, the script captures stdout/stderr
and writes them to a log file within a structured output directory.

Parameters (hard-coded here)
----------------------------
tols : list[float]
    DM tolerance values to test, e.g. ``[1.001, 1.01, 1.05, 1.1, 1.2]``.
boxcars : list[int]
    Maximum boxcar widths to test, e.g. ``[32, 64, 128, 256, 512]``.
base_dir : str
    Base directory for outputs: ``/files_processed/``.
file_list_path : str
    Text file containing CSV paths (one per line):
    ``/files_processed/latest_files_log.txt``.

Path conventions
----------------
- For each input CSV path ``.../<NAME>_params.csv`` the script expects
  ``.../<NAME>_13_processed.fil`` to exist.
- Outputs for each file are organized under:
  ``/files_processed/<NAME>_13_processed/frb_<TOL>_<BOXCAR>/``
  with a log file:
  ``frb_<TOL>_<BOXCAR>_output.txt``.

Heimdall invocation (fixed flags shown)
---------------------------------------
- ``-f <.fil>`` input filterbank
- ``-output_dir <DIR>`` per-parameter output directory
- ``-dm 0 3000.0`` DM range
- ``-dm_tol <TOL>`` from the grid
- ``-boxcar_max <BOXCAR>`` from the grid
- ``-gpu_id 0`` GPU index (adjust as needed)
- ``-V`` verbose mode

Notes
-----
- Missing CSV or derived ``.fil`` files are skipped with a warning.
- Both stdout and stderr are saved for diagnostics.
- Adjust DM range, GPU id, and paths to suit your environment.

Example
-------
Run directly to process all files listed in ``latest_files_log.txt``:

    python3 run_grid_heimdall.py
"""

import os
import subprocess

# Parameters
tols = [1.001, 1.01, 1.05, 1.1, 1.2]
boxcars = [32, 64, 128, 256, 512]

# Adjust the base directory accordingly
base_dir = "/files_processed/"

# Path to the text file containing the list of files
file_list_path = "/files_processed/latest_files_log.txt"


# Read the list of filenames from the text file
with open(file_list_path, "r") as file:
    # csv_paths = [os.path.abspath(os.path.join("../", line.strip())) for line in file.readlines() if line.strip()]
    csv_paths = [os.path.abspath(line.strip().lstrip("../")) for line in file.readlines() if line.strip()]
    print(csv_paths)

# Loop through each CSV file and derive the corresponding .fil file
for csv_path in csv_paths:
    # Ensure the CSV file exists (optional check)
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found. Skipping.")
        continue

    # Get the base directory of the input files
    input_dir = os.path.dirname(csv_path)

    # Extract the base filename without extension
    csv_name = os.path.basename(csv_path)

    # Replace `_params.csv` with `_13_processed.fil` to get the corresponding .fil file
    fil_name = csv_name.replace("_params.csv", "_13_processed.fil")
    fil_path = os.path.join(input_dir, fil_name)

    # Ensure the corresponding .fil file exists before processing
    if not os.path.exists(fil_path):
        print(f"Warning: Expected .fil file {fil_path} not found. Skipping.")
        continue

    # Extract the filename (without extension) to organize output folders
    file_name = os.path.basename(fil_path).replace(".fil", "")

    # Create a base output directory specific to this file
    file_output_dir = os.path.join(base_dir, file_name)

    # Loop through each combination of tol and boxcar
    for tol in tols:
        for box in boxcars:
            # Create a unique output directory for each tol and boxcar combination
            output_dir = os.path.join(file_output_dir, f"frb_{tol}_{box}")
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

            # Build the heimdall command
            command = [
                "heimdall",
                "-f", fil_path,  # Use the derived .fil file
                "-output_dir", output_dir,  # Use the proper output directory
                "-dm", "0", "3000.0",  # Adjust DM range accordingly
                "-dm_tol", str(tol),
                "-boxcar_max", str(box),
                "-gpu_id", "0",  # Adjust GPU ID accordingly
                "-V"  # Adjust verbosity accordingly
            ]

            # Execute the command and capture the output
            result = subprocess.run(command, capture_output=True, text=True)

            # Save the output to a log file in the corresponding directory
            log_file_path = os.path.join(output_dir, f"frb_{tol}_{box}_output.txt")
            with open(log_file_path, "w") as log_file:
                log_file.write(result.stdout)
                log_file.write(result.stderr)  # Save stderr as well, in case of errors

            print(f"Finished processing {file_name} with tol={tol}, boxcar={box}. Output saved in {output_dir}.")
