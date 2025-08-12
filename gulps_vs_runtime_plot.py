#!/usr/bin/env python3
"""
Benchmark Heimdall runtime versus different gulp (nsamps) sizes on a single .fil.

This script iterates over a list of gulp sizes (``-nsamps_gulp``) and invokes
``heimdall`` on a single filterbank file, saving the standard output and error
for each run into text logs under a dedicated output directory.

Inputs
------
- fil_path : str
    Absolute path to the input SIGPROC ``.fil`` file. (Hard-coded in the script.)
- num_samples : list[int]
    Sequence of gulp sizes to test (in samples). (Hard-coded in the script.)

Outputs
-------
- One log file per gulp size written to:
  ``/runtime_vs_gulps/filterbank_file_output/gulp_<NSAMPS>.txt``

Heimdall parameters (fixed here)
--------------------------------
- ``-dm 0 3000.0`` : DM range
- ``-dm_tol 1.01`` : chosen tolerance
- ``-boxcar_max 256`` : chosen maximum boxcar width
- ``-gpu_id 0`` : GPU index
- ``-V`` : verbose output

Notes
-----
- The script raises ``FileNotFoundError`` if ``fil_path`` does not exist.
- Adjust DM range, tolerance, boxcar limit, and GPU ID as required for your setup.
- Both stdout and stderr from Heimdall are saved for diagnostics.
"""

import os
import subprocess

# gulps sizes
num_samples = [
    65536,         # just above minimum requirement
    98304,         # 1.5x
    131072,        # 2x
    196608,        # 3x
    262144,        # 4x
    327680,        # 5x
    393216,        # 6x
    458752,        # 7x
    524288,        # 8x
    589824,        # 9x
    655360,        # 10x
    720896,        # 11x
    786432,        # 12x
    851968,        # 13x
    917504,        # 14x
    983040         # 15x
]

# Path to the single .fil file
fil_path = "/filterbank_file.fil"

# Check the file exists
if not os.path.exists(fil_path):
    raise FileNotFoundError(f"{fil_path} does not exist.")

# Output directory
output_base_dir = "/runtime_vs_gulps/filterbank_file_output"
os.makedirs(output_base_dir, exist_ok=True)

# Extract filename (without .fil extension)
file_name = os.path.basename(fil_path).replace(".fil", "")

# Loop through each combination of tol and boxcar
for gulp in num_samples:
    # Build the heimdall command
    command = [
        "heimdall",
        "-f", fil_path,  # Use the derived .fil file
        "-output_dir", output_base_dir,  # Use the proper output directory
        "-dm", "0", "3000.0",  # Adjust DM range accordingly
        "-dm_tol", "1.01", # Optimal chosen
        "-boxcar_max", "256", # Optimal chosen
        "-nsamps_gulp", str(gulp), # gulp size
        "-gpu_id", "0",  # Adjust GPU ID accordingly
        "-V"  # Adjust verbosity accordingly
        ]
 
    # Execute the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)

    # Save the output to a log file in the corresponding directory
    log_file_path = os.path.join(output_base_dir, f"gulp_{gulp}.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(result.stdout)
        log_file.write(result.stderr)  # Save stderr as well, in case of errors

    print(f"Finished processing {file_name} with sample size = {gulp}. Output saved in {output_base_dir}.")
