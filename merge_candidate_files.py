#!/usr/bin/env python3
"""
Concatenate *.cand files from a directory tree into per-subdir combined files.

This script walks a base directory that contains many "main" directories. 
Inside each main directory are multiple subdirectories. 
For every subdirectory, it concatenates all
``*.cand`` files into a single combined file placed under a new
``combined_cand/`` folder within the base directory.

Layout (example)
----------------
/files_processed/
  ├─ MAIN_000/
  │   ├─ SUB_000/
  │   │   ├─ a.cand
  │   │   ├─ b.cand
  │   │   └─ ...
  │   └─ SUB_001/
  │       └─ *.cand
  └─ MAIN_001/
      └─ SUB_000/
          └─ *.cand

For each (MAIN, SUB) pair, an output file is created:
``/files_processed/combined_cand/beam_<MAIN>_<SUB>.cand``

Assumptions & Notes
-------------------
- The script uses the shell command ``cat`` (POSIX systems). On Windows you may
  need to adapt to ``type`` or use Python file I/O.
- Existing output files with the same name will be overwritten by the shell redirection.
- Non-directories within the tree are skipped.
- Errors from the shell command raise a ``CalledProcessError`` due to ``check=True``.

Outputs
-------
- One combined ``.cand`` file per subdirectory under:
  ``/files_processed/combined_cand/beam_<MAIN>_<SUB>.cand``
"""

import os
import subprocess

# Define the base directory where the main directories are stored
base_dir = "/files_processed"
output_dir = os.path.join(base_dir, "combined_cand")  # New directory for combined files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each of the main directories
for main_dir in sorted(os.listdir(base_dir)):  # Sorting for consistency
    main_dir_path = os.path.join(base_dir, main_dir)

    # Ensure it's a directory
    if not os.path.isdir(main_dir_path):
        continue

    # Loop through the subdirectories
    for sub_dir in sorted(os.listdir(main_dir_path)):  # Sorting for consistency
        sub_dir_path = os.path.join(main_dir_path, sub_dir)

        # Ensure it's a directory
        if not os.path.isdir(sub_dir_path):
            continue

        # Define a unique output file name in the new combined_cand directory
        output_file = os.path.join(output_dir, f"beam_{main_dir}_{sub_dir}.cand")

        # Run the cat command to concatenate files
        command = f"cat {sub_dir_path}/*.cand > {output_file}"
        subprocess.run(command, shell=True, check=True)

        # Print status update
        print(f"Processed: {output_file}")
