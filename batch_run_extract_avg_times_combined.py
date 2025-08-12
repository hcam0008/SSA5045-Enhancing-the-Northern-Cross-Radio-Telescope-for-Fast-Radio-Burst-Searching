#!/usr/bin/env python3
"""
Aggregate per-run average timing sheets into one Excel workbook per main directory.

This script walks a base directory (``/files_processed``) that contains many
"main" directories. Inside each main directory are multiple subdirectories,
each expected to contain a text log named ``<subdir>_output.txt`` with timing
metrics. For every subdirectory:

1) It runs ``extract_avg_times.py`` to parse the text log and write a temporary
   Excel file (``<subdir>_output.xlsx``) containing average times per metric.
2) It reads that temporary Excel file and writes the content as a separate sheet
   into a combined workbook named ``<MAIN>_combined.xlsx`` under
   ``/files_processed/avg_times_combined``.

Path conventions
----------------
- Base directory: ``/files_processed``
- Combined outputs: ``/files_processed/avg_times_combined/<MAIN>_combined.xlsx``
- Per-run input text: ``/files_processed/<MAIN>/<SUB>/<SUB>_output.txt``
- Temporary Excel per run: ``/files_processed/<MAIN>/<SUB>/<SUB>_output.xlsx``

Notes
-----
- Excel sheet names are truncated to 31 characters to satisfy Excel's limit.
- Missing input logs are skipped with a warning.
- Subprocess errors (from ``extract_avg_times.py``) will raise ``CalledProcessError``.
- The script creates the combined output directory if it does not exist.

Example
-------
Run directly to generate one combined workbook per main directory:

    python3 combine_avg_times.py
"""

import os
import subprocess
import pandas as pd

# Define the base directory where the main directories are stored
base_dir = "/files_processed"
output_dir = os.path.join(base_dir, "avg_times_combined")  # New directory for combined Excel files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each of the main directories
for main_dir in sorted(os.listdir(base_dir)):  # Sorting for consistency
    main_dir_path = os.path.join(base_dir, main_dir)

    # Ensure it's a directory
    if not os.path.isdir(main_dir_path):
        continue

    # Initialize an Excel writer to store all sheets in one file
    combined_output_file = os.path.join(output_dir, f"{main_dir}_combined.xlsx")
    with pd.ExcelWriter(combined_output_file, engine="xlsxwriter") as writer:

        # Loop through the subdirectories (tol and boxcar combinations)
        for sub_dir in sorted(os.listdir(main_dir_path)):
            sub_dir_path = os.path.join(main_dir_path, sub_dir)

            # Ensure it's a directory
            if not os.path.isdir(sub_dir_path):
                continue

            # Define the input and output file names
            input_file = os.path.join(sub_dir_path, f"{sub_dir}_output.txt")
            temp_output_file = os.path.join(sub_dir_path, f"{sub_dir}_output.xlsx")

            # Check if the input file exists
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping...")
                continue

            # Run extract_avg_times.py to generate a temporary Excel file
            command = ["./extract_avg_times.py", input_file, temp_output_file]
            subprocess.run(command, check=True)

            # Read the generated Excel file into a DataFrame
            df = pd.read_excel(temp_output_file)

            # Write to the combined Excel file as a separate sheet
            sheet_name = sub_dir[:31]  # Excel sheet names must be <=31 characters
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Created combined Excel file: {combined_output_file}")
