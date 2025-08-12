# run_make_dataframe.py
"""
Discover candidate files, build per-file DataFrames, and export a combined Excel.

This script:
1) discovers benchmark directories under ``files_processed`` whose names start
   with ``CUT_`` (e.g., ``CUT_D_110_inj_13_processed``),
2) derives a benchmark CSV path in ``simulated_frbs`` by removing the
   ``_processed`` suffix and (if present) the trailing ``_13`` (e.g.,
   ``CUT_D_110_inj`` → ``simulated_frbs/CUT_D_110_inj_params.csv``),
3) scans ``files_processed/combined_cand`` for ``.cand`` files whose basenames
   start with ``beam_<benchmark_name>``, and
4) for each match calls ``make_dataframe.process_data_file`` to produce a tidy
   DataFrame with accuracy/runtime/metadata. All non-empty DataFrames are then
   concatenated and saved to a single Excel file.

Inputs & Path Conventions
-------------------------
- Data root: ``files_processed/`` (contains benchmark directories and
  ``combined_cand/`` with merged ``.cand`` files).
- Benchmark CSVs: ``simulated_frbs/<benchmark_name>_params.csv`` where
  ``benchmark_name`` is derived from each benchmark directory name.

Outputs
-------
- Combined Excel workbook at:
  ``files_processed/files_dataframe.xlsx``

Notes
-----
- The helper function :func:`make_dataframe.process_data_file` is imported from
  ``python_scripts/make_dataframe.py`` (path added to ``sys.path`` at runtime).
- ``combined_cand`` must contain files named like:
  ``beam_<benchmark_name>_frb_<TOL>_<BOXCAR>.cand``.
- Files without a matching benchmark CSV are skipped with a message.
- Empty DataFrames are filtered out before concatenation.

Example
-------
Run directly from the project root:

    python3 run_make_dataframe.py
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "python_scripts"))
from make_dataframe import process_data_file

data_root = "files_processed"
benchmark_root = "simulated_frbs"

all_dataframes = []

# Automatically discover benchmark directories
benchmark_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("CUT_")]

for benchmark_id in benchmark_dirs:
    benchmark_name = benchmark_id.replace("_processed", "")
    if benchmark_name.endswith("_13"):
        benchmark_name = benchmark_name[:-3]
    benchmark_file = os.path.join(benchmark_root, f"{benchmark_name}_params.csv")
    if not os.path.exists(benchmark_file):
        print(f"Benchmark file missing: {benchmark_file}")
        continue

    file_dir = os.path.join(data_root, benchmark_id)

    # Look in combined_cand folder for matching .cand files
    combined_dir = os.path.join(data_root, "combined_cand")
    for file in os.listdir(combined_dir):
        if not file.endswith(".cand"):
            continue
        if not file.startswith(f"beam_{benchmark_name}"):
            continue

        cand_path = os.path.join(combined_dir, file)
        if not os.path.exists(cand_path):
            print(f".cand file not found: {cand_path}")
            continue

        try:
            df = process_data_file(cand_path, benchmark_file)
            all_dataframes.append(df)
            #print(f"Processed: {os.path.basename(cand_path)} → {len(df)} rows")
        except Exception as e:
            print(f"Error processing {cand_path}: {e}")

# Remove empty DataFrames before concat
valid_dataframes = [df for df in all_dataframes if not df.empty]
if valid_dataframes:
    combined_df = pd.concat(valid_dataframes, ignore_index=True)
    combined_df.to_excel("files_processed/files_dataframe.xlsx", index=False)
    print("All results saved to files_processed/files_dataframe.xlsx")
    print(f"Total files processed: {len(valid_dataframes)}")
else:
    print("No data was processed.")
