#!/usr/bin/env python3
"""
Extract runtime metrics from a text log and write per-metric averages to Excel.

This script scans a text file for timing lines such as
``"Mem alloc time: 0.123"`` using a regular expression, aggregates values by
metric name, excludes the first and last occurrence for each metric (to reduce
warm-up/tear-down effects) when there are more than two observations, computes
the mean, and saves the results to an Excel file.

Supported metric labels
-----------------------
- Mem alloc time
- 0-DM cleaning time
- Dedispersion time
- Copy time
- Baselining time
- Normalisation time
- Filtering time
- Find giants time
- Process candidates time
- Total time

Command-line usage
------------------
input_file : str
    Path to the input text log containing timing lines.
output_file : str
    Path to the output Excel file (e.g., ``averages.xlsx``).

Examples
--------
Compute averages and write to ``averages.xlsx``:

    python extract_avg_times.py run_log.txt averages.xlsx
"""

import re
import pandas as pd
import argparse
import sys

def extract_average_times(input_file, output_file):
    """
    Parse timing metrics from a text file and write per-metric averages to Excel.

    The function searches for lines matching the pattern
    ``"<Metric name>: <float>"`` for a fixed set of metric names, converts the
    captured values to floats, groups by metric, excludes the first and last
    values for each metric when there are more than two samples, computes the
    mean, and saves a two-column table (``Metric``, ``Time``) to an Excel file.

    Parameters
    ----------
    input_file : str
        Path to the input text file to parse.
    output_file : str
        Path to the output Excel file to create.

    Returns
    -------
    None
        Results are written to ``output_file`` and a status message is printed.

    Raises
    ------
    SystemExit
        If the input file is missing or if no matching metrics are found.

    Notes
    -----
    - The regular expression matches these metric labels:
      ``Mem alloc time``, ``0-DM cleaning time``, ``Dedispersion time``,
      ``Copy time``, ``Baselining time``, ``Normalisation time``,
      ``Filtering time``, ``Find giants time``, ``Process candidates time``,
      ``Total time``.
    - If a metric has two or fewer samples, its mean is computed over all
      available values (no exclusion).
    """
    # Load the text file
    try:
        with open(input_file, "r") as file:
            data = file.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Regular expression to match each time entry and its value
    pattern = r"(Mem alloc time|0-DM cleaning time|Dedispersion time|Copy time|Baselining time|Normalisation time|Filtering time|Find giants time|Process candidates time|Total time):\s+([0-9.]+)"
    
    # Extract all matches
    matches = re.findall(pattern, data)
    
    if not matches:
        print("No matching time metrics found in the input file.")
        sys.exit(1)

    # Convert to a DataFrame
    df = pd.DataFrame(matches, columns=["Metric", "Time"])
    df["Time"] = df["Time"].astype(float)

    # Group by metric and remove the first and last values before calculating the mean
    def exclude_first_last(group):
        """Return the mean excluding the first and last entries if len(group) > 2."""
        if len(group) > 2:
            return group.iloc[1:-1].mean()  # Exclude first and last values
        else:
            return group.mean()  # If there are 2 or fewer values, just take the mean

    avg_times = df.groupby("Metric")["Time"].apply(exclude_first_last).reset_index()

    # Save to Excel
    avg_times.to_excel(output_file, index=False)
    print(f"Average times have been saved to {output_file}.")

def main():
    """
    CLI entry point.

    Parses command-line arguments and invokes :func:`extract_average_times`.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Extract time metrics from a text file and calculate their averages, excluding the first and last values for each metric.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument("output_file", help="Path to the output Excel file.")
    args = parser.parse_args()

    extract_average_times(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
