#!/usr/bin/env python3
"""
Sum all 'Total time' entries from a Heimdall (or similar) log and write the total to Excel.

This script scans a text log for lines like ``"Total time: 0.123"`` using a
regular expression, extracts all numeric values, sums them to produce an overall
runtime (in seconds), and writes a small two-column table to an Excel file.

Command-line usage
------------------
input_file : str
    Path to the input text log to parse.
output_file : str
    Path to the output Excel file (e.g., ``total_runtime.xlsx``).

Example
-------
    python sum_total_time.py run_log.txt total_runtime.xlsx
"""

import re
import pandas as pd
import argparse
import sys

def extract_total_times(input_file, output_file):
    """
    Parse all 'Total time' entries from a text file, sum them, and save to Excel.

    The function searches for occurrences matching the pattern
    ``"Total time: <float>"``, converts all captured values to floats,
    computes their sum, and writes a table with two columns:
    ``Metric`` and ``Value`` (seconds).

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
        If the input file is missing or if no 'Total time' entries are found.

    Notes
    -----
    The regular expression used is: ``r"Total time:\\s+([0-9.]+)"`` which
    captures the numeric part (integer or decimal) following the label.
    """
    # Load the text file
    try:
        with open(input_file, "r") as file:
            data = file.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Regular expression to match each time entry and its value
    pattern = r"Total time:\s+([0-9.]+)"
    
    # Extract all matches
    matches = re.findall(pattern, data)
    
    if not matches:
        print("No total time entries found in the input file.")
        sys.exit(1)

    # Convert to floats and sum
    total_times = [float(t) for t in matches]
    total_runtime = sum(total_times)

    # Convert to a DataFrame
    df = pd.DataFrame({
        "Metric": ["Total runtime (s)"],
        "Value": [total_runtime]
    })

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Total runtimes have been saved to {output_file}.")

def main():
    """
    CLI entry point.

    Parses command-line arguments and calls :func:`extract_total_times`.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Sum all 'Total time' entries from Heimdall log and save the total runtime.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument("output_file", help="Path to the output Excel file.")
    args = parser.parse_args()

    extract_total_times(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
