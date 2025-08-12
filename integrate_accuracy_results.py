#!/usr/bin/env python3
"""
Aggregate accuracy metrics across boxcar sizes and DM tolerances and export to Excel.

This script crawls an `accuracy_results/` directory for per-run text summaries,
extracts numeric metrics (MAE, MSE, RMSE, R², MAPE, Accuracy) using regular
expressions, groups them by **boxcar** size and **DM tolerance**, and produces a
single formatted Excel file per data file in `accuracy_results_final/`.

It expects two kinds of input files (all are plain text):

- **General accuracy** (Time & DM analyses in the same file):
  ``<data_file>_frb_<tolerance>_<boxcar>_accuracy_results.txt``

- **SNR accuracy**:
  ``<data_file>_frb_<tolerance>_<boxcar>_snr_accuracy_results.txt``

For general accuracy files, the script assumes that each metric appears **twice**:
first occurrence corresponds to **Time Analysis**, second to **DM Analysis**.
For SNR files, only a single set of metrics is extracted.

Output
------
For each distinct `<data_file>` it creates:
``<output_dir>/<data_file>_accuracy_formatted.xlsx``

The Excel file contains a table with rows covering:
- Time Analysis
- SNR Analysis
- DM Analysis
- Average Analysis  (mean of Time, SNR, and DM for each tolerance)

Columns include: ``Analysis Type``, ``Boxcar``, ``Metric``, followed by one
column per DM tolerance (e.g., 1.001, 1.01, 1.05, 1.1, 1.2).

Assumptions
-----------
- Boxcar sizes are one of: 32, 64, 128, 256, 512.
- DM tolerances are one of: 1.001, 1.01, 1.05, 1.1, 1.2.
- Metric lines in the text files match the regex patterns defined in ``patterns``.
- Percentage metrics may include a trailing ``%`` sign.

Notes
-----
- The script relies on :func:`clean_metric_value` to strip ``%`` and convert to
  float; unexpected or missing values are treated as ``0.0``.
- Files that do not match the expected filename convention are skipped with a log.
"""

import os
import re
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

# Define base directories
base_dir = "/path/to/your/directory/" # Amend according to your directory
accuracy_dir = os.path.join(base_dir, "accuracy_results")  # Accuracy results directory
output_dir = os.path.join(base_dir, "accuracy_results_final")  # Final formatted Excel output

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define possible boxcar values
boxcar_values = ["32", "64", "128", "256", "512"]

# Define DM tolerances
tolerance_values = ["1.001", "1.01", "1.05", "1.1", "1.2"]

# Regex patterns to extract metrics from text files
patterns = {
    "MAE": r"Mean Absolute Error\s*\(MAE\):\s*([\d\.]+)",
    "MSE": r"Mean Squared Error\s*\(MSE\):\s*([\d\.]+)",
    "RMSE": r"Root Mean Squared Error\s*\(RMSE\):\s*([\d\.]+)",
    "R^2 SCORE": r"R[²2]\s*Score:\s*([\d\.]+)",  # Handle "R²" or "R2"
    "MAPE": r"Mean Absolute Percentage Error\s*\(MAPE\):\s*([\d\.%]+)",
    "ACCURACY": r"Accuracy:\s*([\d\.%]+)"
}

# Function to clean percentage values and convert to float
def clean_metric_value(value):
    """
    Convert a metric string to float, stripping a trailing '%' if present.

    Parameters
    ----------
    value : str or float
        Metric value possibly containing a percent sign (e.g., ``'93.5%'``)
        or already numeric.

    Returns
    -------
    float
        Parsed numeric value; returns ``0.0`` if parsing fails or input is empty.
    """
    """Removes % symbols and converts string to float, returns 0 if empty or invalid."""
    if isinstance(value, str):
        value = value.replace("%", "").strip()  # Remove percentage symbols
    try:
        return float(value) if value else 0.0  # Convert to float, default to 0.0 if empty
    except ValueError:
        return 0.0  # Handle any unexpected values

# Function to extract metrics for both Time and DM
def extract_metrics(file_path):
    """
    Extract metric blocks for Time and DM analyses from a result text file.

    The function searches the file content for each metric in ``patterns`` and
    records **all** matches per metric. If two or more matches exist, the first
    match is assigned to **Time** and the second to **DM**. If only one match is
    found, it is assigned to **Time** and **DM** is set to ``'0'``. If no match
    exists, both are set to ``'0'``.

    Parameters
    ----------
    file_path : str
        Path to the text file containing metric lines.

    Returns
    -------
    (dict, dict)
        Two dictionaries: ``time_metrics`` and ``dm_metrics`` mapping metric
        name to the raw string value captured by regex.
    """
    """Extracts Time and DM analysis separately by identifying the first and second occurrence of metrics."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Identify Time and DM analysis sections
    time_metrics = {}
    dm_metrics = {}

    for metric, pattern in patterns.items():
        matches = re.findall(pattern, text)  # Get **all** occurrences of the metric
        if len(matches) >= 2:
            time_metrics[metric] = matches[0]  # First occurrence = Time Analysis
            dm_metrics[metric] = matches[1]  # Second occurrence = DM Analysis
        elif len(matches) == 1:
            time_metrics[metric] = matches[0]  # If only one match, assume it's for Time
            dm_metrics[metric] = "0"  # Default to 0 if no DM match
        else:
            time_metrics[metric] = "0"
            dm_metrics[metric] = "0"

    return time_metrics, dm_metrics

# Function to parse filename and extract key components
def parse_filename(filename):
    """
    Parse a results filename into (data_file, tolerance, boxcar, is_snr).

    The expected patterns are:
    - ``<data_file>_frb_<tolerance>_<boxcar>_accuracy_results.txt``
    - ``<data_file>_frb_<tolerance>_<boxcar>_snr_accuracy_results.txt``

    Parameters
    ----------
    filename : str
        Basename of the results text file.

    Returns
    -------
    tuple
        ``(data_file, tolerance, boxcar, is_snr)`` where:
        - data_file : str or None
        - tolerance : str or None
        - boxcar : str or None
        - is_snr : bool
        Returns ``(None, None, None, None)`` if the filename does not match.
    """
    """Extracts data file name, tolerance, boxcar, and file type (SNR or general accuracy)."""
    pattern = r"(.+)_frb_([\d.]+)_([\d]+)_(snr_)?accuracy_results\.txt"
    match = re.match(pattern, filename)
    
    if match:
        data_file = match.group(1)  # Extract data file name
        tolerance = match.group(2)  # Extract tolerance value
        boxcar = match.group(3)  # Extract boxcar size
        is_snr = bool(match.group(4))  # Check if it's an SNR file

        return data_file, tolerance, boxcar, is_snr
    
    return None, None, None, None  # Return None for invalid files

# Process each data file (Accuracy & SNR)
data_files = {}

for filename in sorted(os.listdir(accuracy_dir)):  # Sorting for consistency
    if not filename.endswith("_accuracy_results.txt") and not filename.endswith("_snr_accuracy_results.txt"):
        continue

    data_file, tolerance, boxcar, is_snr = parse_filename(filename)

    if not data_file or not tolerance or not boxcar:
        print(f"Skipping unexpected filename format: {filename}")
        continue  # Skip files that don't match the expected format

    file
