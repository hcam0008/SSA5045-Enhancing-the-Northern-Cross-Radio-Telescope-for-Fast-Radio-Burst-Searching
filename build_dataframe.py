# make_dataframe.py
"""
Build a per-file summary DataFrame of detections and related accuracy/runtime.

This module ingests a measured detections file (tab-separated with no header),
aligns/renames columns, derives additional fields (e.g., pulse width), selects
the top 13 rows by SNR and sorts by time, and augments each row with:
- percentage accuracies (SNR, DM, time) parsed from companion accuracy text files
- runtime extracted from a companion Excel output
- metadata parsed from the filename (DM tolerance, boxcar width)

It provides helpers to extract parameters from filenames, read accuracy values
from text artifacts, read total runtime from an Excel artifact, and compute a
final DataFrame ready for analysis/aggregation.

Notes
-----
- Accuracy files are expected in:
  ``/files_processed/accuracy_results``
- The function :func:`process_data_file` asserts that the final table has 13 rows.
"""

import pandas as pd
import os
import re

column_names = ["SNR", "sample number", "time", "filter", "DM trial number",
                "DM", "members", "first sample", "last sample"]

def extract_dm_tol_boxcar(filename):
    """
    Extract DM tolerance and boxcar width from a filename.

    Uses the pattern ``r"frb_([0-9.]+)_([0-9]+)"`` to capture:
    - group 1: DM tolerance (converted to ``float``)
    - group 2: boxcar width (converted to ``int``)

    Parameters
    ----------
    filename : str
        Basename or full path of the data file.

    Returns
    -------
    tuple of (float or None, int or None)
        ``(dm_tol, boxcar_width)``; returns ``(None, None)`` if no match.
    """
    pattern = r"frb_([0-9.]+)_([0-9]+)"
    match = re.search(pattern, filename)
    if match:
        dm_tol = float(match.group(1))
        boxcar_width = int(match.group(2))
        return dm_tol, boxcar_width
    return None, None

def calculate_percentage_accuracy(detected, benchmark):
    """
    Compute percentage accuracy relative to a benchmark value.

    The accuracy is defined as:
    ``100 * (1 - abs((detected / (1000 ** 2)) - benchmark) / benchmark)``

    Parameters
    ----------
    detected : float
        Measured value (will be divided by ``1000**2`` before comparison).
    benchmark : float
        Benchmark/ground-truth value.

    Returns
    -------
    float or None
        Percentage accuracy rounded to two decimals, or ``None`` if
        division by zero occurs.
    """
    try:
        return round(100 * (1 - abs((detected / (1000 ** 2)) - benchmark) / benchmark), 2)
    except ZeroDivisionError:
        return None

def read_accuracy_from_file(base_filename):
    """
    Read SNR, DM, and time percentage accuracies from companion text files.

    For a given data file path, constructs companion accuracy paths in
    ``/files_processed/accuracy_results`` by removing a
    leading ``beam_`` from the stem and appending:
      - ``_accuracy_results.txt`` (contains Time and DM sections)
      - ``_snr_accuracy_results.txt`` (contains SNR section)

    Each accuracy text file is scanned for a section header of the form
    ``--- {VAR} ANALYSIS ---`` followed by a line containing ``accuracy: <val>%``.
    The numeric percentage is returned as a float.

    Parameters
    ----------
    base_filename : str
        Original data filename used to derive accuracy file paths.

    Returns
    -------
    tuple of (float or None, float or None, float or None)
        ``(acc_snr, acc_dm, acc_time)`` — any component may be ``None`` if
        the file/section/line is missing or cannot be parsed.

    Notes
    -----
    - This function prints diagnostic messages when files or sections
      are missing or malformed.
    - It does not raise on parsing errors; it returns ``None`` instead.
    """
    accuracy_dir = "/files_processed/accuracy_results"
    full_base = os.path.splitext(os.path.basename(base_filename))[0]
    base_name = full_base.replace("beam_", "")  # remove beam_ prefix if present
    acc_file = os.path.join(accuracy_dir, f"{base_name}_accuracy_results.txt")
    snr_file = os.path.join(accuracy_dir, f"{base_name}_snr_accuracy_results.txt")

    def extract_accuracy(path, var):
        """
        Extract the numeric accuracy for a given variable from an accuracy file.

        Parameters
        ----------
        path : str
            Path to the accuracy text file.
        var : str
            Variable name ('time', 'dm', 'snr') used to locate the section.

        Returns
        -------
        float or None
            Parsed accuracy percentage, or ``None`` if not found/parsed.
        """
        if not os.path.exists(path):
            print(f"Accuracy file not found: {path}")
            return None
        with open(path, "r") as f:
            lines = f.readlines()
            found_section = False
            for line in lines:
                if f"--- {var.upper()} ANALYSIS ---" in line.upper():
                    found_section = True
                    print(f"Found section for {var} in {path}")
                elif found_section and "accuracy" in line.lower():
                    try:
                        print(f"Extracting accuracy line: {line.strip()}")
                        return float(line.strip().split(":")[-1].replace('%', '').strip())
                    except Exception as e:
                        print(f"Failed to parse accuracy line: {line.strip()} → {e}")
                        return None
        print(f"Accuracy for {var} not found in {path}")
        return None

    acc_time = extract_accuracy(acc_file, "time")
    acc_dm = extract_accuracy(acc_file, "dm")
    acc_snr = extract_accuracy(snr_file, "snr")

    return acc_snr, acc_dm, acc_time


    acc_time = extract_accuracy(acc_file, "time")
    acc_dm = extract_accuracy(acc_file, "dm")
    acc_snr = extract_accuracy(snr_file, "snr")

    return acc_snr, acc_dm, acc_time


def read_runtime_from_file(data_file):
    """
    Read total runtime from a companion Excel file derived from the data filename.

    The function extracts ``benchmark_id``, ``dm_tol``, and ``boxcar_width`` from
    the basename using the pattern:
    ``r"beam_(CUT_[A-Z_\\d]+_inj_\\d+)_processed_frb_([\\d.]+)_([\\d]+)\\.cand"``

    It then constructs:
    ``files_processed/{benchmark_id}_processed/frb_{dm_tol}_{boxcar_width}/frb_{dm_tol}_{boxcar_width}_output.xlsx``

    and reads the last non-null entry of the ``Time`` column.

    Parameters
    ----------
    data_file : str
        Path to the measured data file whose runtime artifact should be read.

    Returns
    -------
    float or None
        The last non-null value from the ``Time`` column if available;
        otherwise ``None``.
    """
    # Extract benchmark ID and parameters
    match = re.search(r"beam_(CUT_[A-Z_\d]+_inj_\d+)_processed_frb_([\d.]+)_([\d]+)\.cand", os.path.basename(data_file))
    if not match:
        print(f"Could not extract benchmark ID and parameters from {data_file}")
        return None

    benchmark_id = match.group(1)
    dm_tol = match.group(2)
    boxcar_width = match.group(3)

    base_dir = f"files_processed/{benchmark_id}_processed/frb_{dm_tol}_{boxcar_width}"
    output_file = os.path.join(base_dir, f"frb_{dm_tol}_{boxcar_width}_output.xlsx")

    if not os.path.exists(output_file):
        print(f"Runtime output file not found: {output_file}")
        return None

    try:
        df = pd.read_excel(output_file)
        if "Time" in df.columns:
            runtime = df["Time"].dropna().iloc[-1]
            return runtime
    except Exception as e:
        print(f"Failed to read runtime from {output_file} → {e}")

    return None


def process_data_file(data_file, benchmark_file):
    """
    Transform a measured detections file into a tidy, augmented DataFrame.

    Steps
    -----
    1. Load measured data (TSV with no header), right-shift columns, drop the
       first column, and assign standard column names.
    2. Derive ``pulse width`` = ``last sample`` - ``first sample``.
    3. Keep the top 13 rows by SNR, then sort by ``time``.
    4. Attach:
       - filename (basename),
       - percentage accuracies (SNR, DM, time) via :func:`read_accuracy_from_file`,
       - total runtime via :func:`read_runtime_from_file`,
       - percentage accuracy for pulse width vs. benchmark ``FWHM_1``.
    5. Extract DM tolerance and boxcar width from the filename via
       :func:`extract_dm_tol_boxcar`.
    6. Return the final table with selected columns and assert it has 13 rows.

    Parameters
    ----------
    data_file : str
        Path to the measured detections (tab-separated, no header).
    benchmark_file : str
        Path to the benchmark CSV with at least columns ``TOA`` and ``FWHM_1``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        ``["file name", "SNR", "time", "DM", "pulse width",
          "percentage accuracy SNR", "percentage accuracy DM",
          "percentage accuracy time", "percentage accuracy pulse width",
          "DM tolerance", "boxcar width", "runtime"]``

    Raises
    ------
    AssertionError
        If the final DataFrame does not have exactly 13 rows.
    """
    data = pd.read_csv(data_file, header=None, sep='\t')
    benchmark = pd.read_csv(benchmark_file)

    data = data.shift(axis=1).drop(columns=[0])
    data.columns = column_names

    # Calculate pulse width
    data["pulse width"] = data["last sample"] - data["first sample"]

    # Select top 13 by SNR, then sort by time
    data = data.sort_values(by="SNR", ascending=False).head(13).reset_index(drop=True)
    data = data.sort_values(by="time").reset_index(drop=True)

    # Add file name column
    data["file name"] = os.path.basename(data_file)

    # Add accuracy values from external files
    acc_snr, acc_dm, acc_time = read_accuracy_from_file(data_file)
    
    data["percentage accuracy SNR"] = [acc_snr] * len(data)
    data["percentage accuracy DM"] = [acc_dm] * len(data)
    data["percentage accuracy time"] = [acc_time] * len(data)

    # Add runtime from Excel outputs
    runtime = read_runtime_from_file(data_file)
    data["runtime"] = [runtime] * len(data)


    # Calculate pulse width accuracy
    benchmark = benchmark.sort_values(by="TOA").reset_index(drop=True)
    def get_accuracy(row):
        """
        Compute pulse width percentage accuracy for a single row.

        Uses :func:`calculate_percentage_accuracy` against benchmark ``FWHM_1``.
        Returns ``None`` if the benchmark value is missing or marked as -999.0.
        """
        fwhm_val = benchmark.iloc[row.name]["FWHM_1"]
        if pd.notnull(fwhm_val) and fwhm_val != -999.0:
            return calculate_percentage_accuracy(row["pulse width"], fwhm_val)
        return None

    data["percentage accuracy pulse width"] = data.apply(get_accuracy, axis=1)

    # Extract metadata
    dm_tol, boxcar_width = extract_dm_tol_boxcar(os.path.basename(data_file))
    data["DM tolerance"] = dm_tol
    data["boxcar width"] = boxcar_width

    final_cols = [
        "file name", "SNR", "time", "DM", "pulse width",
        "percentage accuracy SNR", "percentage accuracy DM",
        "percentage accuracy time", "percentage accuracy pulse width",
        "DM tolerance", "boxcar width", "runtime"
    ]

    assert len(data) == 13, f"Final data has {len(data)} rows, expected 13"

    return data[final_cols]
