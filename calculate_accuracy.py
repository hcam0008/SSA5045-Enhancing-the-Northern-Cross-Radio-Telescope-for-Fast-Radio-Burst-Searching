"""
Analyse accuracy of measured detections against benchmark catalogs.

This script reads a measured-detections file and a benchmark catalog, selects the
top 13 detections by SNR from each, aligns them by nearest time using
``pandas.merge_asof`` (with a tolerance of 1.0 in the same time units as the input),
and computes error metrics (MAE, MSE, RMSE, R², MAPE, and a simple accuracy
defined as ``100 - MAPE``) for `time` and `DM`. It also performs a separate pass
to compare SNR against an SNR benchmark file. Results are saved to text files.

Expected inputs
---------------
- Measured data file (tab-separated, no header) with columns that become:
  ``["SNR_data", "sample number", "time", "filter", "DM trial number",
  "DM_data", "members", "first sample", "last sample"]`` after a one-column
  right shift and dropping the first (NaN) column.
- Benchmark CSV with columns: ``DM``, ``TOA``, ``SNR``.
- SNR benchmark (tab-separated, no header) with columns that become:
  ``["SNR_bench", "sample number", "time", "filter", "DM trial number",
  "DM_bench", "members", "first sample", "last sample"]`` after the same shift.

Outputs
-------
- A summary text file for time and DM metrics (default: ``accuracy_results.txt``).
- A summary text file for SNR metrics (default: ``snr_accuracy_results.txt``).
- Console prints of filtered tables and output file paths.

Assumptions
-----------
- Time columns in measured and benchmark files are numeric and comparable.
- Only the top 13 rows by SNR are considered from each dataset before alignment.
- ``merge_asof`` uses ``direction="nearest"`` and ``tolerance=1.0``.
- Division by zero in MAPE is not expected (measured values should be non-zero).

Command-line arguments
----------------------
-i, --input : str
    Path to measured data (tab-separated).
-b, --benchmark : str
    Path to benchmark CSV (with DM, TOA, SNR).
-o, --output : str, optional
    Output filename for time/DM analysis. Default: ``accuracy_results.txt``.
-s, --snr : str
    Path to SNR benchmark (tab-separated).
-so, --snroutput : str, optional
    Output filename for SNR analysis. Default: ``snr_accuracy_results.txt``.

Examples
--------
Run with measured data, a benchmark CSV, and an SNR benchmark, saving results
to the defaults:

    python compare_accuracy.py \\
        -i measured.tsv \\
        -b benchmark.csv \\
        -s snr_benchmark.tsv

Specify custom output filenames:

    python compare_accuracy.py \\
        -i measured.tsv \\
        -b benchmark.csv \\
        -o dm_time_results.txt \\
        -s snr_benchmark.tsv \\
        -so snr_results.txt
"""

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to validate file paths
def validate_file(f):
    """
    Validate that a given file path exists for CLI argument parsing.

    Parameters
    ----------
    f : str
        Path to a file supplied via a command-line argument.

    Returns
    -------
    str
        The same file path `f` if it exists.

    Raises
    ------
    argparse.ArgumentTypeError
        If the path does not exist on the filesystem.
    """
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError(f"The file {f} does not exist.")
    return f

# Argument parsing for command-line execution
parser = argparse.ArgumentParser(description="Read file from Command line.")
parser.add_argument("-i", "--input", dest="input_filename", required=True, type=validate_file, help="Input file", metavar="FILE")
parser.add_argument("-b", "--benchmark", dest="benchmark_filename", required=True, type=validate_file, help="Benchmark file for comparison", metavar="FILE")
parser.add_argument("-o", "--output", dest="output_filename", required=False, default="accuracy_results.txt", help="Output filename (default: accuracy_results.txt)", metavar="FILE")
parser.add_argument("-s", "--snr", dest= "snr_benchmark", required=True, type=validate_file, help="SNR benchmark file", metavar="FILE")
parser.add_argument("-so", "--snroutput", dest="snr_output_filename", required=False, default="snr_accuracy_results.txt", help="Output filename (default: snr_accuracy_results.txt)", metavar="FILE")

args = parser.parse_args()

# Define column names
column_names = ["SNR_data", "sample number", "time", "filter", "DM trial number", "DM_data", "members", "first sample", "last sample"]

# Load measured and benchmark datasets
data = pd.read_csv(args.input_filename, header=None, sep='\t')
benchmark = pd.read_csv(args.benchmark_filename)

# Realign columns
data = data.shift(axis=1)  # Shift columns to the right
data = data.drop(columns=[0])  # Drop the first column (contains NaN from shift)
data.columns = column_names  # Assign column names

# Ensure necessary columns exist
data_columns = {"DM_data", "time", "SNR_data"}
benchmark_columns = {"DM", "TOA", "SNR"}

if not data_columns.issubset(data.columns):
    raise ValueError(f"The dataset must contain the columns: {data_columns}")

if not benchmark_columns.issubset(benchmark.columns):
    raise ValueError(f"The benchmarking dataset must contain the columns: {benchmark_columns}")

# Convert necessary columns to numeric
data["SNR_data"] = pd.to_numeric(data["SNR_data"], errors='coerce')
data["DM_data"] = pd.to_numeric(data["DM_data"], errors='coerce')
data["time"] = pd.to_numeric(data["time"], errors='coerce')

benchmark["SNR"] = pd.to_numeric(benchmark["SNR"], errors='coerce')
benchmark["DM"] = pd.to_numeric(benchmark["DM"], errors='coerce')
benchmark["TOA"] = pd.to_numeric(benchmark["TOA"], errors='coerce')

# Select the top 13 rows with the highest SNR values
filtered_data = data.nlargest(13, "SNR_data", keep='first')
filtered_benchmark = benchmark.nlargest(13, "SNR", keep='first')

# Rename SNR columns before merging to avoid conflicts
filtered_data = filtered_data.rename(columns={"SNR_data": "SNR_measured", "time": "time_measured", "DM_data" : "DM_measured"})
filtered_benchmark = filtered_benchmark.rename(columns={"SNR": "SNR_predicted", "TOA": "time_predicted", "DM" : "DM_predicted"})

# Sort both DataFrames by time in ascending order
filtered_data = filtered_data.sort_values(by="time_measured", ascending=True)
filtered_benchmark = filtered_benchmark.sort_values(by="time_predicted", ascending=True)

# Debugging: Check sorted values
print("Filtered Data (sorted by time_measured):")
print(filtered_data[["SNR_measured", "time_measured"]])

print("Filtered Benchmark (sorted by time_predicted):")
print(filtered_benchmark[["SNR_predicted", "time_predicted"]])

# Merge datasets with nearest time match using merge_asof
merged_data = pd.merge_asof(
    filtered_data,
    filtered_benchmark,
    left_on="time_measured",
    right_on="time_predicted",
    suffixes=('_measured', '_predicted'),
    tolerance=1.0,
    direction="nearest"
)


# Ensure expected columns exist
expected_columns = {"time_measured", "SNR_measured", "DM_measured", "time_predicted", "SNR_predicted", "DM_predicted"}
missing_columns = expected_columns - set(merged_data.columns)

if missing_columns:
    print(f"Warning: The following expected columns are missing from merged_data: {missing_columns}")


# Ensure merging was successful
if merged_data.empty:
    raise ValueError("Merging failed. Ensure time values have overlap.")


# List of variables to compare
variables = ["time", "DM"]

# Open a text file to save results
with open(args.output_filename, "w") as file:
    file.write("Statistical Analysis of Measured vs. Predicted Data\n")
    file.write("=" * 50 + "\n")

    # Compute error metrics for each variable
    for var in variables:
        measured_col = f"{var}_measured"
        predicted_col = f"{var}_predicted"

        if measured_col not in merged_data.columns or predicted_col not in merged_data.columns:
            file.write(f"\nSkipping {var}: Required columns not found.\n")
            continue

        measured = merged_data[measured_col].dropna()
        predicted = merged_data[predicted_col].dropna()

        if measured.empty or predicted.empty:
            file.write(f"\nSkipping {var}: No valid data after filtering.\n")
            continue


        # Select valid (non-null) rows for DM_measured and DM_predicted
        valid_rows = merged_data[[measured_col, predicted_col]].dropna()

        # Extract measured and predicted after ensuring they are the same length
        measured = valid_rows[measured_col]
        predicted = valid_rows[predicted_col]

        
        # Compute error metrics
        mae = mean_absolute_error(measured, predicted)
        mse = mean_squared_error(measured, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(measured, predicted)
        mape = np.mean(np.abs((measured - predicted) / measured)) * 100
        accuracy = 100 - mape

        # Write results to file
        file.write(f"\n--- {var.upper()} Analysis ---\n")
        file.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        file.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        file.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        file.write(f"R² Score: {r2:.4f}\n")
        file.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")
        file.write(f"Accuracy: {accuracy:.2f}%\n")
        file.write("-" * 40 + "\n")

print(f"Results saved to {args.output_filename}")

bench_column_names = ["SNR_bench", "sample number", "time", "filter", "DM trial number", "DM_bench", "members", "first sample", "last sample"]
snr_bench = pd.read_csv(args.snr_benchmark, header=None, sep='\t')

# Realign columns
snr_bench = snr_bench.shift(axis=1)  # Shift columns to the right
snr_bench = snr_bench.drop(columns=[0])  # Drop the first column (contains NaN from shift)
snr_bench.columns = bench_column_names  # Assign column names

# Ensure necessary columns exist
snr_bench_columns = {"DM_bench", "time", "SNR_bench"}

if not snr_bench_columns.issubset(snr_bench.columns):
    raise ValueError(f"The benchmarking dataset must contain the columns: {snr_bench_columns}")

# Convert necessary columns to numeric
snr_bench["SNR_bench"] = pd.to_numeric(snr_bench["SNR_bench"], errors='coerce')

# Select the top 13 rows with the highest SNR values
filtered_snr_bench = snr_bench.nlargest(13, "SNR_bench", keep='first')

# Rename SNR columns before merging to avoid conflicts
filtered_snr_bench = filtered_snr_bench.rename(columns={"SNR_bench": "SNR_predicted", "time": "time_predicted", "DM_bench" : "DM_predicted"})

# Sort both DataFrames by time in ascending order
filtered_snr_bench = filtered_snr_bench.sort_values(by="time_predicted", ascending=True)

# Debugging: Check sorted values
print("Filtered Benchmark (sorted by time_predicted):")
print(filtered_snr_bench[["SNR_predicted", "time_predicted"]])

# Merge datasets with nearest time match using merge_asof
merged_data = pd.merge_asof(
    filtered_data,
    filtered_snr_bench,
    left_on="time_measured",
    right_on="time_predicted",
    suffixes=('_measured', '_predicted'),
    tolerance=1.0,
    direction="nearest"
)

# List of variables to compare
variables = ["SNR"]

# Open a text file to save results
with open(args.snr_output_filename, "w") as file:
    file.write("Statistical Analysis of Measured vs. Predicted Data\n")
    file.write("=" * 50 + "\n")

    # Compute error metrics for each variable
    for var in variables:
        measured_col = f"{var}_measured"
        predicted_col = f"{var}_predicted"

        if measured_col not in merged_data.columns or predicted_col not in merged_data.columns:
            file.write(f"\nSkipping {var}: Required columns not found.\n")
            continue

        measured = merged_data[measured_col].dropna()
        predicted = merged_data[predicted_col].dropna()

        if measured.empty or predicted.empty:
            file.write(f"\nSkipping {var}: No valid data after filtering.\n")
            continue


        # Select valid (non-null) rows for DM_measured and DM_predicted
        valid_rows = merged_data[[measured_col, predicted_col]].dropna()

        # Extract measured and predicted after ensuring they are the same length
        measured = valid_rows[measured_col]
        predicted = valid_rows[predicted_col]


        # Compute error metrics
        mae = mean_absolute_error(measured, predicted)
        mse = mean_squared_error(measured, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(measured, predicted)
        mape = np.mean(np.abs((measured - predicted) / measured)) * 100
        accuracy = 100 - mape

        # Write results to file
        file.write(f"\n--- {var.upper()} Analysis ---\n")
        file.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        file.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        file.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        file.write(f"R² Score: {r2:.4f}\n")
        file.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")
        file.write(f"Accuracy: {accuracy:.2f}%\n")
        file.write("-" * 40 + "\n")

print(f"Results saved to {args.snr_output_filename}")
