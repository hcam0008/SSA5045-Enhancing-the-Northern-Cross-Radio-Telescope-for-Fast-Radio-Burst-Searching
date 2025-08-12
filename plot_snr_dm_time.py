"""
Plot DM vs Time with point sizes proportional to SNR for measured vs benchmark data.

This script reads a measured detections TSV (no header) and a benchmark CSV,
realigns/labels the measured columns, selects the top 13 rows by SNR from each
dataset, and makes a scatter plot of DM vs Time where marker size encodes SNR.
Measured points are overlaid with benchmark (expected) points for comparison.
The figure is saved to disk.

Command-line arguments
----------------------
-i, --input : str
    Path to the measured data file (tab-separated, no header).
-b, --benchmark : str
    Path to the benchmark CSV (must include columns DM, TOA, SNR).
-o, --output : str, optional
    Output image filename. Default: ``DM_vs_time_vs_SNR.png``.

Notes
-----
- Measured file columns are expected to map to:
  ``["SNR","sample number","time","filter","DM trial number","DM","members","first sample","last sample"]``
  after a one-column right shift and dropping the first (NaN) column.
- The script selects the top 13 rows by SNR from both measured and benchmark
  tables before plotting.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def validate_file(f):
    """
    Validate that a given file path exists for CLI argument parsing.

    Parameters
    ----------
    f : str
        Filesystem path provided as an argument.

    Returns
    -------
    str
        The same path if it exists.

    Raises
    ------
    argparse.ArgumentTypeError
        If the file does not exist.
    """
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError(f"The file {f} does not exist.")
    return f

parser = argparse.ArgumentParser(description="Read file form Command line.")
parser.add_argument("-i", "--input", dest="input_filename", required=True, type=validate_file,
                        help="input file", metavar="FILE")
parser.add_argument("-b", "--benchmark", dest="benchmark_filename", required=True, type=validate_file, help="benchmark file for comparison", metavar="FILE")
parser.add_argument("-o", "--output", dest="output_filename", required=False, default="DM_vs_time_vs_SNR.png", help="output plot filename (default: DM_vs_time_vs_SNR.png)", metavar="FILE")

args = parser.parse_args()

column_names = ["SNR", "sample number", "time", "filter", "DM trial number", "DM", "members", "first sample", "last sample"]

# Read the file without assuming a header
data_file = args.input_filename
data = pd.read_csv(data_file, header=None, sep='\t')

# Read benchmark file normally
benchmark_file = args.benchmark_filename
benchmark = pd.read_csv(benchmark_file)#, header=None, sep='\t')

# Realign columns
data = data.shift(axis=1)  # Shift columns to the right

# Drop the first column created by the shift (contains NaN)
data = data.drop(columns=[0])

# Assign the correct headers
data.columns = column_names

# Extracting data columns
data_columns = {"DM", "time", "SNR"}
if not data_columns.issubset(data.columns):
    raise ValueError(f"The dataset must contain the columns: {data_columns}")

benchmark_columns = {"DM", "TOA", "SNR"}
if not benchmark_columns.issubset(benchmark.columns):
    raise ValueError(f"The benchmarking dataset must contain the columns: {benchmark_columns}")

# Convert SNR, DM, and time to numeric (handle non-numeric issues)
data["SNR"] = pd.to_numeric(data["SNR"])#, errors="coerce")
data["DM"] = pd.to_numeric(data["DM"])#, errors="coerce")
data["time"] = pd.to_numeric(data["time"])#, errors="coerce")

# Sort the dataset by SNR in descending order
sorted_data = data.sort_values(by='SNR', ascending=False)
sorted_benchmark = benchmark.sort_values(by='SNR', ascending=False)

# Select the top 10 rows with the largest SNR values
filtered_data = sorted_data.head(13)
filtered_benchmark = sorted_benchmark.head(13)

# Extracting DM, time, and SNR from the filtered data
dm_data = filtered_data['DM']
time_data = filtered_data['time']
snr_data = filtered_data['SNR']

dm_benchmark = filtered_benchmark['DM']
time_benchmark = filtered_benchmark['TOA']
snr_benchmark = filtered_benchmark['SNR']

# Ensure 'snr', 'time', and 'dm' have the same length
if len(snr_data) != len(time_data) or len(snr_data) != len(dm_data):
    raise ValueError("Length of SNR, time, and DM columns must be the same.")

if len(snr_benchmark) != len(time_benchmark) or len(snr_benchmark) != len(dm_benchmark):
    raise ValueError("Length of SNR, time, and DM columns must be the same.")

# Plotting DM vs time with marker size proportional to SNR
plt.scatter(time_data, dm_data, s=snr_data, alpha=0.75, c='c', label='Measured')
plt.scatter(time_benchmark, dm_benchmark, s=snr_benchmark, alpha=1, c='r', label='Expected')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Dispersion Measure (DM)')
plt.title('DM vs Time vs SNR')
plt.grid(True)
plt.legend()

# Saving and displaying the plot
output_file = args.output_filename
plt.savefig(output_file)
plt.show()

print(f"Plot saved to {output_file}")
