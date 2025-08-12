"""
Load SNR/DM/FWHM data from multiple CSVs and plot 2D histograms with counts.

This script:
1) reads file paths from ``selected_files_log.txt`` (one path per line),
2) loads each CSV selecting columns ``["SNR", "DM", "FWHM_1"]``,
3) applies log10 to ``DM`` and ``FWHM_1`` (treating zeros as missing, then filling with 0),
4) stacks all rows, drops duplicates, and
5) generates three 2D histograms (with overlaid bin counts):
   - DM (log) vs SNR
   - DM (log) vs Width (log FWHM_1)
   - SNR vs Width (log FWHM_1)

The combined figure is saved as ``Histograms_final_{n_bins}.png`` and displayed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data and apply log scaling
def load_data_from_file(file_path, columns=["SNR", "DM", "FWHM_1"]):
    """
    Load selected columns from a CSV and return a NumPy array with log-scaled features.

    The function reads the given CSV, selects the provided columns, and applies
    ``log10`` to ``DM`` and ``FWHM_1``. Zeros are treated as missing (NaN) prior
    to the log transform and then filled back with 0 to avoid ``-inf``.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.
    columns : list of str, optional
        Column names to read from the CSV. Default is ``["SNR", "DM", "FWHM_1"]``.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(n_samples, 3)`` with columns ``[SNR, log10(DM), log10(FWHM_1)]``.
        Returns ``None`` if the file cannot be read or columns are missing.

    Notes
    -----
    - Any zero values in ``DM`` or ``FWHM_1`` are converted to NaN before the
      log transform and subsequently filled with 0.
    - Errors are printed and ``None`` is returned on failure.
    """
    try:
        df = pd.read_csv(file_path, usecols=columns)
        df["DM"] = np.log10(df["DM"].replace(0, np.nan)).fillna(0)
        df["FWHM_1"] = np.log10(df["FWHM_1"].replace(0, np.nan)).fillna(0)
        return df.values
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to plot 2D histograms with bin counts
def plot_2d_histograms(data, n_bins=10):
    """
    Plot three 2D histograms (with count labels) and save the figure.

    Given an array with columns ``[SNR, log10(DM), log10(FWHM_1)]``, the function
    computes equal-width bins per axis and renders:
      1) DM (log) vs SNR
      2) DM (log) vs Width (log FWHM_1)
      3) SNR vs Width (log FWHM_1)

    Each subplot includes a colorbar (bin counts) and overlays the count value at
    the center of non-empty bins. The figure is saved as
    ``Histograms_final_{n_bins}.png`` and shown.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_samples, 3)`` with columns ``[SNR, log10(DM), log10(FWHM_1)]``.
    n_bins : int, optional
        Number of bins per axis for the histograms. Default is ``10``.

    Returns
    -------
    None
        The function saves the figure to disk and displays it.

    Notes
    -----
    - If ``data`` is empty, a message is printed and the function returns without plotting.
    - Binning uses ``np.linspace`` between the observed min and max for each axis.
    """
    if len(data) == 0:
        print("No data available for plotting.")
        return

    SNR = data[:, 0]
    DM = data[:, 1]
    FWHM_1 = data[:, 2]

    snr_bins = np.linspace(min(SNR), max(SNR), n_bins + 1)
    dm_bins = np.linspace(min(DM), max(DM), n_bins + 1)
    fwhm_bins = np.linspace(min(FWHM_1), max(FWHM_1), n_bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    h1, _, _, im1 = axes[0].hist2d(DM, SNR, bins=[dm_bins, snr_bins], cmap="Blues")
    axes[0].set_xlabel("log(DM)")
    axes[0].set_ylabel("SNR")
    axes[0].set_title("DM vs SNR")

    h2, _, _, im2 = axes[1].hist2d(DM, FWHM_1, bins=[dm_bins, fwhm_bins], cmap="Reds")
    axes[1].set_xlabel("log(DM)")
    axes[1].set_ylabel("log(FWHM_1)")
    axes[1].set_title("DM vs Width (FWHM_1)")

    h3, _, _, im3 = axes[2].hist2d(SNR, FWHM_1, bins=[snr_bins, fwhm_bins], cmap="Greens")
    axes[2].set_xlabel("SNR")
    axes[2].set_ylabel("log(FWHM_1)")
    axes[2].set_title("SNR vs Width (FWHM_1)")

    fig.colorbar(im1, ax=axes[0], label="Count")
    fig.colorbar(im2, ax=axes[1], label="Count")
    fig.colorbar(im3, ax=axes[2], label="Count")

    # Overlay bin counts for each plot
    for ax, hist_data, x_bins, y_bins in zip(
        axes, [h1, h2, h3],
        [dm_bins, dm_bins, snr_bins],
        [snr_bins, fwhm_bins, fwhm_bins]
    ):
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                count = hist_data[i, j]
                if count > 0:
                    ax.text(
                        (x_bins[i] + x_bins[i + 1]) / 2,
                        (y_bins[j] + y_bins[j + 1]) / 2,
                        f"{int(count)}",
                        color="black", fontsize=8, ha="center", va="center"
                    )

    filename = f"Histograms_final_{n_bins}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {filename}")
    plt.show()

# Load file list from text file
file_list_path = "selected_files_log.txt"  # Change if your list file has a different name
try:
    with open(file_list_path, "r") as f:
        files_to_use = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(files_to_use)} file paths from {file_list_path}")
except FileNotFoundError:
    print(f"File list '{file_list_path}' not found.")
    files_to_use = []

# Load and stack data
data_list = [load_data_from_file(f) for f in files_to_use if load_data_from_file(f) is not None]
if data_list:
    combined_data = np.unique(np.vstack(data_list), axis=0)
    print(f"Loaded {len(files_to_use)} files; total unique data points: {len(combined_data)}")
    plot_2d_histograms(combined_data, n_bins=10)
else:
    print("No valid data loaded from the supplied file list.")
