"""
Interactively build a diverse dataset by sampling CSVs and tracking 3D bin coverage.

This script helps you curate a set of files whose samples cover a 3D space defined by
[SNR, log10(DM), log10(FWHM_1)]. It:

1) randomly selects files from a folder while avoiding high-duplicate batches using a
   tolerance-based match percentage,
2) computes coverage over a 3D grid (n_bins per axis),
3) saves progress (coverage, bin size, number of files added) to a JSON file so runs
   can be resumed, and
4) plots 2D histograms (DM vs SNR, DM vs Width, SNR vs Width) for the latest batch.

Selected file paths are appended to ``selected_files_log.txt`` for later reuse.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import json
import random

# File to save progress
PROGRESS_FILE = "coverage_progress.json"

# Initialize dataset
data_list = []
coverage_progress = {}

# Function to save progress
def save_progress(coverage, occupied_bins, bin_size, files_added):
    """
    Persist summary progress to disk.

    Parameters
    ----------
    coverage : float
        Overall 3D coverage percentage for the accumulated dataset.
    occupied_bins : set[tuple[int, int, int]]
        Set of occupied (x, y, z) bin indices. (Not persisted — accepted for API symmetry.)
    bin_size : int
        Number of bins used per axis in the 3D space.
    files_added : int
        Number of files added in the latest selection batch.

    Returns
    -------
    None
        Writes a JSON file ``coverage_progress.json`` with coverage, bin_size, files_added.
    """
    progress_data = {
        "coverage": coverage,
        "bin_size": bin_size,
        "files_added": files_added
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f)
    print(f"Progress saved: {coverage:.2f}% coverage, bin size {bin_size}, {files_added} files used.")

# Function to load progress if available
def load_progress():
    """
    Load previously saved progress from ``coverage_progress.json`` if present.

    Returns
    -------
    dict or None
        Dictionary with keys ``coverage``, ``bin_size``, ``files_added`` if the
        progress file exists; otherwise ``None``.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return None

# Function to load data and apply log scale
def load_data_from_file(file_path, columns=["SNR", "DM", "FWHM_1"]):
    """
    Load SNR/DM/FWHM_1 columns from a CSV, apply log10 to DM and FWHM_1, return ndarray.

    Zeros in DM and FWHM_1 are treated as missing (NaN) before log10, then filled with 0.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to read.
    columns : list of str, optional
        Column names to load. Default is ``["SNR", "DM", "FWHM_1"]``.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(n_samples, 3)`` with columns ``[SNR, log10(DM), log10(FWHM_1)]``,
        or ``None`` if reading fails.
    """
    try:
        df = pd.read_csv(file_path, usecols=columns)
        
        # Apply log10 transformation to DM and FWHM_1, handling zero values
        df["DM"] = np.log10(df["DM"].replace(0, np.nan)).fillna(0)
        df["FWHM_1"] = np.log10(df["FWHM_1"].replace(0, np.nan)).fillna(0)
        
        return df.values  # Convert to NumPy array
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to bin and check coverage
def check_coverage(data, n_bins):
    """
    Compute coverage percentage in a 3D grid over [SNR, log10(DM), log10(FWHM_1)].

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_samples, 3)`` with columns ``[SNR, logDM, logFWHM_1]``.
    n_bins : int
        Number of equal-width bins per axis.

    Returns
    -------
    (float, set[tuple[int, int, int]])
        Coverage percentage (0–100) and the set of occupied (x, y, z) bin indices.

    Notes
    -----
    - Bins are defined between the observed min and max for each feature.
    - Uses ``np.digitize`` to map points to bins (0-indexed).
    """
    if len(data) == 0:
        return 0, set()

    x, y, z = data[:, 0], data[:, 1], data[:, 2]  # SNR, log(DM), log(FWHM_1)

    # Get bin edges
    x_bins = np.linspace(min(x), max(x), n_bins + 1)
    y_bins = np.linspace(min(y), max(y), n_bins + 1)
    z_bins = np.linspace(min(z), max(z), n_bins + 1)

    # Digitize points
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    z_idx = np.digitize(z, z_bins) - 1

    # Store unique occupied bins
    occupied_bins = set(zip(x_idx, y_idx, z_idx))

    # Total number of bins
    total_bins = n_bins ** 3
    occupied_bin_count = len(occupied_bins)

    # Coverage percentage
    coverage_percentage = (occupied_bin_count / total_bins) * 100
    
    # print(x_bins, y_bins, z_bins)
    print(f"Total bins: {total_bins}, Occupied bins: {occupied_bin_count}")
    
    return coverage_percentage, occupied_bins


# Function to check how many data points match within a tolerance range
def compute_matching_percentage_with_tolerance(new_data, existing_data, tolerance, feature_index=0): # Change feature index according to which feature you want to use as tolerance parameter
    """
    Compute % of `new_data` points that approximately match `existing_data` on one feature.

    A new point is considered a match if there exists an existing point whose
    feature value differs by at most ``tolerance``.

    Parameters
    ----------
    new_data : numpy.ndarray
        Array of shape ``(n_new, d)``.
    existing_data : numpy.ndarray or None
        Array of shape ``(n_existing, d)``. If ``None`` or empty, returns 0%.
    tolerance : float
        Allowed absolute difference in the chosen feature.
    feature_index : int, optional
        Column index of the feature to compare (default: 0).

    Returns
    -------
    float
        Match percentage (0–100) over the rows of ``new_data``.
    """
    if existing_data is None or len(existing_data) == 0:
        return 0  # No matches if there's no existing data

    # Extract relevant feature values
    existing_feature_values = existing_data[:, feature_index]
    new_feature_values = new_data[:, feature_index]

    match_count = 0

    for value in new_feature_values:
        if np.any(np.abs(existing_feature_values - value) <= tolerance):
            match_count += 1

    match_percentage = (match_count / len(new_feature_values)) * 100
    return match_percentage

# Function to select random files while avoiding high-duplicate ones based on tolerance
def select_random_files(all_files, num_files, existing_data, tolerance, match_threshold, feature_index=0):
    """
    Randomly sample file paths, rejecting those too similar to existing data.

    For each candidate file, its data is loaded and a match percentage is computed
    against ``existing_data`` using :func:`compute_matching_percentage_with_tolerance`.
    Files whose match percentage exceeds ``match_threshold`` are skipped.

    Parameters
    ----------
    all_files : list[str]
        Candidate file paths to choose from.
    num_files : int
        Number of files to select (best effort).
    existing_data : numpy.ndarray or None
        Reference dataset for similarity checks.
    tolerance : float
        Absolute tolerance for feature matching.
    match_threshold : float
        Maximum allowed percent matches (0–100) to accept a file.
    feature_index : int, optional
        Feature column used for tolerance checks (default: 0).

    Returns
    -------
    list[str]
        Paths of selected files. May be fewer than ``num_files`` if few pass the filter.

    Notes
    -----
    - Makes at most ``2 * len(all_files)`` attempts.
    - Prints diagnostic messages for selected and skipped files.
    """
    selected_files = []
    attempts = 0  

    while len(selected_files) < num_files and attempts < len(all_files) * 2:
        file_path = random.choice(all_files)
        new_data = load_data_from_file(file_path)
        if new_data is None:
            continue  

        match_percentage = compute_matching_percentage_with_tolerance(new_data, existing_data, tolerance, feature_index)

        if match_percentage < match_threshold:
            selected_files.append(file_path)
            print(f"Selected: {file_path} (Matching: {match_percentage:.2f}% within ±{tolerance})")
        else:
            print(f"Skipped: {file_path} (Matching: {match_percentage:.2f}% exceeds {match_threshold}%)")

        attempts += 1

        if attempts >= len(all_files) * 2:
            print("Warning: Could not find enough unique files within tolerance limits.")
            break

    return selected_files


# Function to plot data
def plot_data(data, n_bins):
    """
    Plot 2D histograms (with counts) for DM vs SNR, DM vs Width, SNR vs Width.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_samples, 3)`` with columns ``[SNR, logDM, logFWHM_1]``.
    n_bins : int
        Number of bins per axis.

    Returns
    -------
    None
        Saves the figure as ``Histograms_BinSize_{n_bins}.png`` and displays it.
    """
    
    if len(data) == 0:
        print("No data available for plotting.")
        return

    # Extract columns
    SNR = data[:, 0]
    DM = data[:, 1]  # log(DM)
    FWHM_1 = data[:, 2]  # log(FWHM_1)

    # Define bin edges
    snr_bins = np.linspace(min(SNR), max(SNR), n_bins + 1)
    dm_bins = np.linspace(min(DM), max(DM), n_bins + 1)
    fwhm_bins = np.linspace(min(FWHM_1), max(FWHM_1), n_bins + 1)

    # Create subplots for three 2D histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # DM vs SNR plot
    h1, _, _, im1 = axes[0].hist2d(DM, SNR, bins=[dm_bins, snr_bins], cmap="Blues")
    axes[0].set_xlabel("log(DM)")
    axes[0].set_ylabel("SNR")
    axes[0].set_title("DM vs SNR")

    # DM vs FWHM_1 plot
    h2, _, _, im2 = axes[1].hist2d(DM, FWHM_1, bins=[dm_bins, fwhm_bins], cmap="Reds")
    axes[1].set_xlabel("log(DM)")
    axes[1].set_ylabel("log(FWHM_1)")
    axes[1].set_title("DM vs Width (FWHM_1)")

    # SNR vs FWHM_1 plot
    h3, _, _, im3 = axes[2].hist2d(SNR, FWHM_1, bins=[snr_bins, fwhm_bins], cmap="Greens")
    axes[2].set_xlabel("SNR")
    axes[2].set_ylabel("log(FWHM_1)")
    axes[2].set_title("SNR vs Width (FWHM_1)")

    # Add colorbars
    fig.colorbar(im1, ax=axes[0], label="Count")
    fig.colorbar(im2, ax=axes[1], label="Count")
    fig.colorbar(im3, ax=axes[2], label="Count")

    # Overlay text annotations to show bin counts
    for ax, hist_data, x_bins, y_bins in zip(axes, [h1, h2, h3], 
                                              [dm_bins, dm_bins, snr_bins], 
                                              [snr_bins, fwhm_bins, fwhm_bins]):
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                count = hist_data[i, j]
                if count > 0:  # Only display bins with data
                    ax.text((x_bins[i] + x_bins[i+1]) / 2, 
                            (y_bins[j] + y_bins[j+1]) / 2, 
                            f"{int(count)}", 
                            color="black", fontsize=8, ha="center", va="center")

    # Save the figure
    filename = f"Histograms_BinSize_{n_bins}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")

    plt.show()


# Function to save the selected file names to a log file
def save_selected_files(file_list, filename="selected_files_log.txt"):
    """
    Append selected file paths to a text log (one path per line).

    Parameters
    ----------
    file_list : list[str]
        File paths to append.
    filename : str, optional
        Output text file path. Default: ``"selected_files_log.txt"``.

    Returns
    -------
    None
    """
    with open(filename, "a") as f:  # Open file in append mode
        for file in file_list:
            f.write(file + "\n")  # Write each filename on a new line
    print(f"Selected file names saved to {filename}")


# Load previous progress if available
progress = load_progress()
if progress:
    print(f"Loaded previous progress: {progress['coverage']:.2f}% coverage with {progress['files_added']} files.")

# Interactive file addition loop
folder_path = "../simulated_frbs"  # Change this to your folder
all_files = sorted(glob.glob(f"{folder_path}/*.csv"))  # Get all CSV files in order

if not all_files:
    print("No CSV files found in the specified folder.")
else:
    while True:

        # User selects number of bins
        n_bins = int(input("\nEnter bin size for 3D space (default is 10): ") or "10")
        n_bins = max(2, n_bins)  # Ensure at least 2 bins

        # User selects how many files to add
        num_files = int(input(f"Enter number of files to add (1-{len(all_files)}) (default is 50): ") or "50")
        num_files = min(max(1, num_files), len(all_files))  # Keep within range

        tolerance = float(input("Enter DM similarity tolerance (default ±5): ") or "5")
        match_threshold = float(input("Enter max matching DM% before rejecting (default 50%): ") or "50")

        selected_files = select_random_files(all_files, num_files, data_list, tolerance, match_threshold, 0)

        # Save the list of selected file names
        save_selected_files(selected_files)

        # Load only the newly selected files
        new_data_list = [load_data_from_file(f) for f in selected_files if load_data_from_file(f) is not None]

        # Ensure new files are stacked together
        if new_data_list:
            latest_data = np.unique(np.vstack(new_data_list), axis=0)
        else:
            latest_data = np.array([])

        # Compute coverage for selected files
        if latest_data.size > 0:
            latest_coverage, latest_occupied_bins = check_coverage(latest_data, n_bins)
            print(f"Coverage of newly selected files: {latest_coverage:.2f}%")

        # Append newly selected files to full dataset
        if latest_data.size > 0:
            data_list.append(latest_data)

        # Ensure the entire dataset remains unique before coverage and plotting
        all_data = np.unique(np.vstack(data_list), axis=0) if data_list else np.array([])

        print(f"Total unique data points after filtering: {len(all_data)}")

        # Calculate coverage using the full dataset
        coverage, occupied_bins = check_coverage(all_data, n_bins)
        save_progress(coverage, occupied_bins, n_bins, len(selected_files))  # Only count this batch

        #  Plot only the newly selected files
        if latest_data.size > 0:
            print(f"Plotting {len(latest_data)} newly added data points")
            plot_data(latest_data, n_bins)  # Latest batch only

        # Option to stop
        if input("Type 'done' to stop: ").lower() == 'done':
            break
