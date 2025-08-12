"""
Plot a Critical Difference (CD) diagram from average ranks and a CD value.

This module provides a single helper to visualise pairwise comparison results
(e.g., across algorithm–parameter combinations) using average ranks and a
precomputed critical difference (CD). Lower ranks indicate better performance.

This function is to be used in the statistical_tests.py script

Adapted from Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine learning research, 7(Jan), 1-30.

"""

# Adapted version of CD diagram plotter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def plot_cd_diagram(avg_ranks, combo_names, cd, title="Critical Difference Diagram"):
    """
    Plot a Critical Difference diagram from average ranks and a CD value.

    Parameters
    ----------
    avg_ranks : array-like of shape (n_combos,)
        Average ranks for each combination (lower is better). The function
        sorts combinations by rank for display.
    combo_names : array-like of shape (n_combos,)
        Labels corresponding to `avg_ranks` (e.g., algorithm names or
        algorithm–parameter tuples).
    cd : float
        Critical difference value (e.g., from Nemenyi or related post-hoc test).
        A horizontal bar of length `cd` is drawn near the top of the plot.
    title : str, optional
        Title of the plot. Default is ``"Critical Difference Diagram"``.

    Returns
    -------
    None
        The function displays the figure.

    Notes
    -----
    - This function assumes `avg_ranks` and `combo_names` have the same length.
    - The CD bar is drawn at the right side of the axis, from
      ``max(sorted_ranks) - cd`` to ``max(sorted_ranks)``.
    - No statistical computation is performed here; provide `cd` from your own
      test (e.g., Nemenyi) and the corresponding average ranks.

    Examples
    --------
    >>> avg = [2.1, 1.7, 3.0]
    >>> names = ["A", "B", "C"]
    >>> plot_cd_diagram(avg, names, cd=0.6)
    """
    # Sort by rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_ranks = np.array(avg_ranks)[sorted_indices]
    sorted_names = np.array(combo_names)[sorted_indices]

    rcParams["axes.titlesize"] = 20
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18

    # Setup plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(min(sorted_ranks) - 0.25, max(sorted_ranks) + 0.25)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(title)

    # Plot horizontal line for each combo
    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        ax.plot([rank, rank], [0.1, 0.2], color='black')
        ax.text(rank, 0.25, name, rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=16)

    # Plot main horizontal axis
    ax.hlines(0.1, min(sorted_ranks) - 0.25, max(sorted_ranks) + 0.25, color='black')

    # Draw critical difference bar
    x_start = max(sorted_ranks) - cd
    x_end = max(sorted_ranks)
    ax.plot([x_start, x_end], [0.75, 0.75], color='black', lw=2)
    ax.plot([x_start, x_start], [0.7, 0.8], color='black', lw=2)
    ax.plot([x_end, x_end], [0.7, 0.8], color='black', lw=2)
    ax.text((x_start + x_end) / 2, 0.85, f"CD = {cd:.2f}", ha='center', fontsize=16)

    plt.tight_layout()
    plt.show()
