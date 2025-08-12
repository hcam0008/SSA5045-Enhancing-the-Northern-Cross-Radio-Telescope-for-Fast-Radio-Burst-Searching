"""
Statistical comparison of parameter combinations using boxplots, Friedman, Nemenyi, and CD diagrams.

This script loads a wide-format accuracy table (per file × per-combination) and, for
each metric in ``["SNR", "DM", "Time"]``:

1) Visualises distributions across combinations via boxplots and annotates each box
   with its outlier count (based on IQR fences).
2) Runs the Friedman test to detect overall differences among combinations.
3) Applies the Nemenyi post-hoc test (via ``scikit_posthocs``) and visualises the
   pairwise p-value matrix; optionally shows a focused subset.
4) Computes average ranks per combination and plots a Critical Difference (CD) diagram
   using :func:`plot_cd_diagram` (imported from ``functions``).

Inputs
------
- CSV file at ``/path/to/your/file/files_dataframe_combos.csv`` containing columns that
  match the names listed in ``df_specific_cols`` below (e.g., ``"SNR_1.001_512"``).

Outputs
-------
- Boxplots with outlier counts for each metric.
- Heatmaps of Nemenyi post-hoc p-values (full matrix and optional subset).
- A Critical Difference diagram per metric.

Notes
-----
- The simplified CD computation uses a fixed critical value ``q_alpha = 2.569`` (Demsar, 2006 example);
  adjust this according to your actual ``k`` and significance level as needed.
- This script assumes the plotting alias ``plt`` (from Matplotlib) is available in scope.
- No corrections for missing data are performed; ensure input columns are present and numeric.
"""

from scipy.stats import friedmanchisquare
import pandas as pd
import scikit_posthocs as sp
import numpy as np
import seaborn as sns
from matplotlib import pyplot, rcParams
from functions import *
from math import sqrt

# Load Dataframe
df = pd.read_csv("/path/to/your/file/files_dataframe_combos.csv")

df_specific_cols = ["SNR_1.001_512", "SNR_1.001_256", "SNR_1.01_512", "SNR_1.01_256", "SNR_1.05_512", "SNR_1.05_256", "SNR_1.1_512", "SNR_1.1_256", "SNR_1.2_512", "SNR_1.2_256", "DM_1.001_512", "DM_1.001_256", "DM_1.01_512", "DM_1.01_256", "DM_1.05_512", "DM_1.05_256", "DM_1.1_512", "DM_1.1_256", "DM_1.2_512", "DM_1.2_256", "Time_1.001_512", "Time_1.001_256", "Time_1.01_512", "Time_1.01_256", "Time_1.05_512", "Time_1.05_256", "Time_1.1_512", "Time_1.1_256", "Time_1.2_512", "Time_1.2_256"]

# List of metrics to test
metrics = ["SNR", "DM", "Time"]

####################### BOXPLOTS TO VISUALISE MEAN AND VARIANCE OF EACH COMBINATION #######################
for metric in metrics:
    print(f"Generating outlier plot for {metric}...")

    # Filter metric-specific columns
    metric_cols = [col for col in df_specific_cols if col.startswith(f"{metric}_")]
    if not metric_cols:
        print(f"No columns found for {metric}, skipping.")
        continue

    # Melt data for long-form plotting
    df_melted = df[metric_cols].copy()
    df_melted['File'] = df.index
    df_melted = df_melted.melt(id_vars='File', var_name='Combination', value_name='Accuracy')

    # Calculate IQR bounds and identify outliers
    bounds = {}
    outlier_counts = {}

    for combo in metric_cols:
        s = df[combo]
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[combo] = (lower, upper)

        # Outlier count
        outlier_flags_for_combo = s.apply(lambda x: x < lower or x > upper)
        outlier_counts[combo] = outlier_flags_for_combo.sum()

    # Add outlier flags to melted DataFrame
    df_melted['Is_Outlier'] = df_melted.apply(
        lambda row: row['Accuracy'] < bounds[row['Combination']][0] or row['Accuracy'] > bounds[row['Combination']][1],
        axis=1
    )

    # Plotting Boxplot with Outliers
    plt.figure(figsize=(18, 7))
    sns.boxplot(data=df_melted, x='Combination', y='Accuracy', showfliers=False)
    rcParams["axes.titlesize"] = 20
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18

    for combo in metric_cols:
        outliers = df_melted[(df_melted['Combination'] == combo) & (df_melted['Is_Outlier'])]
        x_pos = metric_cols.index(combo)
        #plt.scatter([x_pos] * len(outliers), outliers['Accuracy'],
        #            color='black', s=20)

    # Add outlier count labels above boxes
    for i, combo in enumerate(metric_cols):
        count = outlier_counts[combo]
        plt.text(i, df_melted['Accuracy'].max() + 0.01, str(count),
                    ha='center', fontsize=9)

    plt.title(f"{metric} Accuracy Boxplot with Outlier Counts per Combination")
    plt.ylabel("Accuracy")
    plt.xlabel("Combination")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
###########################################################################################################



######################### FRIEDMAN TEST TO CHECK COMBINATIONS SIGNIFICANCE #########################
for metric in metrics:
    print(f"Running tests for: {metric}")

    # Filter columns for this metric
    metric_cols = [col for col in df_specific_cols if col.startswith(metric + "_")]
    df_metric = df[metric_cols]

    # Skip if not enough columns
    if len(metric_cols) < 3:
        print(f"Not enough parameter combinations for {metric}, skipping.")
        continue

    # Run Friedman test
    stat, p = friedmanchisquare(*[df_metric[col] for col in metric_cols])
    print(f"Friedman Statistic: {stat:.4f}")
    print(f"P-value: {p}")

    if p < 0.05:
        print("Significant differences found between parameter combos.")
    else:
        print("No significant differences found.")

###################################################################################################

####################### NEMENYI TEST AND CORRELATION MATRICES VISUALISATION #######################
    # Run Nemenyi post hoc test
    nemenyi = sp.posthoc_nemenyi_friedman(df_metric.values)
    nemenyi.columns = df_metric.columns
    nemenyi.index = df_metric.columns

    # Display top results
    print("Pairwise p-values (Nemenyi test):")
    print(nemenyi)

    rcParams["axes.titlesize"] = 20
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18

    # Plot original p-value matrix
    plt.figure(figsize=(18, 15))
    sns.heatmap(nemenyi, annot=True, cmap='coolwarm', fmt=".3f")
    plt.title(f"Nemenyi Post Hoc Test - {metric} Accuracy (p-values)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Optional: Define focused combos per metric
    focused_combos = {
        "SNR": ["SNR_1.001_512", "SNR_1.01_512", "SNR_1.05_512", "SNR_1.1_512"],
        "DM": ["DM_1.001_512", "DM_1.01_512", "DM_1.05_512", "DM_1.1_512"],
        "Time": ["Time_1.001_512", "Time_1.01_512", "Time_1.05_512", "Time_1.1_512"]
        # Add more if needed
    }

    # Select subset of specific combinations for better visual
    if metric in focused_combos:
        focus = focused_combos[metric]
        valid_focus = [f for f in focus if f in nemenyi.columns]
        if len(valid_focus) >= 2:
            nemenyi_subset = nemenyi.loc[valid_focus, valid_focus]
            plt.figure(figsize=(8, 6))
            sns.heatmap(nemenyi_subset, annot=True, cmap='coolwarm', fmt=".3f")
            plt.title(f"Nemenyi Test - {metric} Accuracy")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

################################################################################################

############################ CRITICAL DIFFERENCE DIAGRAM ################################

    # Compute average ranks across all combinations
    ranks = df_metric.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().values
    combo_names = df_metric.columns.tolist()

    # Estimate critical difference manually (simplified formula)
    k = len(combo_names)  # number of combos
    N = df_metric.shape[0]  # number of files

    # Critical difference for alpha=0.05 (95% confidence)
    q_alpha = 2.569  # from Demsar 2006, for k=4, alpha=0.05
    cd = q_alpha * sqrt(k * (k + 1) / (6 * N))

    # Plot
    plot_cd_diagram(avg_ranks, combo_names, cd, title=f"Critical Difference Diagram - {metric}")

#########################################################################################
