"""
t-SNE exploration and clustering (DBSCAN, HDBSCAN, OPTICS) with diagnostics and plots.

This script loads a feature table, cleans and standardises selected features,
sweeps t-SNE hyperparameters to record KL divergence and save preview plots,
then runs a final t-SNE with chosen settings and applies multiple clustering
algorithms. It saves visualisations colored by cluster labels and by selected
accuracy metrics, prints silhouette scores, and exports the final DataFrame.

Inputs
------
files_dataframe_all_info.csv : CSV
    Table containing at least the features listed below.

Features used for t-SNE
-----------------------
- 'SNR'
- 'time'
- 'DM'
- 'percentage accuracy SNR'
- 'percentage accuracy DM'
- 'percentage accuracy time'
- 'DM tolerance'
- 'boxcar width'
- 'runtime'

Processing overview
-------------------
1. Replace ±inf with NaN and drop rows with missing values in selected features.
2. Standardize features via ``StandardScaler``.
3. Compute Hopkins statistic (clustering tendency diagnostic).
4. Sweep t-SNE over a grid of perplexities and max iterations; save scatter plots
   and record KL divergence per setting.
5. Run a final t-SNE (perplexity=55, max_iter=5000) and attach components as
   ``TSNE-1`` and ``TSNE-2`` to the DataFrame.
6. DBSCAN:
   - Heuristic minPts = 2 * dim(features) and k-distance graph for eps selection.
   - Evaluate silhouette over a range of ``min_samples`` values.
   - Fit final DBSCAN (eps=3, min_samples=25) on t-SNE space.
7. HDBSCAN:
   - Test multiple ``min_cluster_size`` values; print cluster counts and silhouette.
   - Fit final HDBSCAN (min_cluster_size=450) on t-SNE space.
8. OPTICS:
   - Evaluate several (min_samples, xi) combinations on *scaled feature* space.
   - Fit final OPTICS and predict labels on t-SNE space (as written).
9. Visualize t-SNE colored by DBSCAN/HDBSCAN labels and by selected accuracy variables.
10. Export results to ``tsne_with_clusters.csv``.

Outputs
-------
- Directory ``tsne_plots/`` containing t-SNE sweep images:
  ``tsne_perp{perp}_iter{n_iter}.png``
- ``knn_dbscan.png`` : k-distance plot for DBSCAN eps selection
- ``min_samples.png`` : silhouette vs. min_samples curve (DBSCAN)
- Cluster visualizations:
  - ``Cluster_DBSCAN plot``, ``Cluster_HDBSCAN plot``
  - ``t-sne.png`` (unlabeled)
  - ``t-SNE_<feature>`` for each selected accuracy feature
- ``tsne_with_clusters.csv`` : final DataFrame with t-SNE coords and cluster labels

Notes
-----
- Hopkins statistic < 0.5 suggests clustering tendency.
- Silhouette scores are computed only when >= 2 clusters (noise excluded as noted).
- Be mindful of the data size vs. perplexity: t-SNE requires ``perplexity < n_samples``.
- The script uses both t-SNE space (``X_tsne``) and scaled feature space (``X_scaled``)
  for different clustering/diagnostics, as annotated inline.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np
from pyclustertend import hopkins
import hdbscan
from tqdm import tqdm
import os

output_dir = "tsne_plots"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("files_dataframe_all_info.csv")

# Features to include in t-SNE
features = [
    'SNR',
    'time',
    'DM',
    #'pulse width',
    'percentage accuracy SNR',
    'percentage accuracy DM',
    'percentage accuracy time',
    #'percentage accuracy pulse width',
    'DM tolerance',
    'boxcar width',
    'runtime'
]
################# CLEAN AND STANDARDISE DATA ##################
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
###############################################################

############################## HOPKINS STATISTIC #################################
hopkins_score = hopkins(X_scaled, X_scaled.shape[0])
print(f"Hopkins statistic: {hopkins_score:.4f}")

if hopkins_score < 0.5:
    print("Clustering tendency detected.")
else:
    print("Low clustering tendency — results from clustering may not be meaningful.")
##################################################################################

######################## APPLY T-SNE CALCULATION ##########################

# Hyperparameter ranges
perplexities = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
iterations = [500, 1000, 2000, 5000]

# Store results
results = []

for perp in perplexities:
   for n_iter in iterations:
       print(f"[Running t-SNE] Perplexity={perp}, Iterations={n_iter}")

       # ---- Run t-SNE grid search ----
       tsne = TSNE(
           n_components=2,
           perplexity=perp,
           max_iter=n_iter,
           learning_rate='auto',
           method='barnes_hut',
           random_state=42,
           verbose=0
       )

       Y = tsne.fit_transform(X_scaled)
       kl = tsne.kl_divergence_

       print(f"KL divergence: {kl:.4f}")

       results.append({
           'perplexity': perp,
           'iterations': n_iter,
           'kl_divergence': kl
       })

       # Plot for visual inspection
       plt.figure(figsize=(10, 8))
       plt.scatter(Y[:, 0], Y[:, 1], s=5, alpha=0.6)
       plt.title(f"t-SNE (Perp={perp}, Iter={n_iter}) | KL={kl:.3f}")
       plt.xlabel("Dim 1")
       plt.ylabel("Dim 2")
       plt.tight_layout()
       plt.savefig(os.path.join(output_dir, f"tsne_perp{perp}_iter{n_iter}.png"))
       plt.close()



print("RUNNING OPTIMAL HYPERPARAMETERS T-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=55, # change according to plot
    max_iter=5000, # n_iter was renamed to max_iter
    learning_rate='auto',
    method='barnes_hut',
    init='pca',
    random_state=42,
    verbose=0
)
tsne_results = tsne.fit_transform(X_scaled)
df['TSNE-1'] = tsne_results[:, 0]
df['TSNE-2'] = tsne_results[:, 1]

###########################################################################

########################### APPLY CLUSTERING ##################################

# === DBSCAN CLUSTERING ===
print("RUNNING DBSCAN...")

# obtaining value of k
dim_p = len(features)
min_pts = 2 * dim_p
kappa = min_pts - 1

# k-Distance Graph for DBSCAN (kth Nearest Neighbor) to choose eps value
neighbors = NearestNeighbors(n_neighbors=kappa)
neighbors_fit = neighbors.fit(df[['TSNE-1', 'TSNE-2']])
distances, indices = neighbors_fit.kneighbors(df[['TSNE-1', 'TSNE-2']])

# Plot sorted distances
distances = np.sort(distances[:, (kappa-1)]) # -1 due to how python counts
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.xlabel('Data Points sorted by Nearest Neighbour Distance')
plt.ylabel(f'{kappa}th Nearest Neighbor Distance')
plt.title('k-Distance Graph for DBSCAN')
plt.grid(True)
plt.savefig('knn_dbscan.png')
plt.show()

X_tsne = df[['TSNE-1', 'TSNE-2']].values
eps_value = 3  # use your optimal eps from k-distance plot
min_samples_range = range(3, 31)  # try from 3 to 30
scores = []

print("Evaluating DBSCAN over different min_samples...")

for min_s in min_samples_range:
   dbscan = DBSCAN(eps=eps_value, min_samples=min_s)
   labels = dbscan.fit_predict(X_tsne)

   # Only calculate silhouette if at least 2 clusters (ignoring noise)
   if len(set(labels)) > 1 and -1 in labels and len(set(labels)) > 2:
       score = silhouette_score(X_tsne, labels)
       scores.append(score)
       print(f"min_samples={min_s} → silhouette={score:.4f}, clusters={len(set(labels)) - 1}")
   elif len(set(labels)) > 1 and -1 not in labels:
       score = silhouette_score(X_tsne, labels)
       scores.append(score)
       print(f"min_samples={min_s} → silhouette={score:.4f}, clusters={len(set(labels))}")
   else:
       scores.append(-1)
       print(f"min_samples={min_s} → Not enough clusters")

# Plot scores
plt.figure(figsize=(8, 5))
plt.plot(min_samples_range, scores, marker='o')
plt.xlabel('min_samples')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. min_samples (DBSCAN)')
plt.grid(True)
plt.savefig('min_samples.png')
plt.show()

# Print optimal min_samples
best_index = np.argmax(scores)
optimal_min_samples = min_samples_range[best_index]
print(f"Optimal min_samples based on silhouette score: {optimal_min_samples}")


# === HDBSCAN Testing ===

min_cluster_sizes = [250, 300, 350, 400, 450, 500]

print("Testing HDBSCAN for multiple min_cluster_size values:")
for mcs in min_cluster_sizes:
   hdb = hdbscan.HDBSCAN(min_cluster_size=mcs)
   labels = hdb.fit_predict(X_tsne) # CHECK IF X_SCALED OR X_TSNE
   n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise = list(labels).count(-1)
   if n_clusters > 1:
       score = silhouette_score(X_tsne, labels) # CHECK IF X_SCALED OR X_TSNE
       print(f"min_cluster_size={mcs} → clusters: {n_clusters}, noise: {n_noise}, silhouette: {score:.4f}")
   else:
       print(f"min_cluster_size={mcs} → insufficient clusters")


# === DBSCAN Clustering ===
dbscan = DBSCAN(eps=eps_value, min_samples=25)#min_pts) # min_samples used to be optimal_min_samples
df['Cluster_DBSCAN'] = dbscan.fit_predict(df[['TSNE-1', 'TSNE-2']])

# === HDBSCAN Clustering ===
print("Running HDBSCAN...")
hdb = hdbscan.HDBSCAN(min_cluster_size= 450) # check for variables (silhouette score?)
df['Cluster_HDBSCAN'] = hdb.fit_predict(df[['TSNE-1', 'TSNE-2']])

# === OPTICS Testing ===

print("Evaluating OPTICS with varying min_samples and xi:")
min_samples_list = [5, 10, 20, 50]
xi_values = [0.001, 0.005, 0.01, 0.05]

for ms in min_samples_list:
   for xi_val in xi_values:
       optics = OPTICS(min_samples=ms, xi=xi_val, min_cluster_size=0.05)
       labels = optics.fit_predict(X_scaled)
       n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
       n_noise = list(labels).count(-1)
       if n_clusters > 1:
           score = silhouette_score(X_scaled, labels)
           print(f"min_samples={ms}, xi={xi_val} → clusters: {n_clusters}, noise: {n_noise}, silhouette: {score:.4f}")
       else:
           print(f"min_samples={ms}, xi={xi_val} → insufficient clusters")


# === OPTICS Clustering ===
print("Running OPTICS...")
optics = OPTICS(min_samples=50, xi=0.001, min_cluster_size=0.5) # check for variables
df['Cluster_OPTICS'] = optics.fit_predict(df[['TSNE-1', 'TSNE-2']])

###############################################################################

############################ APPLY T-SNE VISUALISATION ###################################

for label in ['Cluster_DBSCAN', 'Cluster_HDBSCAN']: #, 'Cluster_OPTICS']:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='TSNE-1', y='TSNE-2', hue=label, palette='tab20', data=df, alpha=0.8)
    plt.title(f't-SNE Visualisation Colored by {label}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f'{label} plot')
    plt.show()

plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE-1', y='TSNE-2', data=df, alpha=0.8)
plt.title(f't-SNE Visualisation')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)
plt.savefig('t-sne.png')
plt.show()

##########################################################################################


########################## SILHOUETTE SCORE ###############################

from sklearn.metrics import silhouette_score

print("Silhouette Scores:")

# DBSCAN
if len(set(df['Cluster_DBSCAN'])) > 1:
    score_dbscan = silhouette_score(X_tsne, df['Cluster_DBSCAN'])
    print(f"DBSCAN: {score_dbscan:.4f}")
else:
    print("DBSCAN: Only 1 cluster detected — silhouette score not applicable.")

# HDBSCAN
if len(set(df['Cluster_HDBSCAN'])) > 1:
    score_hdbscan = silhouette_score(X_tsne, df['Cluster_HDBSCAN'])
    print(f"HDBSCAN: {score_hdbscan:.4f}")
else:
    print("HDBSCAN: Only 1 cluster detected — silhouette score not applicable.")

# OPTICS
if len(set(df['Cluster_OPTICS'])) > 1:
   score_optics = silhouette_score(X_scaled, df['Cluster_OPTICS'])
   print(f"OPTICS: {score_optics:.4f}")
else:
   print("OPTICS: Only 1 cluster detected — silhouette score not applicable.")

###########################################################################


################# VISUALISE ACCORDING TO VARIABLES ######################

visual_features = ['percentage accuracy DM', 'percentage accuracy time', 'percentage accuracy SNR']

for feature in visual_features:
    df_filtered = df[(df[feature] >= 0) & (df[feature] <= 100)]
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='TSNE-1', y='TSNE-2',
        hue=feature,
        palette='viridis',
        data=df_filtered,
        alpha=0.8
    )
    plt.title(f't-SNE Colored by {feature}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f't-SNE_{feature}')
    plt.show()

#########################################################################

# Export final results
df.to_csv("tsne_with_clusters.csv", index=False)
print("Exported DataFrame to tsne_with_clusters.csv")
