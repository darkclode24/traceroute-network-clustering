#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs a comparative analysis of 10 different clustering algorithms on the
cleaned_traceroute_with_host.csv dataset.

This script processes the data, runs each clustering model, evaluates its
performance using the Silhouette Score, and saves visualizations (PCA and Radar plots)
and clustered data to the 'output/' directory.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    MiniBatchKMeans,
    OPTICS,
    MeanShift,
    AffinityPropagation,
    SpectralClustering,
    estimate_bandwidth
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

# --- Global Constants ---
DATA_FILE = 'cleaned_traceroute_with_host.csv'
OUTPUT_DIR = 'output'
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# Features to be used for clustering
FEATURES = ['Hop', 'Loss%', 'Snt', 'Last', 'Avg', 'Best', 'Wrst', 'StDev']
# Features for radar plot visualization
RADAR_FEATURES = ['Loss%', 'Avg', 'Best', 'Wrst', 'StDev', 'Hop']

# --- Model-Specific Parameters ---
# Parameters for models that require a pre-defined K
K_OPTIMAL = 4
GMM_K = 5
AGGLO_K = 2

# Parameters for density-based models
DBSCAN_EPS = 3.0
DBSCAN_MIN_SAMPLES = 16  # 2 * len(FEATURES)

# Parameters for sampling (for slow models)
SAMPLE_SIZE_SPECTRAL = 10000
SAMPLE_SIZE_MEANSHIFT = 10000
SAMPLE_SIZE_AFFINITY = 10000
MEANSHIFT_BANDWIDTH = 4.0
AFFINITY_DAMPING = 0.9

# --- Utility Functions ---

def setup_output_directory(dir_name=OUTPUT_DIR):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created output directory: {dir_name}")

def load_and_scale_data(file_name, feature_list):
    """
    Loads the data, checks for features, and scales it.
    Returns the original df, scaled data (X_scaled), and the scaler object.
    """
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_name}'")
        return None, None, None

    # Verify all features are present
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in data: {missing_features}")
        return None, None, None
    
    print("Data loaded successfully.")
    X = df[feature_list]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data scaled. Shape: {X_scaled.shape}")
    
    return df, X_scaled, scaler

def plot_pca(df_plot, title, filename):
    """
    Generates and saves a 2D PCA scatter plot.
    Assumes df_plot has 'PC1', 'PC2', and 'cluster' columns.
    """
    plt.figure(figsize=(14, 9))
    
    # Handle noise (-1) cluster from DBSCAN/OPTICS
    hue_order = sorted(df_plot['cluster'].unique())
    palette = sns.color_palette('deep', len(hue_order))
    
    legend_labels = []
    if -1 in hue_order:
        # Make noise grey and put it last
        noise_index = hue_order.index(-1)
        palette[noise_index] = (0.5, 0.5, 0.5, 0.5) # Grey
        legend_labels = [f"Cluster {l}" if l != -1 else "Noise" for l in hue_order]
    else:
        legend_labels = [f"Cluster {l}" for l in hue_order]

    ax = sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='cluster',
        hue_order=hue_order,
        palette=palette,
        alpha=0.7,
        s=40
    )
    
    if len(hue_order) > 50:
        ax.legend_ = None
        print(f"Skipping legend for {title} (>50 clusters).")
    else:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot: {filename}")

def plot_radar_chart(stats_df, radar_features, title, filename):
    """
    Generates and saves a radar chart of cluster profiles.
    Assumes stats_df index is cluster ID and columns are features + 'Population'.
    """
    if stats_df.empty:
        print(f"Skipping radar plot for {title}: No clusters found.")
        return
        
    stats_to_plot = stats_df[radar_features]
    
    # Normalize data 0-1 for radar plot
    scaler_radar = MinMaxScaler()
    
    if stats_to_plot.shape[0] == 1:
        # Special handling for single-row dataframe
        stats_normalized = scaler_radar.fit_transform(stats_to_plot.values.reshape(1, -1))
    else:
        stats_normalized = scaler_radar.fit_transform(stats_to_plot)
        
    stats_normalized = pd.DataFrame(stats_normalized, columns=radar_features, index=stats_df.index)

    labels = stats_normalized.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    colors = sns.color_palette('deep', stats_normalized.shape[0])

    for idx, (cluster_num, row) in enumerate(stats_normalized.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        pop = int(stats_df.loc[cluster_num, 'Population'])

        ax.plot(angles, values, color=colors[idx], linewidth=2, label=f"Cluster {cluster_num} (n={pop})")
        ax.fill(angles, values, color=colors[idx], alpha=0.2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    plt.title(title, size=20, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved Radar plot: {filename}")

def get_cluster_summary(df, cluster_labels, features):
    """
    Adds cluster labels to a DataFrame and returns the summary statistics.
    """
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    summary = df_clustered.groupby('cluster')[features].mean()
    summary['Population'] = df_clustered['cluster'].value_counts()
    
    return df_clustered, summary

def get_pca_df(df_clustered, X_scaled):
    """
    Performs PCA and adds PC1, PC2 to the clustered DataFrame.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clustered['PC1'] = X_pca[:, 0]
    df_clustered['PC2'] = X_pca[:, 1]
    return df_clustered

# --- Model Pre-analysis Functions ---

def find_optimal_k_elbow(X_scaled, k_range, output_dir):
    """Plots the Elbow method to help find optimal K."""
    print("Running Elbow Method...")
    inertia = []
    for k in k_range:
        kmeans_elbow = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        kmeans_elbow.fit(X_scaled)
        inertia.append(kmeans_elbow.inertia_)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list(k_range), y=inertia, marker='o', markersize=8)
    plt.title('Elbow Method for Optimal K', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Sum of squared distances)', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    filename = os.path.join(output_dir, 'kmeans_elbow_plot.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved Elbow plot: {filename}")

def find_dbscan_eps(X_scaled, min_samples, output_dir):
    """Plots the k-NN distance plot to help find optimal eps for DBSCAN."""
    print("Running k-NN distance plot for DBSCAN...")
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1], axis=0)

    plt.figure(figsize=(12, 7))
    sns.lineplot(x=np.arange(len(k_distances)), y=k_distances)
    plt.title(f'k-NN Distance Plot (k = {min_samples}) for Epsilon', fontsize=16)
    plt.xlabel('Data Points (sorted by distance)', fontsize=12)
    plt.ylabel(f'Distance to {min_samples}-th Neighbor (eps)', fontsize=12)
    
    filename = os.path.join(output_dir, 'dbscan_knn_plot.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved k-NN plot: {filename}")

def find_gmm_components(X_scaled, k_range, output_dir):
    """Plots AIC/BIC scores to find the optimal number of GMM components."""
    print("Running AIC/BIC analysis for GMM...")
    aic_scores = []
    bic_scores = []
    
    for k in k_range:
        gmm = GaussianMixture(n_components=k, n_init=5, random_state=RANDOM_STATE)
        gmm.fit(X_scaled)
        aic_scores.append(gmm.aic(X_scaled))
        bic_scores.append(gmm.bic(X_scaled))
        print(f"GMM analysis for K={k} complete.")

    df_scores = pd.DataFrame({'K': k_range, 'AIC': aic_scores, 'BIC': bic_scores})
    df_melted = df_scores.melt('K', var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_melted, x='K', y='Score', hue='Metric', marker='o', markersize=8)
    plt.title('AIC and BIC Scores for GMM', fontsize=16)
    plt.xlabel('Number of Components (K)', fontsize=12)
    plt.ylabel('Score (Lower is better)', fontsize=12)
    plt.xticks(k_range)
    
    filename = os.path.join(output_dir, 'gmm_aic_bic_plot.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved GMM AIC/BIC plot: {filename}")

# --- Core Model Functions ---

def run_kmeans(X_scaled, df, features, k, output_dir):
    """Runs K-Means, saves plots and data, returns silhouette score."""
    print("\n--- Running K-Means ---")
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    clusters = model.fit_predict(X_scaled)
    
    df_clustered, summary = get_cluster_summary(df, clusters, features)
    score = silhouette_score(X_scaled, clusters)
    print(f"K-Means (K={k}) Silhouette Score: {score:.4f}")
    
    df_pca = get_pca_df(df_clustered, X_scaled)
    
    plot_pca(df_pca, f'K-Means Clustering (K={k})', os.path.join(output_dir, 'kmeans_pca_plot.png'))
    plot_radar_chart(summary, RADAR_FEATURES, f'K-Means Profiles (K={k})', os.path.join(output_dir, 'kmeans_radar_plot.png'))
    df_clustered.to_csv(os.path.join(output_dir, 'kmeans_clustered_data.csv'), index=False)
    
    return score

def run_dbscan(X_scaled, df, features, eps, min_samples, output_dir):
    """Runs DBSCAN, saves plots and data, returns silhouette score."""
    print("\n--- Running DBSCAN ---")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X_scaled)
    
    df_clustered, summary = get_cluster_summary(df, clusters, features)
    
    # Filter out noise for silhouette score and radar plot
    non_noise_mask = (clusters != -1)
    if len(np.unique(clusters[non_noise_mask])) > 1:
        score = silhouette_score(X_scaled[non_noise_mask], clusters[non_noise_mask])
        print(f"DBSCAN (eps={eps}, min={min_samples}) Silhouette Score (non-noise): {score:.4f}")
        summary_no_noise = summary.drop(index=-1, errors='ignore')
    else:
        score = -1
        print("DBSCAN found < 2 clusters. Silhouette score not applicable.")
        summary_no_noise = pd.DataFrame()

    n_noise = list(clusters).count(-1)
    print(f"DBSCAN found {n_noise} noise points ({n_noise/len(clusters):.1%}).")

    df_pca = get_pca_df(df_clustered, X_scaled)
    plot_pca(df_pca, f'DBSCAN Clustering (eps={eps})', os.path.join(output_dir, 'dbscan_pca_plot.png'))
    plot_radar_chart(summary_no_noise, RADAR_FEATURES, 'DBSCAN Profiles (Non-Noise)', os.path.join(output_dir, 'dbscan_radar_plot.png'))
    df_clustered.to_csv(os.path.join(output_dir, 'dbscan_clustered_data.csv'), index=False)
    
    return score

def run_agglomerative(X_scaled, df, features, k, output_dir):
    """Runs Agglomerative Clustering, saves plots and data, returns silhouette score."""
    print("\n--- Running Agglomerative Clustering ---")
    model = AgglomerativeClustering(n_clusters=k)
    clusters = model.fit_predict(X_scaled)
    
    df_clustered, summary = get_cluster_summary(df, clusters, features)
    score = silhouette_score(X_scaled, clusters)
    print(f"Agglomerative (K={k}) Silhouette Score: {score:.4f}")
    
    df_pca = get_pca_df(df_clustered, X_scaled)
    
    plot_pca(df_pca, f'Agglomerative Clustering (K={k})', os.path.join(output_dir, 'agglo_pca_plot.png'))
    plot_radar_chart(summary, RADAR_FEATURES, f'Agglomerative Profiles (K={k})', os.path.join(output_dir, 'agglo_radar_plot.png'))
    df_clustered.to_csv(os.path.join(output_dir, 'agglo_clustered_data.csv'), index=False)
    
    return score

def run_birch(X_scaled, df, features, k, output_dir):
    """Runs BIRCH, saves plots and data, returns silhouette score."""
    print("\n--- Running BIRCH ---")
    model = Birch(n_clusters=k)
    clusters = model.fit_predict(X_scaled)
    
    df_clustered, summary = get_cluster_summary(df, clusters, features)
    score = silhouette_score(X_scaled, clusters)
    print(f"BIRCH (K={k}) Silhouette Score: {score:.4f}")
    
    df_pca = get_pca_df(df_clustered, X_scaled)
    
    plot_pca(df_pca, f'BIRCH Clustering (K={k})', os.path.join(output_dir, 'birch_pca_plot.png'))
    plot_radar_chart(summary, RADAR_FEATURES, f'BIRCH Profiles (K={k})', os.path.join(output_dir, 'birch_radar_plot.png'))
    df_clustered.to_csv(os.path.join(output_dir, 'birch_clustered_data.csv'), index=False)
    
    return score

def run_minibatch_kmeans(X_scaled, df, features, k, output_dir):
    """Runs Mini-Batch K-Means, saves plots and data, returns silhouette score."""
    print("\n--- Running Mini-Batch K-Means ---")
    model = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    clusters = model.fit_predict(X_scaled)
    
    df_clustered, summary = get_cluster_summary(df, clusters, features)
    score = silhouette_score(X_scaled, clusters)
    print(f"Mini-Batch K-Means (K={k}) Silhouette Score: {score:.4f}")
    
    df_pca = get_pca_df(df_clustered, X_scaled)
    
    plot_pca(df_pca, f'Mini-Batch K-Means (K={k})', os.path.join(output_dir, 'minibatch_pca_plot.png'))
    plot_radar_chart(summary, RADAR_FEATURES, f'Mini-Batch K-Means Profiles (K={k})', os.path.join(output_dir, 'minibatch_radar_plot.png'))
    df_clustered.to_csv(os.path.join(output_dir, 'minibatch_clustered_data.csv'), index=False)
    
    return score

def run_optics(X_scaled, df, features, eps, min_samples, output_dir):
    """Runs OPTICS, saves plots and data, returns silhouette score."""
    print("\n--- Running OPTICS ---")
    # Use cluster_method='dbscan' to extract flat clusters
    model = OPTICS(min_samples=min_samples, cluster_method='dbscan', eps=eps)
    clusters = model.fit_predict(X_scaled)
    
    df_clustered, summary = get_cluster_summary(df, clusters, features)
    
    non_noise_mask = (clusters != -1)
    if len(np.unique(clusters[non_noise_mask])) > 1:
        score = silhouette_score(X_scaled[non_noise_mask], clusters[non_noise_mask])
        print(f"OPTICS (eps={eps}, min={min_samples}) Silhouette Score (non-noise): {score:.4f}")
        summary_no_noise = summary.drop(index=-1, errors='ignore')
    else:
        score = -1
        print("OPTICS found < 2 clusters. Silhouette score not applicable.")
        summary_no_noise = pd.DataFrame()

    n_noise = list(clusters).count(-1)
    print(f"OPTICS found {n_noise} noise points ({n_noise/len(clusters):.1%}).")
    
    df_pca = get_pca_df(df_clustered, X_scaled)
    plot_pca(df_pca, f'OPTICS Clustering (eps={eps})', os.path.join(output_dir, 'optics_pca_plot.png'))
    plot_radar_chart(summary_no_noise, RADAR_FEATURES, 'OPTICS Profiles (Non-Noise)', os.path.join(output_dir, 'optics_radar_plot.png'))
    df_clustered.to_csv(os.path.join(output_dir, 'optics_clustered_data.csv'), index=False)
    
    return score

def run_mean_shift(X_scaled, df, features, bandwidth, sample_size, output_dir):
    """Runs Mean Shift on a sample, saves plots, returns silhouette score."""
    print("\n--- Running Mean Shift ---")
    print(f"Using sample of {sample_size} for Mean Shift.")
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_sample = X_scaled[sample_indices]
    df_sample = df.iloc[sample_indices]
    
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clusters = model.fit_predict(X_sample)
    
    df_clustered, summary = get_cluster_summary(df_sample, clusters, features)
    n_clusters = len(summary)
    
    if n_clusters > 1:
        score = silhouette_score(X_sample, clusters)
        print(f"Mean Shift (bandwidth={bandwidth}) found {n_clusters} clusters. Silhouette Score: {score:.4f}")
    else:
        score = -1
        print(f"Mean Shift found {n_clusters} cluster. Silhouette score not applicable.")
    
    # Need to run PCA on the sample data
    pca = PCA(n_components=2)
    X_pca_sample = pca.fit_transform(X_sample)
    df_clustered['PC1'] = X_pca_sample[:, 0]
    df_clustered['PC2'] = X_pca_sample[:, 1]
    
    plot_pca(df_clustered, f'Mean Shift (Sample={sample_size}, bw={bandwidth})', os.path.join(output_dir, 'meanshift_pca_plot.png'))
    plot_radar_chart(summary, RADAR_FEATURES, f'Mean Shift Profiles (Sample={sample_size})', os.path.join(output_dir, 'meanshift_radar_plot.png'))
    
    return score

def run_gmm(X_scaled, df, features, k, output_dir):
    """Runs Gaussian Mixture Model, saves plots and data, returns silhouette score."""
    print("\n--- Running Gaussian Mixture Model (GMM) ---")
    model = GaussianMixture(n_components=k, n_init=5, random_state=RANDOM_STATE)
    clusters = model.fit_predict(X_scaled)
    
    df_clustered, summary = get_cluster_summary(df, clusters, features)
    score = silhouette_score(X_scaled, clusters)
    print(f"GMM (K={k}) Silhouette Score: {score:.4f}")
    
    df_pca = get_pca_df(df_clustered, X_scaled)
    
    plot_pca(df_pca, f'GMM Clustering (K={k})', os.path.join(output_dir, 'gmm_pca_plot.png'))
    plot_radar_chart(summary, RADAR_FEATURES, f'GMM Profiles (K={k})', os.path.join(output_dir, 'gmm_radar_plot.png'))
    df_clustered.to_csv(os.path.join(output_dir, 'gmm_clustered_data.csv'), index=False)
    
    return score

def run_affinity_propagation(X_scaled, df, features, damping, sample_size, output_dir):
    """Runs Affinity Propagation on a sample, saves plots, returns silhouette score."""
    print("\n--- Running Affinity Propagation ---")
    print(f"Using sample of {sample_size} for Affinity Propagation.")
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_sample = X_scaled[sample_indices]
    df_sample = df.iloc[sample_indices]
    
    model = AffinityPropagation(damping=damping, random_state=RANDOM_STATE)
    clusters = model.fit_predict(X_sample)
    
    df_clustered, summary = get_cluster_summary(df_sample, clusters, features)
    n_clusters = len(summary)
    
    if n_clusters > 1:
        score = silhouette_score(X_sample, clusters)
        print(f"Affinity Prop (damping={damping}) found {n_clusters} clusters. Silhouette Score: {score:.4f}")
    else:
        score = -1
        print(f"Affinity Prop found {n_clusters} cluster. Silhouette score not applicable.")

    pca = PCA(n_components=2)
    X_pca_sample = pca.fit_transform(X_sample)
    df_clustered['PC1'] = X_pca_sample[:, 0]
    df_clustered['PC2'] = X_pca_sample[:, 1]
    
    plot_pca(df_clustered, f'Affinity Prop (Sample={sample_size})', os.path.join(output_dir, 'affinity_pca_plot.png'))
    
    if n_clusters > 10:
        print("Skipping radar plot for Affinity Prop: Too many clusters (>10).")
    else:
        plot_radar_chart(summary, RADAR_FEATURES, f'Affinity Prop Profiles (Sample={sample_size})', os.path.join(output_dir, 'affinity_radar_plot.png'))
    
    return score

def run_spectral(X_scaled, df, features, k, sample_size, output_dir):
    """Runs Spectral Clustering on a sample, saves plots, returns silhouette score."""
    print("\n--- Running Spectral Clustering ---")
    print(f"Using sample of {sample_size} for Spectral Clustering.")
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_sample = X_scaled[sample_indices]
    df_sample = df.iloc[sample_indices]
    
    model = SpectralClustering(n_clusters=k, assign_labels='kmeans', random_state=RANDOM_STATE)
    clusters = model.fit_predict(X_sample)
    
    df_clustered, summary = get_cluster_summary(df_sample, clusters, features)
    
    score = silhouette_score(X_sample, clusters)
    print(f"Spectral (K={k}, Sample={sample_size}) Silhouette Score: {score:.4f}")

    pca = PCA(n_components=2)
    X_pca_sample = pca.fit_transform(X_sample)
    df_clustered['PC1'] = X_pca_sample[:, 0]
    df_clustered['PC2'] = X_pca_sample[:, 1]
    
    plot_pca(df_clustered, f'Spectral Clustering (Sample={sample_size}, K={k})', os.path.join(output_dir, 'spectral_pca_plot.png'))
    plot_radar_chart(summary, RADAR_FEATURES, f'Spectral Profiles (Sample={sample_size})', os.path.join(output_dir, 'spectral_radar_plot.png'))
    
    return score

def print_summary_report(scores):
    """Prints the final performance summary table."""
    print("\n\n" + "="*80)
    print("CLUSTERING MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Create a DataFrame for easy sorting and printing
    report_df = pd.DataFrame.from_dict(scores, orient='index')
    report_df.columns = ['Silhouette Score', 'Parameters', 'Dataset']
    report_df = report_df.sort_values(by='Silhouette Score', ascending=False)
    
    print(report_df.to_markdown(floatfmt=".4f"))
    
    print("\n--- Notes ---")
    print("1. Silhouette Score: Measures cluster density and separation. Higher is better (Max: 1.0).")
    print("2. 'Sample' dataset means the model was too slow to run on the full 22k rows.")
    print("3. DBSCAN/OPTICS score is for non-noise points. Their main finding was 66% 'Noise'.")
    print("="*80)

# --- Main Execution ---

def main():
    """
    Main function to run the complete clustering analysis pipeline.
    """
    setup_output_directory(OUTPUT_DIR)
    
    df, X_scaled, scaler = load_and_scale_data(DATA_FILE, FEATURES)
    if df is None:
        return
        
    performance_scores = {}
    
    # --- Pre-analysis  ---
    k_range = range(2, 11)
    find_optimal_k_elbow(X_scaled, k_range, OUTPUT_DIR)
    find_dbscan_eps(X_scaled, DBSCAN_MIN_SAMPLES, OUTPUT_DIR)
    find_gmm_components(X_scaled, range(2, 8), OUTPUT_DIR)
    
    # --- Run Models ---
    
    # K-Means
    score = run_kmeans(X_scaled, df, FEATURES, K_OPTIMAL, OUTPUT_DIR)
    performance_scores['K-Means'] = [score, f"K={K_OPTIMAL}", "Full"]
    
    # DBSCAN
    score = run_dbscan(X_scaled, df, FEATURES, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, OUTPUT_DIR)
    performance_scores['DBSCAN'] = [score, f"eps={DBSCAN_EPS}, min={DBSCAN_MIN_SAMPLES}", "Full (Non-Noise)"]

    # Agglomerative
    score = run_agglomerative(X_scaled, df, FEATURES, AGGLO_K, OUTPUT_DIR)
    performance_scores['Agglomerative'] = [score, f"K={AGGLO_K}", "Full"]

    # BIRCH
    score = run_birch(X_scaled, df, FEATURES, K_OPTIMAL, OUTPUT_DIR)
    performance_scores['BIRCH'] = [score, f"K={K_OPTIMAL}", "Full"]

    # Mini-Batch K-Means
    score = run_minibatch_kmeans(X_scaled, df, FEATURES, K_OPTIMAL, OUTPUT_DIR)
    performance_scores['Mini-Batch K-Means'] = [score, f"K={K_OPTIMAL}", "Full"]

    # OPTICS
    score = run_optics(X_scaled, df, FEATURES, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, OUTPUT_DIR)
    performance_scores['OPTICS'] = [score, f"eps={DBSCAN_EPS}, min={DBSCAN_MIN_SAMPLES}", "Full (Non-Noise)"]

    # GMM
    score = run_gmm(X_scaled, df, FEATURES, GMM_K, OUTPUT_DIR)
    performance_scores['GMM'] = [score, f"K={GMM_K}", "Full"]

    # --- Run Sampling-based Models ---
    
    # Mean Shift
    score = run_mean_shift(X_scaled, df, FEATURES, MEANSHIFT_BANDWIDTH, SAMPLE_SIZE_MEANSHIFT, OUTPUT_DIR)
    performance_scores['Mean Shift'] = [score, f"bw={MEANSHIFT_BANDWIDTH}", f"Sample ({SAMPLE_SIZE_MEANSHIFT})"]
    
    # Affinity Propagation
    score = run_affinity_propagation(X_scaled, df, FEATURES, AFFINITY_DAMPING, SAMPLE_SIZE_AFFINITY, OUTPUT_DIR)
    performance_scores['Affinity Prop.'] = [score, f"damping={AFFINITY_DAMPING}", f"Sample ({SAMPLE_SIZE_AFFINITY})"]

    # Spectral Clustering
    score = run_spectral(X_scaled, df, FEATURES, K_OPTIMAL, SAMPLE_SIZE_SPECTRAL, OUTPUT_DIR)
    performance_scores['Spectral'] = [score, f"K={K_OPTIMAL}", f"Sample ({SAMPLE_SIZE_SPECTRAL})"]

    # --- Final Report ---
    print_summary_report(performance_scores)

if __name__ == "__main__":
    main()