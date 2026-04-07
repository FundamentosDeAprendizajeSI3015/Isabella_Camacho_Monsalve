# ==========================================
# STUDENT-CAREER ADJUSTMENT CLUSTERING ANALYSIS
# Machine Learning Workshop - Clustering + Supervised Models
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px # new import

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from fcmeans import FCM  # For Fuzzy C-Means
from sklearn.cluster import SpectralClustering

# Preprocessing and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Supervised models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import warnings
warnings.filterwarnings('ignore')

# ===========================================
# GLOBAL CONFIGURATION
# ===========================================

# Data paths
DATA_DIR = Path("../Informe1/extended_ds")
X_RAW_PATH = DATA_DIR / "X_raw.parquet"
Y_PATH = DATA_DIR / "y.parquet"

# Output directory
# Adjusted to be relative to the script's location within Isabella_Camacho_Monsalve/Informe2
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Random state for reproducibility
RANDOM_STATE = 42

# ===========================================
# 1. DATA LOADING AND PREPROCESSING
# ===========================================

print("="*60)
print("STUDENT-CAREER ADJUSTMENT CLUSTERING ANALYSIS")
print("="*60)

# Load data
print("\n1. LOADING DATA")
print("-" * 30)

X_raw = pd.read_parquet(X_RAW_PATH)
y_raw = pd.read_parquet(Y_PATH)['high_adjustment']

print(f"Features shape: {X_raw.shape}")
print(f"Target shape: {y_raw.shape}")
print(f"Features: {list(X_raw.columns)}")
print(f"\nClass distribution:")
print(y_raw.value_counts())
print(f"Class proportions:")
print(y_raw.value_counts(normalize=True).round(3))

# Check for any remaining missing values
print(f"\nMissing values in features: {X_raw.isnull().sum().sum()}")
print(f"Missing values in target: {y_raw.isnull().sum()}")

# Remove any samples with missing values
mask = ~(X_raw.isnull().any(axis=1) | y_raw.isnull())
X_raw = X_raw[mask]
y_raw = y_raw[mask]
y = y_raw.copy()

print(f"Final dataset shape after cleaning: {X_raw.shape}")

# ===========================================
# 2. FEATURE SCALING AND DIMENSIONALITY REDUCTION
# ===========================================

print("\n2. PREPROCESSING FOR CLUSTERING")
print("-" * 30)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

print(f"Features scaled to mean=0, std=1")
print(f"Scaled features shape: {X_scaled.shape}")

# Apply PCA for dimensionality reduction (95% variance)
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA applied - Original dimensions: {X_scaled.shape[1]}")
print(f"PCA dimensions (95% variance): {X_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

# Create sample for clustering 
# Given the user mentioned "pocas muestras", the existing logic
# already handles using the full dataset if smaller than 1000.
sample_size = min(len(X_raw), 1000)  # Use full dataset if smaller than 1000
if len(X_raw) <= sample_size:
    print(f"\nUsing full dataset for clustering (n={len(X_raw)})")
    X_cluster = X_pca
    y_cluster = y_raw
    sample_indices = X_raw.index
else:
    print(f"\nCreating stratified sample for clustering (n={sample_size})")
    _, X_cluster, _, y_cluster, sample_indices, _ = train_test_split(
        X_pca, y_raw, X_raw.index,
        test_size=sample_size,
        stratify=y_raw,
        random_state=RANDOM_STATE
    )

print(f"Clustering dataset shape: {X_cluster.shape}")
print(f"Clustering target distribution:")
print(y_cluster.value_counts())

# ===========================================
# 3. CLUSTERING ALGORITHMS
# ===========================================

print("\n3. APPLYING CLUSTERING ALGORITHMS")
print("-" * 30)

# Dictionary to store clustering results
clustering_results = {}

# 3.1 K-Means
print("Running K-Means...")
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster)
clustering_results['kmeans'] = kmeans_labels

# 3.2 Fuzzy C-Means
print("Running Fuzzy C-Means...")
try:
    fcm = FCM(n_clusters=2, random_state=RANDOM_STATE)
    fcm.fit(X_cluster)
    fcm_labels = fcm.predict(X_cluster)
    clustering_results['fcm'] = fcm_labels
except Exception as e:
    print(f"Fuzzy C-Means failed: {e}")
    print("Using K-Medoids as alternative...")
    kmedoids = KMedoids(n_clusters=2, random_state=RANDOM_STATE)
    fcm_labels = kmedoids.fit_predict(X_cluster)
    clustering_results['fcm'] = fcm_labels

# 3.3 DBSCAN
print("Running DBSCAN...")
# Try different eps values to find reasonable clusters
eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
best_dbscan = None
best_n_clusters = 0

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=3)
    dbscan_labels = dbscan.fit_predict(X_cluster)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    if 1 < n_clusters <= 3 and n_noise < len(X_cluster) * 0.5:  # Reasonable clustering
        best_dbscan = dbscan_labels
        best_n_clusters = n_clusters
        print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise points")
        break

if best_dbscan is not None:
    # Convert to binary (largest cluster vs others)
    unique_labels, counts = np.unique(best_dbscan[best_dbscan != -1], return_counts=True)
    if len(unique_labels) >= 2:
        largest_cluster = unique_labels[np.argmax(counts)]
        dbscan_binary = (best_dbscan != largest_cluster).astype(int)
        # Handle noise as anomalies
        dbscan_binary[best_dbscan == -1] = 1
    else:
        dbscan_binary = (best_dbscan == -1).astype(int)
    clustering_results['dbscan'] = dbscan_binary
else:
    print("DBSCAN could not find reasonable clusters, using spectral clustering...")
    spectral = SpectralClustering(n_clusters=2, random_state=RANDOM_STATE)
    dbscan_binary = spectral.fit_predict(X_cluster)
    clustering_results['dbscan'] = dbscan_binary

# 3.4 Alternative to Subtractive Clustering - Use Spectral Clustering
print("Running Spectral Clustering (alternative to Subtractive)...")
spectral = SpectralClustering(n_clusters=2, random_state=RANDOM_STATE)
spectral_labels = spectral.fit_predict(X_cluster)
clustering_results['spectral'] = spectral_labels

print(f"Completed clustering with {len(clustering_results)} algorithms")

# ===========================================
# 4. CLUSTERING ANALYSIS AND LIFT CALCULATION
# ===========================================

print("\n4. ANALYZING CLUSTERING PATTERNS")
print("-" * 30)

# Calculate baseline rate (rate of high adjustment)
baseline_rate = y_cluster.mean()
print(f"Baseline rate (high adjustment): {baseline_rate:.3f}")

# Dictionary to store lift values
lifts = {}

# Analyze each clustering result
for algorithm, labels in clustering_results.items():
    print(f"\n{algorithm.upper()} Analysis:")
    
    # Ensure binary labels (0, 1)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        # If already binary, keep as is
        binary_labels = labels
    else:
        # Convert to binary by treating smallest cluster as anomalous
        label_counts = pd.Series(labels).value_counts()
        minority_label = label_counts.idxmin()
        binary_labels = (labels == minority_label).astype(int)
    
    # Update with binary labels
    clustering_results[algorithm] = binary_labels
    
    # Calculate statistics for each cluster
    cluster_stats = pd.DataFrame({
        'cluster': binary_labels,
        'target': y_cluster
    }).groupby('cluster')['target'].agg(['count', 'mean']).reset_index()
    
    print(cluster_stats)
    
    # Calculate lift for anomalous cluster (cluster 1)
    if 1 in cluster_stats['cluster'].values:
        anomaly_rate = cluster_stats[cluster_stats['cluster'] == 1]['mean'].iloc[0]
        lift = anomaly_rate / baseline_rate if baseline_rate > 0 else 0
        lifts[algorithm] = max(lift, 1.0)  # Ensure lift >= 1
        print(f"  Lift for anomalous cluster: {lift:.3f}")
    else:
        lifts[algorithm] = 1.0
        print(f"  No anomalous cluster found, using lift = 1.0")

print(f"\nLift values for ensemble weighting: {lifts}")

# ===========================================
# 5. ENSEMBLE-BASED LABEL REEVALUATION
# ===========================================

print("\n5. ENSEMBLE-BASED LABEL REEVALUATION")
print("-" * 30)

# Create ensemble predictions using weighted voting
ensemble_scores = np.zeros(len(y_cluster))

for algorithm, labels in clustering_results.items():
    weight = lifts[algorithm]
    ensemble_scores += weight * labels

# Normalize scores
total_weight = sum(lifts.values())
ensemble_scores = ensemble_scores / total_weight

# Apply conservative threshold for relabeling
# Only add positive labels, never remove existing ones
threshold = 0.5
conservative_relabeling = ensemble_scores > threshold

# Create reevaluated labels
y_reevaluated = y_cluster.copy()
newly_labeled = conservative_relabeling & (y_cluster == 0)
y_reevaluated.loc[newly_labeled] = 1

print(f"Original positive samples: {y_cluster.sum()}")
print(f"Newly labeled positive samples: {newly_labeled.sum()}")
print(f"Total positive samples after reevaluation: {y_reevaluated.sum()}")
print(f"Percentage increase: {((y_reevaluated.sum() - y_cluster.sum()) / y_cluster.sum() * 100):.1f}%")

# ===========================================
# 6. SUPERVISED MODEL TRAINING AND COMPARISON
# ===========================================

print("\n6. SUPERVISED MODEL TRAINING")
print("-" * 30)

# Prepare full dataset for supervised learning
X_full_scaled = scaler.transform(X_raw)

# Create reevaluated labels for full dataset
y_full_original = y_raw.copy()
y_full_reevaluated = y_raw.copy()

# Apply reevaluation to the full dataset using the same logic
if len(sample_indices) < len(X_raw):
    # If we used a sample, apply a simple heuristic to full dataset
    # Use ensemble predictions on full dataset
    print("Applying reevaluation logic to full dataset...")
    X_full_pca = pca.transform(X_full_scaled)
    
    full_ensemble_scores = np.zeros(len(X_raw))
    for algorithm in ['kmeans', 'fcm', 'spectral']:  # Use most reliable algorithms
        if algorithm == 'kmeans':
            full_labels = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(X_full_pca)
        elif algorithm == 'fcm':
            try:
                fcm_full = FCM(n_clusters=2, random_state=RANDOM_STATE)
                fcm_full.fit(X_full_pca)
                full_labels = fcm_full.predict(X_full_pca)
            except:
                full_labels = KMedoids(n_clusters=2, random_state=RANDOM_STATE).fit_predict(X_full_pca)
        else:  # spectral
            full_labels = SpectralClustering(n_clusters=2, random_state=RANDOM_STATE).fit_predict(X_full_pca)
        
        # Convert to binary if needed
        label_counts = pd.Series(full_labels).value_counts()
        if len(label_counts) == 2:
            minority_label = label_counts.idxmin()
            binary_full_labels = (full_labels == minority_label).astype(int)
        else:
            binary_full_labels = full_labels
            
        weight = lifts.get(algorithm, 1.0)
        full_ensemble_scores += weight * binary_full_labels
    
    full_ensemble_scores = full_ensemble_scores / sum([lifts.get(alg, 1.0) for alg in ['kmeans', 'fcm', 'spectral']])
    full_conservative_relabeling = full_ensemble_scores > threshold
    full_newly_labeled = full_conservative_relabeling & (y_raw == 0)
    y_full_reevaluated.loc[full_newly_labeled] = 1
else:
    # If we used full dataset for clustering, apply results directly
    y_full_reevaluated.loc[sample_indices] = y_reevaluated

print(f"Full dataset - Original positive: {y_full_original.sum()}")
print(f"Full dataset - Reevaluated positive: {y_full_reevaluated.sum()}")

# Train-test split
X_train, X_test, y_train_orig, y_test_orig = train_test_split(
    X_full_scaled, y_full_original, test_size=0.2, stratify=y_full_original, random_state=RANDOM_STATE
)

_, _, y_train_reeval, y_test_reeval = train_test_split(
    X_full_scaled, y_full_reevaluated, test_size=0.2, stratify=y_full_reevaluated, random_state=RANDOM_STATE
)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
}

# Results storage
results = []

# Train and evaluate models
for model_name, model in models.items():
    for label_type, (y_tr, y_te) in [('Original', (y_train_orig, y_test_orig)), 
                                     ('Reevaluated', (y_train_reeval, y_test_reeval))]:
        
        print(f"\nTraining {model_name} with {label_type} labels...")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_tr)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)
        auc_roc = roc_auc_score(y_te, y_pred_proba)
        auc_pr = average_precision_score(y_te, y_pred_proba)
        
        # Store results
        results.append({
            'Model': model_name,
            'Labels': label_type,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr
        })
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC-ROC: {auc_roc:.3f}")
        print(f"  AUC-PR: {auc_pr:.3f}")

# ===========================================
# 7. RESULTS COMPARISON TABLE
# ===========================================

print("\n7. RESULTS COMPARISON")
print("-" * 30)

results_df = pd.DataFrame(results)
print("\nComparative Results Table:")
print(results_df.round(3))

# Save results
results_df.to_csv(OUTPUT_DIR / 'model_comparison_results.csv', index=False)

# Calculate improvements
print("\nImprovement Analysis:")
for model_name in models.keys():
    orig_metrics = results_df[(results_df['Model'] == model_name) & (results_df['Labels'] == 'Original')]
    reeval_metrics = results_df[(results_df['Model'] == model_name) & (results_df['Labels'] == 'Reevaluated')]
    
    if not orig_metrics.empty and not reeval_metrics.empty:
        print(f"\n{model_name}:")
        for metric in ['Accuracy', 'F1-Score', 'AUC-ROC', 'AUC-PR']:
            orig_val = orig_metrics[metric].iloc[0]
            reeval_val = reeval_metrics[metric].iloc[0]
            improvement = ((reeval_val - orig_val) / orig_val) * 100
            print(f"  {metric}: {orig_val:.3f} → {reeval_val:.3f} ({improvement:+.1f}%)")

# ===========================================
# 8. VISUALIZATIONS
# ===========================================

print("\n8. CREATING VISUALIZATIONS")
print("-" * 30)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# 8.1 Clustering Results Visualization
# ── PCA compartido para TODAS las visualizaciones ───────────────────────────
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_vis = pca_2d.fit_transform(X_cluster)

pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
X_vis3d = pca_3d.fit_transform(X_cluster)

# ── Clustering visualization (4 subplots) ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

algorithms = list(clustering_results.keys())
for i, algorithm in enumerate(algorithms):
    ax = axes[i]
    labels = clustering_results[algorithm]

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    cmap = plt.cm.get_cmap('viridis', max(n_clusters, 1))

    scatter = ax.scatter(
        X_vis[:, 0], X_vis[:, 1],
        c=labels, cmap=cmap,
        vmin=unique_labels.min(), vmax=unique_labels.max(),
        alpha=0.7
    )
    ax.set_title(f'{algorithm.upper()} Clustering')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels(
        ['Ruido' if l == -1 else f'Cluster {l}' for l in unique_labels]
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'clustering_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Distribución original 3D — UNA sola vez ─────────────────────────────────
y_plot_global = y_cluster.reset_index(drop=True)
fig_orig_global = px.scatter_3d(
    x=X_vis3d[:, 0], y=X_vis3d[:, 1], z=X_vis3d[:, 2],
    color=y_plot_global.astype(str),
    title='Distribución 3D con Etiquetas Originales',
    color_discrete_map={'0': 'blue', '1': 'red'}
)
fig_orig_global.update_layout(scene=dict(
    xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'
))
fig_orig_global.write_html(os.path.join(OUTPUT_DIR, 'dist_3d_original.html'))

# 3.5 Function to map cluster labels to match majority class in y_original
def map_cluster_labels(cluster_labels, true_labels):
    cluster_labels = pd.Series(cluster_labels).reset_index(drop=True)
    true_labels = true_labels.reset_index(drop=True)

    mapped_labels = cluster_labels.copy()
    unique_clusters = cluster_labels.unique()

    for cluster_id in unique_clusters:
        if cluster_id == -1: # Ignore noise for mapping
            continue

        mask = cluster_labels == cluster_id
        samples_in_cluster = true_labels[mask]

        if len(samples_in_cluster) > 0:
            majority_class = samples_in_cluster.value_counts().idxmax()
            mapped_labels[mask] = majority_class

    # Handle noise (DBSCAN) by assigning it to the global majority class
    if -1 in unique_clusters:
        global_majority = true_labels.value_counts().idxmax()
        mapped_labels[cluster_labels == -1] = global_majority

    return mapped_labels.astype(int)

# ── Loop por algoritmo ───────────────────────────────────────────────────────
for algo_name, cluster_label_array in clustering_results.items():

    # Get the actual labels from the clustering algorithm before binary mapping
    # This ensures we plot the *distinct* clusters found by each algorithm
    original_cluster_labels = clustering_results[algo_name] 

    if algo_name != "DBSCAN":
        # For K-Means, FCM, Spectral, the 'labels' are the distinct clusters
        labels_to_plot = original_cluster_labels
        # Confusion matrix still uses mapped binary labels for comparison with y_original
        mapped_labels = map_cluster_labels(original_cluster_labels, y_cluster) 
    else:
        # DBSCAN might have noise (-1), we use the direct output for plotting.
        # The map_cluster_labels already handles DBSCAN's -1 in a specific way for confusion matrix.
        labels_to_plot = original_cluster_labels
        non_noise_indices = original_cluster_labels != -1
        if np.sum(non_noise_indices) > 0 and len(np.unique(original_cluster_labels[non_noise_indices])) > 1:
            temp_mapped_labels = map_cluster_labels(
                original_cluster_labels[non_noise_indices], y_cluster[non_noise_indices]
            )
            mapped_labels = np.full_like(original_cluster_labels, fill_value=y_cluster.mode()[0])
            mapped_labels[non_noise_indices] = temp_mapped_labels
        else:
            mapped_labels = np.full_like(original_cluster_labels, fill_value=y_cluster.mode()[0])


    # Confusion Matrix
    cm = confusion_matrix(y_cluster, mapped_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Bajo Ajuste (0)', 'Alto Ajuste (1)'],
                yticklabels=['Bajo Ajuste (0)', 'Alto Ajuste (1)'])
    plt.title(f'Matriz de Confusión para {algo_name} (vs. y_original)')
    plt.ylabel('Etiquetas Verdaderas (y_original)')
    plt.xlabel('Etiquetas del Cluster Mapeadas')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{algo_name}.png'))
    plt.close()

    # Distribution Comparison 2D
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y_cluster,
                    palette={0: 'blue', 1: 'red'}, legend='full', alpha=0.6)
    plt.title('Distribución con Etiquetas Originales (y_original)')
    plt.xlabel('PCA Componente 1')
    plt.ylabel('PCA Componente 2')

    plt.subplot(1, 2, 2)
    # Use a generic colormap for `labels_to_plot` to show distinct clusters
    # And ensure the hue is labels_to_plot, not mapped_labels
    unique_plot_labels = np.unique(labels_to_plot)
    n_plot_clusters = len(unique_plot_labels[unique_plot_labels != -1])
    # Create a divergent colormap for visualization, or 'viridis' like clustering_visualization.
    if algo_name == 'dbscan': # Special handling for DBSCAN noise label -1
         cmap_name = 'viridis' # or 'Paired', 'Set1' etc.
         colors = plt.cm.get_cmap(cmap_name, max(n_plot_clusters,1))
         # Create a custom palette: map -1 to black/grey, then other clusters to cmap
         palette_dict = {lbl: colors(i) for i, lbl in enumerate(unique_plot_labels)}
         if -1 in unique_plot_labels:
             palette_dict[-1] = 'grey' # Assign grey to noise points
         sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=labels_to_plot,
                         palette=palette_dict, legend='full', alpha=0.6)
    else:
        sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=labels_to_plot,
                        palette="viridis", legend='full', alpha=0.6) # Using viridis for consistency with the first plot
    plt.title(f'Distribución con Clusters de {algo_name}')
    plt.xlabel('PCA Componente 1')
    plt.ylabel('PCA Componente 2')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'dist_comparison_2d_{algo_name}.png'))
    plt.close()

    # 3D: solo el clusterizado
    # Use labels_to_plot instead of mapped_plot to show distinct clusters
    labels_for_3d_plot = pd.Series(labels_to_plot).reset_index(drop=True)
    
    # Generate discrete colors for each unique cluster label
    unique_cluster_ids = labels_for_3d_plot.unique()
    num_unique_clusters = len(unique_cluster_ids)
    
    # Use a consistent color sequence across plots if possible, or a color scale
    # If DBSCAN has -1, ensure it's handled distinctly (e.g., grey)
    if -1 in unique_cluster_ids and algo_name == 'dbscan':
        color_map = {str(label): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i, label in enumerate(sorted([l for l in unique_cluster_ids if l != -1]))}
        color_map[str(-1)] = 'grey'
    else:
        color_map = {str(label): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i, label in enumerate(sorted(unique_cluster_ids))}

    fig_clust = px.scatter_3d(
        x=X_vis3d[:, 0], y=X_vis3d[:, 1], z=X_vis3d[:, 2],
        color=labels_for_3d_plot.astype(str), # Pass original cluster labels
        title=f'Distribución 3D con Clusters ({algo_name})',
        color_discrete_map=color_map # Use the generated color map
    )
    fig_clust.update_layout(scene=dict(
        xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'
    ))
    fig_clust.write_html(os.path.join(OUTPUT_DIR, f'dist_3d_clustered_{algo_name}.html'))

# 8.2 Model Performance Comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

metrics = ['Accuracy', 'F1-Score', 'AUC-ROC', 'AUC-PR']
axes_list = [ax1, ax2, ax3, ax4]

for metric, ax in zip(metrics, axes_list):
    pivot_data = results_df.pivot(index='Model', columns='Labels', values=metric)
    pivot_data.plot(kind='bar', ax=ax, rot=45)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.legend(title='Label Type')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 8.3 Label Distribution Before/After
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original distribution
y_full_original.value_counts().plot(kind='bar', ax=ax1, color=['lightcoral', 'lightblue'])
ax1.set_title('Original Label Distribution')
ax1.set_xlabel('High Adjustment')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['Low (0)', 'High (1)'], rotation=0)

# Reevaluated distribution
y_full_reevaluated.value_counts().plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
ax2.set_title('Reevaluated Label Distribution')
ax2.set_xlabel('High Adjustment')
ax2.set_ylabel('Count')
ax2.set_xticklabels(['Low (0)', 'High (1)'], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'label_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ===========================================
# 9. SUMMARY AND CONCLUSIONS
# ===========================================

print("\n9. ANALYSIS SUMMARY")
print("="*60)

print(f"Dataset Characteristics:")
print(f"  - Total samples: {len(X_raw)}")
print(f"  - Features: {X_raw.shape[1]}")
print(f"  - Original positive rate: {y_full_original.mean():.1%}")
print(f"  - Reevaluated positive rate: {y_full_reevaluated.mean():.1%}")

print(f"\nClustering Analysis:")
print(f"  - Algorithms used: {', '.join(algorithms)}")
print(f"  - PCA components (95% variance): {X_pca.shape[1]}")
print(f"  - Samples reevaluated: {(y_full_reevaluated != y_full_original).sum()}")

print(f"\nBest Performing Model:")
best_result = results_df.loc[results_df['AUC-PR'].idxmax()]
print(f"  - Model: {best_result['Model']} with {best_result['Labels']} labels")
print(f"  - AUC-PR: {best_result['AUC-PR']:.3f}")
print(f"  - F1-Score: {best_result['F1-Score']:.3f}")

print("\nFiles generated:")
print(f"  - model_comparison_results.csv")
print(f"  - clustering_visualization.png")
print(f"  - model_performance_comparison.png")
print(f"  - label_distribution_comparison.png")
print(f"  - dist_3d_original.html")
for algo_name in algorithms:
    print(f"  - dist_3d_clustered_{algo_name}.html")
    print(f"  - dist_comparison_2d_{algo_name}.png")
    print(f"  - confusion_matrix_{algo_name}.png")


print("\n" + "="*60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("="*60)