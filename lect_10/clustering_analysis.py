"""
Analisis de Clustering: K-Means vs DBSCAN
Dataset: Flexible (especificar ruta abajo)

Este script realiza un analisis comparativo entre dos algoritmos de clustering:
- K-Means (k=2)
- DBSCAN

Genera visualizaciones con PCA y analisis de precision de predicciones.

Características:
- Preprocesamiento genérico que funciona con cualquier dataset
- Detección automática de columnas target y features
- Compatible con datasets de clustering de cualquier tamaño
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (confusion_matrix, classification_report, 
                             silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ============================================================================

# ESPECIFICAR AQUÍ LA RUTA DEL DATASET
# Opciones disponibles:
#   - '../lect_08/datasets/dataset_sintetico_FIRE_UdeA.csv'
#   - '../lect_08/datasets/dataset_sintetico_FIRE_UdeA_realista.csv'
DATASET_PATH = '../lect_08/datasets/dataset_sintetico_FIRE_UdeA.csv'

OUTPUT_DIR = './outputs_pocas_dim'

# Crear directorio de outputs si no existe
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("=" * 80)
print("CLUSTERING ANALYSIS: K-MEANS vs DBSCAN")
print("=" * 80)

# Cargar dataset
print("\n[1] Cargando dataset...")
print(f"Ruta del dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"Shape del dataset: {df.shape}")
print(f"\nÚltimas filas del dataset:")
print(df.tail())

# ============================================================================
# 2. PREPROCESAMIENTO DE DATOS (GENÉRICO)
# ============================================================================

print("\n[2] Preprocesamiento de datos...")

# Detectar la columna target automáticamente
# Busca columnas que comúnmente contienen labels/target
target_columns = ['label', 'target', 'clase', 'class', 'y', 'Label', 'Target']
target_col = None

for col in target_columns:
    if col in df.columns:
        target_col = col
        break

# Si no encuentra una columna target común, usa la última columna numérica
if target_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        # Asumir que la última columna numérica es el target
        target_col = numeric_cols[-1]
        print(f"Columna target detectada automáticamente: '{target_col}'")
else:
    print(f"Columna target encontrada: '{target_col}'")

# Obtener la variable target
if target_col:
    y_true = df[target_col].values
    print(f"\nDistribución de clases en '{target_col}':")
    unique_labels = np.unique(y_true)
    for label in sorted(unique_labels):
        count = (y_true == label).sum()
        percentage = count / len(y_true) * 100
        print(f"Clase {label}: {count} ({percentage:.2f}%)")
else:
    raise ValueError("No se pudo detectar la columna target en el dataset")

# Identificar características de forma genérica
# Excluir columnas no numéricas y la columna target
numeric_df = df.select_dtypes(include=[np.number]).copy()
columns_to_exclude = [target_col]

# Crear conjunto de características
X = numeric_df.drop(columns=columns_to_exclude, errors='ignore').copy()

print(f"\nNumero de caracteristicas: {X.shape[1]}")
print(f"Caracteristicas seleccionadas: {list(X.columns)}")

# Manejar valores faltantes
print(f"\nValores faltantes por columna:")
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(missing_counts[missing_counts > 0])
    # Imputar valores faltantes con la media
    X = X.fillna(X.mean())
    print(f"Valores faltantes después de imputación: {X.isnull().sum().sum()}")
else:
    print("No hay valores faltantes")

# Estandarizar características
print("\nEstandarizando caracteristicas...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# 3. K-MEANS CLUSTERING (K=2)
# ============================================================================

print("\n[3] Ejecutando K-Means (k=2)...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Métricas K-Means
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"K-Means Davies-Bouldin Index: {kmeans_davies_bouldin:.4f}")
print(f"K-Means Calinski-Harabasz Score: {kmeans_calinski:.4f}")

print(f"\nDistribución de clusters K-Means:")
print(f"Cluster 0: {(kmeans_labels == 0).sum()}")
print(f"Cluster 1: {(kmeans_labels == 1).sum()}")

# ============================================================================
# 4. DBSCAN CLUSTERING
# ============================================================================

print("\n[4] Ejecutando DBSCAN...")

# Para DBSCAN, necesitamos encontrar parámetros adecuados
# Usaremos eps=0.5 como valor inicial
from sklearn.neighbors import NearestNeighbors

# Calcular k-distance graph para encontrar eps óptimo
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1], axis=0)

# Usar el codo de la gráfica de distancias
eps_value = np.percentile(distances, 90)  # 90th percentile
min_samples = 5

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Contar clusters y ruido
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Parámetros DBSCAN: eps={eps_value:.4f}, min_samples={min_samples}")
print(f"Número de clusters: {n_clusters_dbscan}")
print(f"Número de puntos de ruido: {n_noise}")

print(f"\nDistribución de clusters DBSCAN:")
unique_labels = set(dbscan_labels)
for label in sorted(unique_labels):
    if label == -1:
        print(f"Ruido: {(dbscan_labels == label).sum()}")
    else:
        print(f"Cluster {label}: {(dbscan_labels == label).sum()}")

# Métricas DBSCAN (solo si hay clusters válidos)
if n_clusters_dbscan > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    dbscan_davies_bouldin = davies_bouldin_score(X_scaled, dbscan_labels)
    dbscan_calinski = calinski_harabasz_score(X_scaled, dbscan_labels)
    
    print(f"\nDBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
    print(f"DBSCAN Davies-Bouldin Index: {dbscan_davies_bouldin:.4f}")
    print(f"DBSCAN Calinski-Harabasz Score: {dbscan_calinski:.4f}")
else:
    print("\nNo se pudo calcular métricas DBSCAN (clusters insuficientes)")

# ============================================================================
# 5. ANÁLISIS COMPARATIVO CON LABELS VERDADEROS
# ============================================================================

print("\n[5] Análisis Comparativo...")

# Para K-Means
print("\n--- K-MEANS vs LABELS REALES ---")
cm_kmeans = confusion_matrix(y_true, kmeans_labels)
print(f"\nMatriz de Confusión K-Means:")
print(cm_kmeans)

# Analizar cada clase
unique_labels = sorted(np.unique(y_true))
analysis_results_kmeans = {}

for y_label in unique_labels:
    label_indices = y_true == y_label
    print(f"\nDesglose de datos etiquetados como clase {y_label} (en label original):")
    print(f"Total de datos con label={y_label}: {label_indices.sum()}")
    
    kmeans_pred_on_label = kmeans_labels[label_indices]
    print(f"\nK-Means predicciones para datos con label={y_label}:")
    
    for cluster in range(2):  # K-means siempre tiene 2 clusters
        count = (kmeans_pred_on_label == cluster).sum()
        percentage = count / label_indices.sum() * 100 if label_indices.sum() > 0 else 0
        print(f"  Predijo Cluster {cluster}: {count} ({percentage:.2f}%)")
        analysis_results_kmeans[f"label{y_label}_cluster{cluster}"] = (count, percentage)

# Para DBSCAN
print("\n--- DBSCAN vs LABELS REALES ---")

# Para comparación justa, mapear ruido (-1) a la clase más cercana
dbscan_labels_processed = dbscan_labels.copy()
if -1 in dbscan_labels_processed:
    # Mapear puntos de ruido al cluster más cercano
    noise_mask = dbscan_labels_processed == -1
    if noise_mask.sum() > 0:
        # Encontrar el cluster más cercano para cada punto de ruido
        X_noise = X_scaled[noise_mask]
        distances_to_centers = np.array([
            np.min([np.linalg.norm(p - center) for center in dbscan.components_])
            if len(dbscan.components_) > 0 else float('inf')
            for p in X_noise
        ])
        # Para simplificar, asignar al cluster 0 o 1 basado en la mayoría
        cluster_counts = {}
        for label in set(dbscan_labels):
            if label != -1:
                cluster_counts[label] = (dbscan_labels == label).sum()
        
        if cluster_counts:
            most_common_cluster = max(cluster_counts, key=cluster_counts.get)
            dbscan_labels_processed[noise_mask] = most_common_cluster

cm_dbscan = confusion_matrix(y_true, dbscan_labels_processed)
print(f"\nMatriz de Confusión DBSCAN:")
print(cm_dbscan)

# Analizar cada clase para DBSCAN
analysis_results_dbscan = {}
for y_label in unique_labels:
    label_indices = y_true == y_label
    print(f"\nDesglose de datos etiquetados como clase {y_label} (en label original):")
    print(f"Total de datos con label={y_label}: {label_indices.sum()}")
    
    dbscan_pred_on_label = dbscan_labels_processed[label_indices]
    print(f"\nDBSCAN predicciones para datos con label={y_label}:")
    
    unique_clusters = sorted(np.unique(dbscan_pred_on_label))
    for cluster in unique_clusters:
        count = (dbscan_pred_on_label == cluster).sum()
        percentage = count / label_indices.sum() * 100 if label_indices.sum() > 0 else 0
        print(f"  Predijo Cluster {cluster}: {count} ({percentage:.2f}%)")
        analysis_results_dbscan[f"label{y_label}_cluster{cluster}"] = (count, percentage)

# ============================================================================
# 6. PCA PARA VISUALIZACIÓN
# ============================================================================

print("\n[6] Aplicando PCA para visualización...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Varianza explicada por PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Varianza explicada por PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Varianza total explicada: {pca.explained_variance_ratio_.sum():.4f}")

# ============================================================================
# 7. GENERACIÓN DE GRÁFICAS
# ============================================================================

print("\n[7] Generando gráficas...")

# Configurar estilo
sns.set_style("whitegrid")
colors_kmeans = ['#FF6B6B', '#4ECDC4']
colors_dbscan = ['#95E1D3', '#F38181']
colors_labels = ['#FFD93D', '#6BCB77']

# ------
# Gráfica 1: PCA - K-Means
# ------
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                     cmap='Set2', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], 
           pca.transform(kmeans.cluster_centers_)[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroides')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12, fontweight='bold')
ax.set_title('Clustering K-Means (k=2) - Proyección PCA', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_kmeans_pca.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 01_kmeans_pca.png")

# ------
# Gráfica 2: PCA - DBSCAN
# ------
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, 
                     cmap='Set1', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
# Marcar puntos de ruido
noise_mask = dbscan_labels == -1
if noise_mask.sum() > 0:
    ax.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], 
              marker='x', s=200, c='red', linewidths=2, label='Ruido')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12, fontweight='bold')
ax.set_title('Clustering DBSCAN - Proyección PCA', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_dbscan_pca.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 02_dbscan_pca.png")

# ------
# Gráfica 3: PCA - Labels Reales
# ------
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, 
                     cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12, fontweight='bold')
ax.set_title('Labels Reales - Proyección PCA', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, label='Clase')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Clase 0', 'Clase 1'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_labels_pca.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 03_labels_pca.png")

# ------
# Gráfica 4: Comparación K-Means vs Labels
# ------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                          cmap='Set2', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
axes[0].set_title('K-Means Clustering', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, 
                          cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
axes[1].set_title('Labels Reales', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter2, ax=axes[1], label='Clase')
cbar.set_ticks([0, 1])

fig.suptitle('Comparación: K-Means vs Labels Reales', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_comparison_kmeans_vs_labels.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 04_comparison_kmeans_vs_labels.png")

# ------
# Gráfica 5: Comparación DBSCAN vs Labels
# ------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, 
                          cmap='Set1', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
noise_mask = dbscan_labels == -1
if noise_mask.sum() > 0:
    axes[0].scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], 
                   marker='x', s=200, c='red', linewidths=2, label='Ruido')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
axes[0].set_title('DBSCAN Clustering', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')
axes[0].legend()

scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, 
                          cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
axes[1].set_title('Labels Reales', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter2, ax=axes[1], label='Clase')
cbar.set_ticks([0, 1])

fig.suptitle('Comparación: DBSCAN vs Labels Reales', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_comparison_dbscan_vs_labels.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 05_comparison_dbscan_vs_labels.png")

# ------
# Gráfica 6: Matrices de Confusión
# ------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Crear etiquetas dinámicas para las matrices de confusión
y_labels = [f'Label {i}' for i in sorted(np.unique(y_true))]
cluster_labels = ['Cluster 0', 'Cluster 1']

# K-Means
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=cluster_labels,
            yticklabels=y_labels)
axes[0].set_title('Matriz de Confusión - K-Means', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Label Real', fontweight='bold')
axes[0].set_xlabel('Predicción K-Means', fontweight='bold')

# DBSCAN
sns.heatmap(cm_dbscan, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=cluster_labels,
            yticklabels=y_labels)
axes[1].set_title('Matriz de Confusión - DBSCAN', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Label Real', fontweight='bold')
axes[1].set_xlabel('Predicción DBSCAN', fontweight='bold')

fig.suptitle('Matrices de Confusión: Comparación de Algoritmos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 06_confusion_matrices.png")

# ------
# Gráfica 7: Distribución de Clusters vs Labels
# ------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Labels reales
label_counts = np.bincount(y_true)
label_names = [f'Clase {i}' for i in sorted(np.unique(y_true))]
colors_labels = plt.cm.Set2(np.linspace(0, 1, len(label_names)))
axes[0].bar(label_names, label_counts, color=colors_labels, edgecolor='black', linewidth=2)
axes[0].set_title('Distribución de Labels Reales', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cantidad', fontweight='bold')
for i, v in enumerate(label_counts):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=11)

# K-Means
kmeans_counts = np.bincount(kmeans_labels, minlength=2)
kmeans_names = [f'Cluster {i}' for i in range(len(kmeans_counts))]
colors_kmeans = plt.cm.Set1(np.linspace(0, 1, len(kmeans_names)))
axes[1].bar(kmeans_names, kmeans_counts, color=colors_kmeans, edgecolor='black', linewidth=2)
axes[1].set_title('Distribución de Clusters K-Means', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cantidad', fontweight='bold')
for i, v in enumerate(kmeans_counts):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=11)

# DBSCAN
dbscan_counts = {}
for label in sorted(set(dbscan_labels)):
    if label == -1:
        dbscan_counts['Ruido'] = (dbscan_labels == label).sum()
    else:
        dbscan_counts[f'Cluster {label}'] = (dbscan_labels == label).sum()

labels_dbscan = list(dbscan_counts.keys())
counts_dbscan = list(dbscan_counts.values())
colors_dbscan = ['#F38181' if 'Ruido' in l else '#95E1D3' for l in labels_dbscan]
axes[2].bar(labels_dbscan, counts_dbscan, color=colors_dbscan, edgecolor='black', linewidth=2)
axes[2].set_title('Distribución de Clusters DBSCAN', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Cantidad', fontweight='bold')
for i, v in enumerate(counts_dbscan):
    axes[2].text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=11)
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

fig.suptitle('Distribución de Clusters vs Labels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_cluster_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 07_cluster_distribution.png")

# ------
# Gráfica 8: Análisis de Predicciones por Clase
# ------

# Analizar la clase con más datos
unique_classes = sorted(np.unique(y_true))
class_to_analyze = unique_classes[np.argmax([sum(y_true == c) for c in unique_classes])]
label_indices_analysis = y_true == class_to_analyze

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means - Predicciones para la clase seleccionada
kmeans_pred_on_class = kmeans_labels[label_indices_analysis]
unique_clusters = sorted(np.unique(kmeans_pred_on_class))
categories_kmeans = [f'Predijo\nCluster {c}' for c in unique_clusters]
values_kmeans = [(kmeans_pred_on_class == c).sum() for c in unique_clusters]
percentages_kmeans = [v / label_indices_analysis.sum() * 100 if label_indices_analysis.sum() > 0 else 0 
                      for v in values_kmeans]
colors_kmeans_plot = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
bars1 = axes[0].bar(categories_kmeans, values_kmeans, color=colors_kmeans_plot, edgecolor='black', linewidth=2)
axes[0].set_title(f'K-Means: Predicción para Datos con Label={class_to_analyze}', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cantidad de Datos', fontweight='bold')
for i, (v, p) in enumerate(zip(values_kmeans, percentages_kmeans)):
    axes[0].text(i, v + 5, f'{v}\n({p:.1f}%)', ha='center', fontweight='bold', fontsize=10)

# DBSCAN - Predicciones para la clase seleccionada
dbscan_pred_on_class = dbscan_labels_processed[label_indices_analysis]
unique_clusters_dbscan = sorted(np.unique(dbscan_pred_on_class))
categories_dbscan = [f'Predijo\nCluster {c}' for c in unique_clusters_dbscan]
values_dbscan = [(dbscan_pred_on_class == c).sum() for c in unique_clusters_dbscan]
percentages_dbscan = [v / label_indices_analysis.sum() * 100 if label_indices_analysis.sum() > 0 else 0 
                      for v in values_dbscan]
colors_dbscan_plot = plt.cm.Pastel1(np.linspace(0, 1, len(unique_clusters_dbscan)))
bars2 = axes[1].bar(categories_dbscan, values_dbscan, color=colors_dbscan_plot, edgecolor='black', linewidth=2)
axes[1].set_title(f'DBSCAN: Predicción para Datos con Label={class_to_analyze}', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cantidad de Datos', fontweight='bold')
for i, (v, p) in enumerate(zip(values_dbscan, percentages_dbscan)):
    axes[1].text(i, v + 5, f'{v}\n({p:.1f}%)', ha='center', fontweight='bold', fontsize=10)

fig.suptitle(f'Análisis de Predicciones: Clase {class_to_analyze} (Total: {label_indices_analysis.sum()})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_class_prediction_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 08_class_prediction_analysis.png")

# ------
# Gráfica 9: Métricas de Clustering
# ------
fig, ax = plt.subplots(figsize=(12, 6))

metrics_names = ['Silhouette Score', 'Davies-Bouldin Index\n(menor es mejor)', 'Calinski-Harabasz\nScore']
kmeans_metrics = [kmeans_silhouette, kmeans_davies_bouldin, kmeans_calinski]
dbscan_metrics = [dbscan_silhouette if n_clusters_dbscan > 1 else 0, 
                  dbscan_davies_bouldin if n_clusters_dbscan > 1 else 0,
                  dbscan_calinski if n_clusters_dbscan > 1 else 0]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, kmeans_metrics, width, label='K-Means', 
               color='#FF6B6B', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, dbscan_metrics, width, label='DBSCAN', 
               color='#95E1D3', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Métrica', fontweight='bold', fontsize=11)
ax.set_ylabel('Valor', fontweight='bold', fontsize=11)
ax.set_title('Comparación de Métricas de Clustering', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height != 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_clustering_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 09_clustering_metrics.png")

# ============================================================================
# 8. REPORTE FINAL EN TEXTO
# ============================================================================

print("\n[8] Generando reporte summary...")

# Generar contenido dinámico para las clases
labels_distribution = "\n".join([
    f"  - Clase {label}: {(y_true == label).sum()} ({(y_true == label).sum() / len(y_true) * 100:.2f}%)"
    for label in sorted(np.unique(y_true))
])

kmeans_distribution = "\n".join([
    f"  - Cluster {i}: {(kmeans_labels == i).sum()}"
    for i in range(2)
])

dbscan_distribution_str = "\n".join([
    f"  - {'Ruido' if label == -1 else f'Cluster {label}'}: {(dbscan_labels == label).sum()}"
    for label in sorted(set(dbscan_labels))
])

# Crear análisis dinámico para K-Means
kmeans_analysis = ""
for y_label in sorted(np.unique(y_true)):
    label_indices = y_true == y_label
    kmeans_analysis += f"\nAnálisis para Datos con Label={y_label}:\n"
    kmeans_analysis += f"  Total de datos con label={y_label}: {label_indices.sum()}\n"
    kmeans_pred_on_label = kmeans_labels[label_indices]
    for cluster in range(2):
        count = (kmeans_pred_on_label == cluster).sum()
        percentage = count / label_indices.sum() * 100 if label_indices.sum() > 0 else 0
        kmeans_analysis += f"  K-Means predijo Cluster {cluster}: {count} ({percentage:.2f}%)\n"

# Crear análisis dinámico para DBSCAN
dbscan_analysis = ""
for y_label in sorted(np.unique(y_true)):
    label_indices = y_true == y_label
    dbscan_analysis += f"\nAnálisis para Datos con Label={y_label}:\n"
    dbscan_analysis += f"  Total de datos con label={y_label}: {label_indices.sum()}\n"
    dbscan_pred_on_label = dbscan_labels_processed[label_indices]
    unique_clusters = sorted(np.unique(dbscan_pred_on_label))
    for cluster in unique_clusters:
        count = (dbscan_pred_on_label == cluster).sum()
        percentage = count / label_indices.sum() * 100 if label_indices.sum() > 0 else 0
        dbscan_analysis += f"  DBSCAN predijo Cluster {cluster}: {count} ({percentage:.2f}%)\n"

report_content = f"""
{"=" * 80}
REPORTE DE ANÁLISIS: K-MEANS vs DBSCAN
Dataset: {Path(DATASET_PATH).name}
{"=" * 80}

1. INFORMACIÓN DEL DATASET
{"-" * 80}
Shape: {X.shape}
Características: {X.shape[1]}
Numero de muestras: {X.shape[0]}

Distribución de Labels:
{labels_distribution}

2. K-MEANS CLUSTERING (k=2)
{"-" * 80}
Distribución de Clusters:
{kmeans_distribution}

Métricas de Calidad:
  - Silhouette Score: {kmeans_silhouette:.4f}
  - Davies-Bouldin Index: {kmeans_davies_bouldin:.4f}
  - Calinski-Harabasz Score: {kmeans_calinski:.4f}

Matriz de Confusión K-Means:
{cm_kmeans}

{kmeans_analysis}

3. DBSCAN CLUSTERING
{"-" * 80}
Parámetros:
  - eps: {eps_value:.4f}
  - min_samples: {min_samples}

Distribución de Clusters:
  - Número de clusters: {n_clusters_dbscan}
  - Número de puntos de ruido: {n_noise}
"""

if n_clusters_dbscan > 0:
    report_content += f"""
Métricas de Calidad:
  - Silhouette Score: {dbscan_silhouette if n_clusters_dbscan > 1 else 'N/A'}
  - Davies-Bouldin Index: {dbscan_davies_bouldin if n_clusters_dbscan > 1 else 'N/A'}
  - Calinski-Harabasz Score: {dbscan_calinski if n_clusters_dbscan > 1 else 'N/A'}

Matriz de Confusión DBSCAN:
{cm_dbscan}

{dbscan_analysis}
"""

report_content += f"""
4. PCA ANÁLISIS
{"-" * 80}
Varianza explicada:
  - PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)
  - PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)
  - Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)

5. CONCLUSIONES
{"-" * 80}
- El análisis se realizó sobre {X.shape[0]} muestras con {X.shape[1]} características.
- Se identificaron {len(np.unique(y_true))} clases en los datos: {', '.join(map(str, np.unique(y_true)))}
- K-Means agrupa los datos en 2 clusters con Silhouette Score de {kmeans_silhouette:.4f}
- DBSCAN encontró {n_clusters_dbscan} cluster(s) con {n_noise} punto(s) de ruido y Silhouette Score de {dbscan_silhouette if n_clusters_dbscan > 1 else 'N/A'}
- Las gráficas PCA muestran la proyección bidimensional de los datos en el espacio de
  componentes principales más importantes.
- La varianza total explicada por PC1 y PC2 es del {pca.explained_variance_ratio_.sum()*100:.2f}%, indicando que
  la mayoría de la variación en los datos se captura en dos dimensiones.

Archivos generados:
  - 01_kmeans_pca.png: K-Means clustering en espacio PCA
  - 02_dbscan_pca.png: DBSCAN clustering en espacio PCA
  - 03_labels_pca.png: Labels verdaderos en espacio PCA
  - 04_comparison_kmeans_vs_labels.png: Comparación K-Means vs Labels
  - 05_comparison_dbscan_vs_labels.png: Comparación DBSCAN vs Labels
  - 06_confusion_matrices.png: Matrices de confusión
  - 07_cluster_distribution.png: Distribución de clusters
  - 08_class_prediction_analysis.png: Análisis de predicciones por clase
  - 09_clustering_metrics.png: Métricas de clustering
  - clustering_report.txt: Reporte completo (este archivo)

{"=" * 80}
Análisis completado exitosamente.
{"=" * 80}
"""

# Guardar reporte
report_path = os.path.join(OUTPUT_DIR, 'clustering_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✓ Guardado: clustering_report.txt")

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"\nTodas las gráficas y reportes se encuentran en: {os.path.abspath(OUTPUT_DIR)}")
print(f"\nArchivos generados:")
print("  - 01_kmeans_pca.png")
print("  - 02_dbscan_pca.png")
print("  - 03_labels_pca.png")
print("  - 04_comparison_kmeans_vs_labels.png")
print("  - 05_comparison_dbscan_vs_labels.png")
print("  - 06_confusion_matrices.png")
print("  - 07_cluster_distribution.png")
print("  - 08_class_prediction_analysis.png")
print("  - 09_clustering_metrics.png")
print("  - clustering_report.txt")
print("\n" + "=" * 80)
