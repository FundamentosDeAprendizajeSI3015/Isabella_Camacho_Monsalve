"""
Análisis de Agrupamiento (Clustering) - Dataset FIRE UdeA Realista
SI3015 - Fundamentos de Aprendizaje Automático

Este script aplica K-means y DBSCAN al dataset sintético FIRE UdeA Realista
y almacena las gráficas en la carpeta outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Configuración
random_state = 42
np.random.seed(random_state)

# Configurar matplotlib
plt.rc('font', family='serif', size=12)

# Crear carpeta outputs si no existe
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 80)
print("Análisis de Clustering - Dataset FIRE UdeA Realista")
print("=" * 80)

# ============================================================================
# 1. Cargar y preparar datos
# ============================================================================
print("\n[1] Cargando datos...")
df = pd.read_csv('../lect_08/datasets/dataset_sintetico_FIRE_UdeA_realista.csv')

print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")

# Seleccionar solo características numéricas (excluir 'anio', 'unidad' y 'label')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_cols:
    numeric_cols.remove('label')

print(f"\nCaracterísticas numéricas seleccionadas: {len(numeric_cols)}")

# Preparar datos: eliminar valores faltantes
data = df[numeric_cols].fillna(df[numeric_cols].mean())
print(f"Datos después de llenar valores faltantes: {data.shape}")

# ============================================================================
# 2. Definir pipeline de preprocessamiento
# ============================================================================
print("\n[2] Configurando pipeline de preprocessamiento...")
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, np.arange(data.shape[1])),
    ],
)

# ============================================================================
# 3. K-Means con K=2
# ============================================================================
print("\n[3] Aplicando K-Means con K=2...")
clu_kmeans_2 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clustering", KMeans(n_clusters=2, random_state=random_state))
])
clu_kmeans_2.fit(data)
inertia_2 = clu_kmeans_2['clustering'].inertia_
print(f"    Inercia con K=2: {inertia_2:.2f}")

# Visualización con PCA
print("    Creando visualización con PCA...")
pca = PCA(n_components=2, random_state=random_state)
data_pca = pca.fit_transform(data)
labels_2 = clu_kmeans_2['clustering'].labels_

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_2, cmap='viridis', s=50, alpha=0.7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('K-Means Clustering (K=2) - FIRE UdeA Realista')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_kmeans_k2.png'), dpi=300, bbox_inches='tight')
print(f"    Gráfica guardada: {output_dir}/01_kmeans_k2.png")
plt.close()

# ============================================================================
# 4. Método del Codo
# ============================================================================
print("\n[4] Aplicando Método del Codo para encontrar K óptimo...")
inertias = []
k_range = list(range(2, 11))  # Empezar en 2 (K=1 no permite Silhouette Score)
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in k_range:
    clu_kmeans = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=k, random_state=random_state, n_init=10))
    ])
    clu_kmeans.fit(data)
    inertias.append(clu_kmeans['clustering'].inertia_)
    
    # Calcular Silhouette Score
    labels = clu_kmeans['clustering'].labels_
    silhouette_avg = silhouette_score(data, labels)
    silhouette_scores.append(silhouette_avg)
    
    print(f"    K={k:2d}: Inercia={inertias[-1]:.2f}, Silhouette={silhouette_avg:.3f}")

# Gráfica del Codo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Número de Clusters (K)')
ax1.set_ylabel('Inercia')
ax1.set_title('Método del Codo - Inercia')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_range)

ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Número de Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Método del Codo - Silhouette Score')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(k_range)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_metodo_codo.png'), dpi=300, bbox_inches='tight')
print(f"\nGráfica guardada: {output_dir}/02_metodo_codo.png")
plt.close()

# Determinar K óptimo basado en Silhouette Score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nK óptimo según Silhouette Score: {optimal_k}")

# ============================================================================
# 5. K-Means con K óptimo
# ============================================================================
print(f"\n[5] Aplicando K-Means con K={optimal_k}...")
clu_kmeans_optimal = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clustering", KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10))
])
clu_kmeans_optimal.fit(data)
inertia_optimal = clu_kmeans_optimal['clustering'].inertia_
labels_optimal = clu_kmeans_optimal['clustering'].labels_

print(f"    Inercia con K={optimal_k}: {inertia_optimal:.2f}")
print(f"    Distribución de clusters:")
unique, counts = np.unique(labels_optimal, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"      Cluster {cluster_id}: {count} muestras ({count/len(labels_optimal)*100:.1f}%)")

# Visualización con PCA
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_optimal, cmap='viridis', s=50, alpha=0.7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title(f'K-Means Clustering (K={optimal_k}) - FIRE UdeA Realista')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'03_kmeans_k{optimal_k}.png'), dpi=300, bbox_inches='tight')
print(f"    Gráfica guardada: {output_dir}/03_kmeans_k{optimal_k}.png")
plt.close()

# ============================================================================
# 6. DBSCAN
# ============================================================================
print(f"\n[6] Aplicando DBSCAN con eps=0.5, min_samples=5...")
clu_dbscan = Pipeline(steps=[
    ("clustering", DBSCAN(eps=0.5, min_samples=5))
])

# Primero escalar los datos para DBSCAN
data_scaled = preprocessor.fit_transform(data)
clu_dbscan.fit(data_scaled)
labels_dbscan = clu_dbscan['clustering'].labels_

# Contar clusters (notar que -1 representa ruido)
unique_dbscan, counts_dbscan = np.unique(labels_dbscan, return_counts=True)
print(f"    Número de clusters encontrados: {len(unique_dbscan) - (1 if -1 in unique_dbscan else 0)}")
print(f"    Distribución de clusters:")
for cluster_id, count in zip(unique_dbscan, counts_dbscan):
    if cluster_id == -1:
        print(f"      Ruido: {count} muestras ({count/len(labels_dbscan)*100:.1f}%)")
    else:
        print(f"      Cluster {cluster_id}: {count} muestras ({count/len(labels_dbscan)*100:.1f}%)")

# Convertir PCA completo de datos escalados
pca_dbscan = PCA(n_components=2, random_state=random_state)
data_pca_dbscan = pca_dbscan.fit_transform(data_scaled)

# Visualización DBSCAN
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(data_pca_dbscan[:, 0], data_pca_dbscan[:, 1], 
                     c=labels_dbscan, cmap='viridis', s=50, alpha=0.7)
ax.set_xlabel(f'PC1 ({pca_dbscan.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_dbscan.explained_variance_ratio_[1]:.1%})')
ax.set_title('DBSCAN Clustering - FIRE UdeA Realista')
plt.colorbar(scatter, ax=ax, label='Cluster (-1 = Ruido)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_dbscan.png'), dpi=300, bbox_inches='tight')
print(f"    Gráfica guardada: {output_dir}/04_dbscan.png")
plt.close()

# ============================================================================
# 7. Resumen
# ============================================================================
print("\n" + "=" * 80)
print("Resumen de Resultados")
print("=" * 80)
print(f"Total de muestras: {len(data)}")
print(f"Características numéricas: {len(numeric_cols)}")
print(f"\nK-Means (K=2):")
print(f"  Inercia: {inertia_2:.2f}")
print(f"\nK-Means (K={optimal_k}):")
print(f"  Inercia: {inertia_optimal:.2f}")
print(f"  Silhouette Score: {silhouette_scores[optimal_k-1]:.3f}")
print(f"\nDBSCAN (eps=0.5, min_samples=5):")
print(f"  Clusters encontrados: {len(unique_dbscan) - (1 if -1 in unique_dbscan else 0)}")
print(f"\nGráficas guardadas en: {os.path.abspath(output_dir)}")
print("=" * 80)
