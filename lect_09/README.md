# Análisis de Clustering - Dataset FIRE UdeA Realista

## Descripción

Este script realiza un análisis completo de **agrupamiento (clustering)** sobre el dataset sintético FIRE UdeA Realista, basándose en el notebook `ejAgrupamiento_kmeans_dbscan.ipynb`.

## Contenido

### Script Principal
- `fire_clustering_analysis.py`: Script Python que ejecuta el análisis completo

### Carpeta de Salida
- `outputs/`: Contiene las gráficas generadas:
  - `01_kmeans_k2.png` - K-Means con K=2
  - `02_metodo_codo.png` - Método del Codo e Silhouette Score
  - `03_kmeans_kN.png` - K-Means con K óptimo
  - `04_dbscan.png` - Clustering con DBSCAN

## Procedimiento

El análisis sigue estos pasos:

1. **Carga de datos**: Lee el dataset `dataset_sintetico_FIRE_UdeA_realista.csv`
2. **Preprocessamiento**: Normaliza los datos usando StandardScaler
3. **K-Means (K=2)**: Aplica clustering con K=2 y visualiza
4. **Método del Codo**: Prueba diferentes valores de K (1-10) y calcula:
   - Inercia
   - Silhouette Score
5. **K-Means Óptimo**: Aplica K-Means con el K óptimo determinado
6. **DBSCAN**: Aplica clustering con densidad (eps=0.5, min_samples=5)
7. **Visualización**: Todas las gráficas usan PCA para proyectar a 2D

## Requisitos

```
numpy
pandas
scikit-learn
matplotlib
```

## Uso

```bash
cd lect_09
python fire_clustering_analysis.py
```

## Resultados

El script genera:
- **4 gráficas en alta resolución (300 dpi)** guardadas en `outputs/`
- **Métricas de evaluación**:
  - Inercia para cada K
  - Silhouette Score para cada K
  - Distribución de clusters
  - Número de puntos de ruido (DBSCAN)

## Notas Técnicas

- **PCA**: Se utiliza para visualizar en 2D datos de alta dimensionalidad
- **Normalización**: Todos los algoritmos requieren datos escalados
- **Random State**: Configurado en 42 para reproducibilidad
- **Silhouette Score**: Métrica utilizada para determinar K óptimo
  - Valores cercanos a 1 indican buen clustering
  - Valores cercanos a -1 indican clusters superpuestos

## Referencias

- Basado en: `ejAgrupamiento_kmeans_dbscan.ipynb`
- Curso: SI3015 - Fundamentos de Aprendizaje Automático
