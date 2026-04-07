# Fundamentos de Aprendizaje Automático - Ejercicios

Este repositorio contiene los ejercicios prácticos del curso **Fundamentos de Aprendizaje Automático**, desarrollados por Isabella Camacho Monsalve. Los ejercicios cubren temas fundamentales del análisis de datos, preprocesamiento, visualización y modelado predictivo.

## Objetivo del Repositorio

El propósito de este repositorio es demostrar la aplicación práctica de conceptos clave en aprendizaje automático, incluyendo:
- **Análisis Exploratorio de Datos (EDA)**: Comprensión y visualización de datos
- **Preprocesamiento de Datos**: Limpieza, transformación y preparación de datos
- **Modelado Predictivo**: Implementación de modelos de clasificación y regresión
- **Visualización de Resultados**: Representación gráfica de análisis e insights

---

## Contenido del Repositorio

### 📊 [Informe1](./Informe1/) - Pipeline de EDA y Preprocesamiento

**Objetivo**: Realizar un análisis exploratorio completo y un pipeline de preprocesamiento de datos sobre un proyecto de ajuste estudiantil-carrera.

**Contenido**:
- `eda_pipeline.py`: Script principal que ejecuta análisis exploratorio de datos y preprocesamiento
- `survey.csv`: Dataset de encuesta sobre ajuste estudiantil
- `README.md`: Documentación detallada del informe
- `outputs/`: Directorio con gráficos y resultados generados

**Conceptos cubiertos**:
- Análisis descriptivo de datos
- Visualización con matplotlib y seaborn
- Reducción de dimensionalidad (PCA, t-SNE, UMAP)
- División train-test
- Escalado de características (StandardScaler)
- Manejo de datos desbalanceados

---

### 📚 [Informe2](./Informe2/) - Reevaluación de Etiquetas mediante Clustering Ensemblado

**Objetivo**: Utilizar técnicas de aprendizaje no supervisado (clustering) para reevaluar las etiquetas originales de ajuste estudiante-carrera y mejorar la precisión de modelos supervisados.

**Contenido**:
- `clustering_analysis.py`: Script completo de análisis de clustering y reevaluación de etiquetas
- `data/`: Directorio con datasets actualizados.
- `outputs/`: Directorio con gráficos, tablas comparativas y reportes
- `README.md`: Documentación detallada de metodología y resultados

Nota: Los datos utilizados para el entrenamiento de los modelos se encuentran en la carpeta Informe1/extended_ds, y corresponden a los archivos X_raw.parquet y y.parquet. Estos son el resultado del pipeline de EDA previamente realizado.

**Conceptos cubiertos**:
- Preprocesamiento y eliminación de duplicados
- Reducción de dimensionalidad con PCA (95% varianza)
- Múltiples algoritmos de clustering: K-Means, Fuzzy C-Means, Subtractive Clustering, DBSCAN
- Ensemble ponderado con cálculo de "Lift" de cada algoritmo
- Reetiquetado conservador basado en voto mayoritario del ensemble
- SMOTE para balance de clases en entrenamiento
- Comparación de Random Forest y Logistic Regression
- Métrica principal: AUPRC (Area bajo la Curva Precisión-Recall)
- Evaluación completa con Precisión, Recall, F1-Score y Accuracy

---

### 📈 [lect_02](./lect_02/) - Predicción del Rendimiento Estudiantil

**Objetivo**: Construir un modelo de regresión logística para predecir el rendimiento académico (G3 - calificación final) de estudiantes.

**Contenido**:
- `camacho_isabella_student_performance.py`: Modelo predictivo usando Regresión Logística
- `outputs/`: Directorio con resultados, gráficos y métricas de evaluación

**Conceptos cubiertos**:
- Preprocesamiento con RobustScaler
- Division train-test
- Regresión Logística
- Validación cruzada (Cross-validation)
- Búsqueda de hiperparámetros (GridSearchCV)
- Métricas de clasificación (accuracy, precision, recall, F1-score, ROC-AUC)
- Curvas ROC y Precision-Recall

---

### 💰 [lect_03](./lect_03/) - Laboratorio Fintech - EDA y Preprocesamiento

**Objetivo**: Realizar análisis exploratorio y preprocesamiento de datos sobre empresas fintech de forma sintética para 2025.

**Contenido**:
- `lab_fintech_sintetico_2025.py`: Script de preprocesamiento y análisis exploratorio
- `fintech_top_sintetico_2025.csv`: Dataset sintético de fintech
- `fintech_top_sintetico_dictionary.json`: Diccionario de variables explicando cada columna
- `data_output_finanzas_sintetico/`: Directorio con salidas procesadas

**Conceptos cubiertos**:
- Preprocesamiento de datos financieros
- Análisis de correlaciones (heatmap)
- Series de tiempo (análisis de ingresos)
- Reducción de dimensionalidad (UMAP)
- División de datos train-test
- Exportación de datos procesados (Parquet)
- Documentación de esquema de datos

---

### 🚢 [lect_04](./lect_04/) - Análisis Exploratorio del Dataset Titanic

**Objetivo**: Realizar un análisis exploratorio completo del dataset histórico del Titanic, con énfasis en visualización de datos y estadísticas descriptivas.

**Contenido**:
- `titanic_eda.py`: Script de análisis exploratorio con visualizaciones
- `Titanic-Dataset.csv`: Dataset completo con información de pasajeros del Titanic
- `outputs/`: Directorio con gráficos exportados en PNG/JPG

**Conceptos cubiertos**:
- Limpieza de datos (imputación de valores faltantes)
- Estadísticas descriptivas y medidas de dispersión
- Análisis univariado y bivariado
- Visualización exploratoria (histogramas, boxplots, scatter plots, etc.)
- Análisis de supervivencia según variables demográficas

---

### 🎯 [lect_05](./lect_05/) - Modelo de Clasificación: Predicción de Supervivencia en el Titanic

**Objetivo**: Construir y entrenar un modelo de regresión logística para predecir si un pasajero del Titanic sobrevivió o no, aplicando técnicas de optimización de hiperparámetros y evaluación exhaustiva del modelo.

**Contenido**:
- `titanic_Model.py`: Modelo completo de clasificación con búsqueda de hiperparámetros
- `ej_regresiónLogística.ipynb`: Notebook educativo del ejemplo base en Jupyter
- Visualizaciones generadas:
  - `confusion_matrix_titanic.png`: Matriz de confusión del modelo
  - `feature_importance_titanic.png`: Gráfico de importancia de características

**Conceptos cubiertos**:
- Pipeline de scikit-learn (StandardScaler + LogisticRegression)
- Búsqueda de hiperparámetros con RandomizedSearchCV
- Validación cruzada (5-fold CV)
- Codificación de variables categóricas (one-hot encoding)
- Métricas de evaluación completas (Accuracy, Precision, Recall, F1-score)
- Matriz de confusión y análisis de predicciones
- Interpretación de coeficientes y importancia de características

---

### 🌳 [lect_06](./lect_06/) - Modelos de Clasificación con Árboles: Ajuste Académico Estudiantil

**Objetivo**: Construir y comparar modelos de clasificación basados en árboles de decisión (Random Forest y Gradient Boosting) para predecir el nivel de ajuste académico (ALTO o BAJO) de estudiantes de Ingeniería de Sistemas.

**Contenido**:
- `student_adjustment_classification.py`: Script completo de entrenamiento y evaluación de modelos
- `outputs/`: Directorio con resultados, visualizaciones y reportes generados

**Conceptos cubiertos**:
- Modelos de árboles: Random Forest y Gradient Boosting
- Búsqueda de hiperparámetros con GridSearchCV
- Validación cruzada (3-fold CV)
- Métricas de clasificación multiconjunto (train, validation, test)
- Visualización de árboles de decisión con `plot_tree`
- Análisis de importancia de características
- Matrices de confusión y curvas ROC
- Comparación de modelos y selección del mejor modelo
- Exportación de visualizaciones y reportes

---

### 🏦 [lect_08](./lect_08/) - EDA y Modelado del Dataset FIRE UdeA

**Objetivo**: Realizar un análisis exploratorio completo y desarrollar un modelo de Random Forest para clasificación del dataset sintético FIRE UdeA Realista, abordando desafíos como desbalance de clases y data leakage.

**Contenido**:
- `eda_pipeline_fixed.py`: Pipeline completo de análisis exploratorio, preprocesamiento y visualización
- `random_forest_fire_model.py`: Modelo de Random Forest con correcciones para desbalance de clases
- `datasets/`: Datasets sintéticos FIRE UdeA (original y realista)
- `outputs/`: Visualizaciones EDA y análisis exploratorio
- `outputs_modelo/`: Métricas y reportes del modelo

**Conceptos cubiertos**:
- Pipeline EDA completo con múltiples visualizaciones
- Manejo de datos faltantes (imputación)
- Desbalance de clases y técnicas de remuestreo
- Reducción de dimensionalidad (PCA, t-SNE, UMAP)
- División temporal de datos (time-series aware split)
- Validación cruzada estratificada (StratifiedKFold)
- Random Forest Classifier con ajuste de hiperparámetros
- Métricas apropiadas para clases desbalanceadas (ROC-AUC, Precision-Recall, F1-score)
- Curvas de desempeño (ROC, Precision-Recall)
- Prevención de data leakage en pipelines de ML

---

### 🔀 [lect_09](./lect_09/) - Análisis de Clustering - FIRE UdeA Realista

**Objetivo**: Realizar un análisis completo de agrupamiento (clustering) utilizando K-Means y DBSCAN sobre el dataset sintético FIRE UdeA Realista, determinando el número óptimo de clusters.

**Contenido**:
- `fire_clustering_analysis.py`: Script completo de análisis de clustering
- `outputs/`: Gráficas de clustering y análisis del método del codo
- `README.md`: Documentación detallada del análisis

**Conceptos cubiertos**:
- Preprocesamiento con StandardScaler
- K-Means Clustering con búsqueda de K óptimo
- Método del Codo (Elbow Method) para determinación de clusters
- Silhouette Score para evaluación de calidad de clustering
- DBSCAN para clustering basado en densidad
- PCA para visualización en 2D de datos de alta dimensionalidad
- Análisis de distribución de clusters
- Identificación de puntos de ruido (DBSCAN)
- Generación de visualizaciones en alta resolución

---

### 🔍 [lect_10](./lect_10/) - Análisis Comparativo de Clustering: K-Means vs DBSCAN

**Objetivo**: Desarrollar un análisis comparativo entre K-Means y DBSCAN en el dataset sintético FIRE UdeA, evaluar la calidad de los clustering mediante múltiples métricas y explorar la estabilidad del clustering con UMAP.

**Contenido**:
- `clustering_analysis.py`: Script análisis comparativo de K-Means vs DBSCAN con visualización PCA
- `clustering_pipeline_umap_stability.py`: Pipeline de clustering con reducción UMAP y análisis de estabilidad
- `outputs_pocas_dim/`: Visualizaciones de clustering con pocas dimensiones
- `outputs_realista/`: Análisis en dataset realista con todas las características

**Conceptos cubiertos**:
- Preprocesamiento genérico compatible con múltiples datasets
- Comparación directa: K-Means (k=2) vs DBSCAN
- Métricas de evaluación de clustering:
  - Silhouette Score (medida de cohesión y separación)
  - Davies-Bouldin Index (validación de clusters)
  - Calinski-Harabasz Index (ratio dispersión/densidad)
- Visualización multidimensional con PCA
- Reducción dimensionalidad alternativa con UMAP
- Estabilidad del clustering bajo diferentes inicializaciones
- Matriz de confusión y análisis de precisión de predicciones
- Automatización para trabajar con diferentes datasets

## Requisitos

Para ejecutar cualquiera de los scripts, asegúrate hacer la instalación de las librerías:

```bash
pip install -r requirements.txt
```

## Autor

Isabella Camacho Monsalve

## Curso

Fundamentos de Aprendizaje Automático
