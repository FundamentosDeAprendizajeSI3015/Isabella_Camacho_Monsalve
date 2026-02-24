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

## Requisitos

Para ejecutar cualquiera de los scripts, asegúrate de tener instaladas las siguientes librerías:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn
```

## Autor

Isabella Camacho Monsalve

## Curso

Fundamentos de Aprendizaje Automático
