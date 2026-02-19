# Proyecto: Predicción del Ajuste Estudiante-Carrera mediante Machine Learning

Este repositorio contiene el pipeline de Análisis Exploratorio de Datos (EDA) y Preprocesamiento para el proyecto de investigación sobre el **Ciclo de Vida del Aprendizaje Automático** aplicado al entorno universitario.

## Introducción y Objetivos
Este proyecto aborda una problemática crítica en la educación superior:

> "Las universidades carecen de una herramienta objetiva y temprana para identificar si un estudiante presenta un buen o mal ajuste con la carrera que eligió, más allá de su rendimiento académico." 

### Pregunta de Investigación
¿Puede un modelo de aprendizaje automático predecir la probabilidad de buen ajuste estudiante-carrera utilizando información académica, expectativas y experiencia temprana del estudiante?

### Alcance del Modelo
* **Finalidad:** Diagnóstica, no de recomendación. Se busca evitar la "caja negra" priorizando la **interpretabilidad** de los factores que generan desalineaciones críticas.
* **Enfoque:** El modelo se limita inicialmente a una sola carrera (**Ingeniería de Sistemas**) para reducir la heterogeneidad de los datos y aumentar la precisión.

---

## Diccionario de Datos (Variables)

El conjunto de datos contiene información sobre el ajuste académico y vocacional de los estudiantes de Ingeniería de Sistemas. A continuación se detallan las variables de entrada, su nombre técnico y la pregunta o ítem correspondiente de la encuesta.

### Información Demográfica y Académica
* **`semester`**: Semestre académico que cursa el estudiante actualmente.
* **`major`**: Carrera o programa académico en el que se encuentra inscrito.

### Expectativas y Ajuste de Carrera
* **`interest_field`**: Mi interés por el campo profesional de la carrera es alto.
* **`expectation_clarity`**: Antes de entrar, tenía claro de qué trataba la carrera.
* **`expectation_alignment`**: La carrera ha cumplido mis expectativas iniciales.
* **`career_expectations`**: La carrera cumple mis expectativas laborales futuras.

### Desempeño y Hábitos Académicos
* **`perceived_difficulty`**: El nivel de dificultad de las materias es alto.
* **`study_habits`**: Tengo buenos hábitos de estudio.
* **`time_management`**: Administro bien mi tiempo académico.
* **`academic_stress`**: Mi nivel de estrés académico es alto.

### Alineación Personal y Motivación
* **`capacity_alignment`**: Mis capacidades personales se alinean con las exigencias de la carrera.
* **`academic_enjoyment`**: Disfruto las actividades académicas propias de la carrera.
* **`area_identification`**: Me siento identificado/a con el área de conocimiento de la carrera.
* **`intrinsic_motivation`**: Mi motivación por la carrera es principalmente personal/vocacional.
* **`extrinsic_motivation`**: Elegí esta carrera principalmente por factores externos (empleo, ingresos, presión).

### Valor Social y Profesional
* **`social_impact`**: Considero que esta carrera tiene un impacto positivo en la sociedad.
* **`professional_value`**: La carrera es valorada social y profesionalmente.

---

### Nota sobre la escala de medición
Todas las variables de percepción (ítems de la encuesta) utilizan una **Escala Likert de 5 puntos**, donde los valores numéricos corresponden a:

| Valor | Etiqueta |
| :---: | :--- |
| 1 | Muy en desacuerdo |
| 2 | En desacuerdo |
| 3 | Ni de acuerdo ni en desacuerdo |
| 4 | De acuerdo |
| 5 | Muy de acuerdo |

---

## Funcionamiento del Código (Pipeline de Preprocesamiento)

El script `eda_pipeline.py` realiza un flujo completo de limpieza, transformación, ingeniería de características y preparación de datos para modelos de clasificación.

### 1. Carga y Limpieza Inicial
* **Filtrado:** Selecciona exclusivamente a los estudiantes de **Ingeniería de Sistemas**.
* **Tratamiento de Escalas:** Transforma las respuestas cualitativas de la **Escala de Likert** (18 preguntas originales) en valores numéricos del 1 al 5.
* **Gestión de Nulos:** Elimina registros con valores faltantes para asegurar la integridad estadística.

### 2. Análisis Estadístico y Visualización
* **Estadística Descriptiva:** Calcula media, mediana, moda, desviación estándar y rango intercuartílico (IQR). Se exporta a `descriptive_statistics.csv`.
* **Detección de Outliers:** Utiliza el método IQR para identificar casos atípicos en el comportamiento estudiantil.
* **Visualizaciones:** Genera histogramas de distribución de frecuencias y boxplots de todas las variables Likert.

### 3. Ingeniería de Características (Feature Engineering)
El código crea nuevos índices coherentes con los constructos teóricos del proyecto:

* **Índice de Motivación Neta:** `motivation_net = intrinsic_motivation - extrinsic_motivation`
  - Diferencia entre motivación personal/vocacional y factores externos (empleo, ingresos, presión)

* **Índice de Ajuste Académico:** `academic_adjustment = media(difficulty_reversed, study_habits, time_management)`
  - Combina la dificultad percibida (invertida), hábitos de estudio y administración del tiempo
  - **Este es el índice base para crear la variable objetivo**

### 4. Control de Multicolinealidad
* **Análisis de Correlación:** Genera heatmaps (matrices de calor) para identificar variables altamente correlacionadas.
* **Reducción de Dimensiones:** Elimina automáticamente variables con correlación superior a 0.85 para evitar sobreajuste y problemas de multicolinealidad.
* **Heatmaps generados:**
  - `correlation_matrix.png`: Variables originales
  - `correlation_matrix_fe.png`: Posterior a feature engineering
  - `correlation_matrix_reduced.png`: Conjunto final de características

### 5. Creación de Variable Objetivo (Target)
* **Variable:** `high_adjustment` (clasificación binaria)
* **Definición:** Basada en la **mediana** del índice `academic_adjustment`
  - Clase 1: Estudiantes con ajuste académico ALTO (>mediana)
  - Clase 0: Estudiantes con ajuste académico BAJO (≤mediana)
* **Justificación:** La mediana proporciona una división equitativa del espacio de características y es una métrica robusta ante outliers.

### 6. División Estratificada (Train/Val/Test)
* **Proporción:** 70% Entrenamiento | 15% Validación | 15% Test
* **Estratificación:** Mantiene la proporción de clases en cada subset
* **Random State:** Seed 42 asegura reproducibilidad

### 7. Balanceo de Clases
* **Método:** Downsampling de la clase mayoritaria en el conjunto de entrenamiento
  - Se reduce la clase mayoritaria al tamaño de la clase minoritaria
  - Evita sesgos hacia la clase más frecuente
* **Conjuntos afectados:** Solo entrenamiento (val y test permanecen sin alterar para evaluación honesta)

### 8. Escalado (Standardization)
* **Método:** `StandardScaler` de scikit-learn (media=0, desviación estándar=1)
* **Aplicación:** Solo a características numéricas
* **Fit:** Realizado sobre entrenamiento y aplicado a validación y test (evita información leakage)

### 9. Visualización de Espacios Reducidos
Genera proyecciones bidimensionales para explorar la estructura de los datos:
* **PCA (Principal Component Analysis):** Captura la máxima varianza en comportamiento lineal
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Visualiza la similitud local entre estudiantes
* **UMAP (Uniform Manifold Approximation and Projection):** Preserva estructura global y local

---

## Estructura de Salidas (Folder: `outputs/`)
Tras ejecutar el código, se generarán los siguientes archivos:

### Visualizaciones EDA
* `likert_distribution.png`: Histogramas de frecuencia para cada variable Likert
* `boxplot_likert.png`: Diagrama de cajas comparativo de variables Likert
* `correlation_matrix.png`: Mapa de calor con las variables originales (18 características)
* `correlation_matrix_fe.png`: Mapa de calor posterior a feature engineering
* `correlation_matrix_reduced.png`: Mapa de calor del conjunto final (sin variables correlacionadas >0.85)

### Visualizaciones de Reducción Dimensional
* `pca_plot.png`: Proyección PCA de 2 componentes (con varianza explicada)
* `tsne_plot.png`: Proyección t-SNE de 2 dimensiones
* `umap_plot.png`: Proyección UMAP de 2 dimensiones

### Datos Procesados (Formato Parquet)
**Conjuntos de Entrenamiento, Validación y Test con datos escalados:**
* `X_train.parquet`: Características de entrenamiento (70% del dataset, balanceado)
* `y_train.parquet`: Variable objetivo del entrenamiento
* `X_val.parquet`: Características de validación (15% del dataset)
* `y_val.parquet`: Variable objetivo de validación
* `X_test.parquet`: Características de test (15% del dataset, sin afectar)
* `y_test.parquet`: Variable objetivo de test

**Nota:** Los datasets están escalados (StandardScaler) y listos para entrenar modelos de clasificación. El conjunto de entrenamiento ha sido balanceado mediante downsampling de la clase mayoritaria.

### Estadísticas Descriptivas
* `descriptive_statistics.csv`: Tabla con media, mediana, moda, desviación estándar, rango, cuartiles e IQR de todas las variables Likert

---

## Requisitos
* **Python:** 3.8 o superior
* **Librerías principales:**
  - `pandas` (manipulación y análisis de datos)
  - `numpy` (operaciones numéricas)
  - `matplotlib` (visualización estática)
  - `seaborn` (visualización avanzada)
  - `scikit-learn` (preprocessing, dimensionality reduction, train_test_split)
  - `umap` (visualización UMAP)

### Instalación
```bash
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn
```

---

## Flujo de Ejecución del Pipeline

```
1. Carga datos (survey.csv)
   ↓
2. Filtrado estudiantes de Ingeniería de Sistemas
   ↓
3. Transformación Escala Likert (texto → números 1-5)
   ↓
4. Eliminación de nulos
   ↓
5. Estadística Descriptiva + Visualizaciones (histogramas, boxplots)
   ↓
6. Feature Engineering (motivation_net, academic_adjustment)
   ↓
7. Análisis de Correlación + Reducción de variables (r > 0.85)
   ↓
8. Creación de Variable Objetivo (high_adjustment basada en mediana)
   ↓
9. División Estratificada (70% train | 15% val | 15% test)
   ↓
10. Balanceo de clases (downsampling en train)
    ↓
11. Escalado StandardScaler (media=0, std=1)
    ↓
12. Visualización Dimensional (PCA, t-SNE, UMAP)
    ↓
13. Exportación de datos (parquet) y estadísticas (CSV)
```

---

## Interpretación de Resultados Clave

### Índices Derivados
- **`motivation_net` > 0:** Motivación predominantemente intrínseca (carrera elegida por vocación)
- **`motivation_net` < 0:** Motivación predominantemente extrínseca (presión externa, ingresos, etc.)
- **`academic_adjustment` > mediana:** Estudiante con buenos hábitos de estudio, baja percepción de dificultad y buena gestión del tiempo
- **`high_adjustment` = 1:** Predicción: Buen ajuste estudiante-carrera

### Multicolinealidad
- Variables eliminadas (r > 0.85): Las presentes en `to_drop` impresas durante ejecución
- Esto reduce la redundancia y mejora la interpretabilidad del modelo

---

**Presentado por:** Isabella Camacho
**Enfoque:** Desarrollo de modelos con interpretabilidad para acompañamiento académico temprano.
**Última actualización:** 2025
