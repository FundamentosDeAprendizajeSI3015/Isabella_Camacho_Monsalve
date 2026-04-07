# Taller de Machine Learning: Predicción de Ajuste Estudiante-Carrera

Este taller explora la predicción de la probabilidad de buen ajuste entre un estudiante y su carrera, utilizando información académica, expectativas y experiencia temprana. Se implementa un enfoque de aprendizaje no supervisado (clustering) para reevaluar etiquetas de datos y mejorar la precisión de modelos supervisados.

## 1. Planteamiento del Problema

El objetivo es predecir si un estudiante tendrá un "alto ajuste" (1) o un "bajo ajuste" (0) con su carrera. Esto se logra analizando datos de encuestas en escala Likert (1 a 5) que capturan percepciones, hábitos y motivaciones de los estudiantes.

## 2. Dataset

Los datos de entrada provienen de una encuesta con las siguientes variables:
`Id`, `Hora de inicio`, `Hora de finalizaci n`, `Correo electr nico`, `Nombre`, `semester`, `career`, `interest_field`, `expectation_clarity`, `expectation_alignment`, `career_expectations`, `perceived_difficulty`, `study_habits`, `time_management`, `academic_stress`, `capacity_alignment`, `academic_enjoyment`, `area_identification`, `intrinsic_motivation`, `extrinsic_motivation`, `social_impact`, `professional_value`.

Los archivos de datos se encuentran en formato parquet:
- `X_raw.parquet`: Contiene las características de entrada (sin escalar).
- `y.parquet`: Contiene las etiquetas originales de ajuste (`0` para bajo ajuste, `1` para alto ajuste).

**Ubicación de los datos:** `Isabella_Camacho_Monsalve/Informe1/extended_ds/`

## 3. Metodología

La metodología se estructura en las siguientes fases:

### 3.1. Carga y Preprocesamiento de Datos

1.  **Carga**: Los datos `X_raw.parquet` y `y.parquet` se cargan desde la ruta especificada.
2.  **Limpieza**: Eliminación de columnas no relevantes (`Id`, `Hora de inicio`, `Hora de finalizaci n`, `Correo electr nico`, `Nombre`).
3.  **Manejo de Duplicados**: Se eliminan filas duplicadas para asegurar la unicidad de las observaciones.
4.  **Escalamiento**: Las variables numéricas son escaladas utilizando `StandardScaler`.
5.  **Reducción de Dimensionalidad (PCA)**: Se aplica PCA para reducir la dimensionalidad, manteniendo el 95% de la varianza explicada.

### 3.2. Análisis No Supervisado (Clustering)

Se aplican varios algoritmos de clustering para identificar patrones latentes en los datos y potencialmente redefinir las etiquetas de ajuste:

-   **K-Means**: Agrupamiento basado en centroides.
-   **Fuzzy C-Means**: Permite que una muestra pertenezca a múltiples clústeres con diferentes grados de pertenencia.
-   **Subtractive Clustering**: Identifica los centros de los clústeres basándose en la densidad de los puntos de datos.
-   **DBSCAN**: Agrupamiento basado en la densidad, adecuado para detectar clústeres de formas arbitrarias y ruido.

Para los algoritmos que no escalan bien a grandes conjuntos de datos (Fuzzy C-Means y Subtractive Clustering), se utiliza una **muestra estratificada** del dataset, manteniendo la proporción de las clases originales. Cada algoritmo produce etiquetas binarias donde `1` indica un cluster "anómalo" (en este contexto, un patrón distinto de ajuste) y `0` un cluster "normal".

### 3.3. Reevaluación de Etiquetas por Ensemble Ponderado

Las etiquetas generadas por los algoritmos de clustering se combinan para reevaluar las etiquetas originales de los estudiantes.

1.  **Cálculo de "Lift"**: Cada algoritmo recibe un peso igual a su "lift", que mide cuánto supera la tasa base de la clase minoritaria (bajo ajuste) en su cluster "anómalo".
2.  **Reetiquetado Conservador**: Si una mayoría ponderada de los algoritmos de clustering marca una muestra como "anómala" (indicando bajo ajuste), la etiqueta original se reasigna a "bajo ajuste" (0). Este enfoque es conservador, solo se añaden etiquetas de la clase minoritaria, y las etiquetas existentes de "alto ajuste" (1) nunca se revierten.

### 3.4. Entrenamiento de Modelos Supervisados

Se entrenan modelos de clasificación supervisada con las etiquetas originales y con las etiquetas reevaluadas para comparar su rendimiento.

-   **Modelos**:
    -   Random Forest Classifier
    -   Logistic Regression

-   **Estrategia de Entrenamiento**:
    -   División del dataset en conjuntos train, val, test.
    -   **Balanceo de Clases**: Se aplica SMOTE (Synthetic Minority Over-sampling Technique) en el conjunto de entrenamiento para abordar el desequilibrio de clases, especialmente crucial en datasets con pocas muestras.
    -   **Métrica Principal**: Se utiliza el Área bajo la Curva de Precisión-Recall (AUPRC), recomendada para datasets desbalanceados, junto con otras métricas como Precisión, Recall, F1-Score y Accuracy.

### 3.5. Comparación y Visualizaciones

Se genera una tabla comparativa de métricas para evaluar el rendimiento de los modelos entrenados con etiquetas originales versus reevaluadas. Se incluyen visualizaciones para facilitar la comprensión de los resultados del clustering, la distribución de las etiquetas y el rendimiento de los modelos.

## 4. Estructura del Código

El código se encuentra en un único script (`main.py` o similar) con las siguientes secciones:

1.  Carga y preprocesamiento
2.  Clustering (K-Means, Fuzzy C-Means, Subtractive, DBSCAN)
3.  Análisis de patrones por clustering
4.  Reevaluación de etiquetas por ensemble ponderado
5.  Modelos supervisados (Random Forest + Logistic Regression) con etiquetas originales y reevaluadas
6.  Tabla comparativa de métricas
7.  Visualizaciones

## 5. Ejecución del Código

Para ejecutar el análisis completo, navega a la carpeta `Isabella_Camacho_Monsalve/Informe2` en tu terminal y corre el script principal.
