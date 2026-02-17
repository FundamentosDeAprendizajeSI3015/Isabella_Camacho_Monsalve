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

El script proporcionado realiza un flujo completo de limpieza y transformación de datos, preparando el dataset para modelos de clasificación o regresión.

### 1. Carga y Limpieza Inicial
* **Filtrado:** Selecciona exclusivamente a los estudiantes de **Ingeniería de Sistemas**.
* **Tratamiento de Escalas:** Transforma las respuestas cualitativas de la **Escala de Likert** (18 preguntas originales) en valores numéricos del 1 al 5.
* **Gestión de Nulos:** Identifica y elimina registros con valores faltantes para asegurar la integridad estadística.

### 2. Análisis Estadístico y Visualización
* **Estadística Descriptiva:** Calcula media, mediana, moda, desviación estándar y rango intercuartílico (IQR).
* **Detección de Outliers:** Utiliza el método IQR para identificar casos atípicos en el comportamiento estudiantil.
* **Visualizaciones:** Genera histogramas de distribución de frecuencias y boxplots de todas las variables involucradas.

### 3. Ingeniería de Características (Feature Engineering)
El código crea nuevos índices basados en los constructos teóricos del proyecto:
* **Índice de Motivación Neta:** Diferencia entre motivación intrínseca y extrínseca.
* **Índice de Ajuste Académico:** Promedio ponderado de los hábitos de estudio, gestión del tiempo y la percepción invertida de la dificultad.

### 4. Control de Multicolinealidad y Estandarización
* **Análisis de Correlación:** Genera mapas de calor (heatmaps) para identificar variables altamente correlacionadas.
* **Reducción de Dimensiones:** Elimina automáticamente variables con una correlación superior al 0.85 para evitar el sobreajuste.
* **Escalamiento:** Aplica `StandardScaler` para normalizar los datos, permitiendo que variables con diferentes rangos contribuyan equitativamente al modelo de ML.

---

## Estructura de Salidas (Folder: `outputs/`)
Tras ejecutar el código, se generarán los siguientes archivos:
* `likert_distribution.png`: Gráficos de barras con la tendencia de cada respuesta.
* `correlation_matrix.png`: Mapa de calor con las variables de entrada sin modificación.
* `correlation_matrix_reduced.png`: Mapa de calor al eliminar las variables altamente correlacionadas.
* `correlation_matrix_fe.png`: Mapa de calor posterior a feature engineering.
* `boxplot_likert.png`: Diagrama de cajas para representar la distribución de las variables likert.
* `descriptive_statistics.csv`: Resumen estadístico completo.
* `final_scaled_dataset.csv`: **Dataset final listo para entrenar modelos** (Regresión Logística, Árboles de Decisión o Random Forest).

---

## Requisitos
* Python 3.x
* Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---
**Presentado por:** Isabella Camacho  
**Enfoque:** Desarrollo de modelos con interpretabilidad para el acompañamiento académico temprano.
