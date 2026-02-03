
# =============================================================
# LAB FINTECH (SINTÉTICO 2025) — PREPROCESAMIENTO Y EDA
# Datos de entrada fijos para evitar errores de ruta/nombre.
# -------------------------------------------------------------
# Este script está listo para ejecutarse sin argumentos:
#   python lab_fintech_sintetico_2025.py
# 
# Archivos esperados en el mismo directorio:
#   - fintech_top_sintetico_2025.csv
#   - fintech_top_sintetico_dictionary.json
# Salidas (por defecto):
#   ./data_output_finanzas_sintetico/
#       ├─ eda_plots/
#       │    ├─ corr_heatmap.png
#       │    ├─ revenue_timeseries.png
#       │    └─ umap_fintechs.png
#       ├─ fintech_train.parquet
#       ├─ fintech_test.parquet
#       ├─ processed_schema.json
#       └─ features_columns.txt
#
# Diferencia con el material de clase recibido: Las funciones principales de este archivo están comentadas, 
# explicando cada paso del proceso de EDA e incluyendo algunas explicaciones teóricas para mejor comprensión.
# =============================================================

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import umap
import matplotlib.pyplot as plt

# ---------------------------
# Constantes de la práctica
# ---------------------------
DATA_CSV = 'fintech_top_sintetico_2025.csv'
DATA_DICT = 'fintech_top_sintetico_dictionary.json'
OUTDIR = Path('./data_output_finanzas_sintetico')
SPLIT_DATE = '2025-09-01'  # partición temporal por defecto
PLOTDIR = OUTDIR / 'eda_plots'

# Columnas esperadas por diseño del dataset sintético
DATE_COL = 'Month'
ID_COLS = ['Company']
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']
NUM_COLS = [
    'Users_M','NewUsers_K','TPV_USD_B','TakeRate_pct','Revenue_USD_M',
    'ARPU_USD','Churn_pct','Marketing_Spend_USD_M','CAC_USD','CAC_Total_USD_M',
    'Close_USD','Private_Valuation_USD_B'
]
PRICE_COLS = ['Close_USD']  # para calcular retornos opcionales

# ---------------------------
# 0) Carga de diccionario
# ---------------------------
print("\n=== 0) Cargando diccionario de datos ===")
dict_path = Path(DATA_DICT)
if not dict_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta.")

with open(dict_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)
print("Descripción:", data_dict.get('description', '(sin descripción)'))
print("Periodo:", data_dict.get('period', '(desconocido)'))

# ---------------------------
# 1) Carga del CSV
# ---------------------------
print("\n=== 1) Cargando CSV sintético ===")
csv_path = Path(DATA_CSV)
if not csv_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta.")

df = pd.read_csv(csv_path)
print("Shape:", df.shape)

# Parseo de fecha y orden temporal
if DATE_COL not in df.columns:
    raise KeyError(f"La columna de fecha '{DATE_COL}' no existe en el CSV.")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

print("Primeras filas:")
print(df.head(3))

# ---------------------------
# 2) EDA breve
# ---------------------------
print("\n=== 2) EDA rápido ===")
print("Info:")
# info() imprime en consola información sobre el DataFrame, incluyendo el número de entradas, columnas, tipos de datos y memoria utilizada. 
print(df.info())
# .isna().sum() calcula la cantidad de valores nulos en cada columna del DataFrame.
# .sort_values(ascending=False).head(15) ordena las columnas por la cantidad de valores nulos en orden descendente y muestra las 15 columnas con más nulos.
print("\nNulos por columna (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))

# ---------------------------
# 3) Limpieza básica
# ---------------------------
print("\n=== 3) Limpieza ===")
# Imputación simple: numéricos con mediana, categóricos con marcador
for c in NUM_COLS:
    # Si hay alguna columna numérica con nulos:
    if c in df.columns and df[c].isna().any():
        # .to_numeric(...) intenta convertir los valores a int o float.
        # errors='coerce' convierte valores no convertibles en NaN.
        df[c] = pd.to_numeric(df[c], errors='coerce')
        # .fillna(...) reemplaza los NaN en la columna c con la mediana de los valores de esta.
        df[c] = df[c].fillna(df[c].median())

for c in CAT_COLS:
    # Si hay alguna columna categórica con nulos:
    if c in df.columns and df[c].isna().any():
        # .fillna(...) reemplaza los NaN en la columna c con el string '__MISSING__'.
        df[c] = df[c].fillna('__MISSING__')

# ---------------------------
# 4) Ingeniería ligera: retornos/log-retornos de precio
# ---------------------------
# Explicación conceptual y estadística de retornos:
# -> Retorno: cambio porcentual mensual del precio
# -> Log-retorno: versión transformada y más estable estadísticamente, ya que penaliza menos los picos extremos respecto al retorno simple.
# En este practica, se calcula sobre 'Close_USD' con el objetivo de capturar la dinámica temporal del valor de mercado de las empresas fintech.
# Estos features son clave para modelos financieros y de series temporales.

# Si todas las columnas definidas en PRICE_COLS están en el DataFrame:
print("\n=== 4) Ingeniería de rasgos (retornos) ===")
if all([pc in df.columns for pc in PRICE_COLS]):
    for pc in PRICE_COLS:
        # Retornos por empresa y fecha
        # Se crea una nueva columna llamada pc + '_ret' que contiene los retornos porcentuales diarios del precio pc.
        df[pc + '_ret'] = (
            df.sort_values([ID_COLS[0], DATE_COL]) # Ordena los datos por empresa y fecha
              .groupby(ID_COLS)[pc] # Agrupa los datos por empresa y selecciona la columna de precio.
              .pct_change() # Calcula el cambio porcentual entre filas consecutivas dentro de cada grupo.

        )
        df[pc + '_logret'] = np.log1p(df[pc + '_ret']) # Calcula el log-retorno usando la función log1p para mayor precisión en valores cercanos a cero.
        # Imputar primeros NA en 0.0 para continuidad ( ya que los primeros valores no tienen retorno calculable)
        df[pc + '_ret'] = df[pc + '_ret'].fillna(0.0)
        df[pc + '_logret'] = df[pc + '_logret'].fillna(0.0)
else:
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Actualizamos lista de numéricos tras ingeniería
extra_num = [c for c in [pc + '_ret' for pc in PRICE_COLS] + [pc + '_logret' for pc in PRICE_COLS] if c in df.columns]
NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num

# ---------------------------
# Paso extra: EDA visual (babyplots + UMAP)
# ---------------------------
print("\n=== Paso extra: EDA visual ===")

PLOTDIR.mkdir(parents=True, exist_ok=True)

# Correlaciones (matplotlib)
print("Generando heatmap de correlaciones...")

corr_cols = [
    'Users_M',
    'TPV_USD_B',
    'Revenue_USD_M',
    'ARPU_USD',
    'Churn_pct',
    'Marketing_Spend_USD_M',
    'CAC_USD'
]

corr = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right')
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlación entre métricas fintech")
plt.tight_layout()

corr_path = PLOTDIR / "corr_heatmap.png"
plt.savefig(corr_path)
plt.close()

print("✓ Guardado:", corr_path)


# Serie temporal de ingresos por empresa
print("Generando series temporales de ingresos...")

plt.figure(figsize=(10, 6))

for company, g in df.groupby('Company'):
    plt.plot(g[DATE_COL], g['Revenue_USD_M'], label=company)

plt.xlabel("Mes")
plt.ylabel("Revenue (USD M)")
plt.title("Evolución temporal de ingresos por empresa")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()

ts_path = PLOTDIR / "revenue_timeseries.png"
plt.savefig(ts_path)
plt.close()

print("✓ Guardado:", ts_path)

# UMAP: Similitud entre fintechs
print("Generando UMAP de fintechs...")

umap_cols = [
    'Users_M',
    'TPV_USD_B',
    'Revenue_USD_M',
    'ARPU_USD',
    'Churn_pct',
    'CAC_USD'
]

umap_df = df[umap_cols].dropna()
meta = df.loc[umap_df.index, 'Segment']

X_scaled = StandardScaler().fit_transform(umap_df)

reducer = umap.UMAP(
    n_neighbors=5,
    min_dist=0.3,
    random_state=42
)

embedding = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

for seg in meta.unique():
    idx = meta == seg
    plt.scatter(
        embedding[idx, 0],
        embedding[idx, 1],
        label=seg,
        alpha=0.7
    )

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP de similitud entre fintechs")
plt.legend(fontsize=8)
plt.tight_layout()

umap_path = PLOTDIR / "umap_fintechs.png"
plt.savefig(umap_path)
plt.close()

print("✓ Guardado:", umap_path)

# ---------------------------
# 5) Separación X / y (sin y por defecto) + codificación
# ---------------------------
print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")
# Quitamos identificadores y fecha de las variables predictoras
# .copy() crea una copia independiente del df original para no modificarlo.
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy() # evita error si alguna columna no existe.

# One-hot en categóricas
cat_in_X = [c for c in CAT_COLS if c in X.columns] # toma solo las columnas categóricas presentes en X
# .get_dummies(...) convierte las columnas categóricas en variables dummy (one-hot encoding) creando columnas binarias.
# drop_first=True evita la multicolinealidad eliminando la primera categoría de cada variable.
# -> en problemas de regresión lineal este paso es importante para romper la dependencia lineal y así tener coeficientes estables.
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# Partición temporal por defecto utilizando la fecha de corte
cutoff = pd.to_datetime(SPLIT_DATE) # para poder compararla con la columna de fechas.

# Estas variables son seríes booleanas que indican si cada fila pertenece al conjunto de entrenamiento o prueba (con 0s y 1s)
idx_train = df[DATE_COL] < cutoff
idx_test = df[DATE_COL] >= cutoff

# En esta división de datos, se entrena el modelo con datos en el pasado (antes de la fecha de corte) y se prueba con datos futuros.
# Esto es correcto para series temporales y datos financieros (evita data leakage).
X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()

# Escalado de numéricos (solo columnas presentes en X)
num_in_X = [c for c in NUM_USED if c in X_train.columns]
scaler = StandardScaler() # Inicializa un escalador estándar (que escala los datos para que tengan media 0, desviación estándar 1).
if num_in_X:
    # .fit_transform(...) ajusta el escalador a los datos de entrenamiento y luego transforma esos datos.
    # .transform(...) aplica la transformación ya ajustada a los datos de prueba.
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X]) # Ajusta el scaler SOLO con el conjunto de entrenamiento, de esta manera se evita data leakage.
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])
else:
    print("[INFO] No se encontraron columnas numéricas para escalar.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# ---------------------------
# 6) Exportación
# ---------------------------
print("\n=== 6) Exportación ===")
OUTDIR.mkdir(parents=True, exist_ok=True)
train_path = OUTDIR / 'fintech_train.parquet'
test_path = OUTDIR / 'fintech_test.parquet'

# Guardamos sólo X (sin objetivo)
X_train.to_parquet(train_path, index=False)
X_test.to_parquet(test_path, index=False)

# Guardar esquema procesado
processed_schema = {
    'source_csv': str(csv_path.resolve()),
    'source_dict': str(dict_path.resolve()),
    'date_col': DATE_COL,
    'id_cols': ID_COLS,
    'categorical_cols_used': cat_in_X,
    'numeric_cols_used': num_in_X,
    'engineered_cols': extra_num,
    'split': {
        'type': 'time_split',
        'cutoff': SPLIT_DATE,
        'train_rows': int(idx_train.sum()),
        'test_rows': int(idx_test.sum()),
    },
    'X_train_shape': list(X_train.shape),
    'X_test_shape': list(X_test.shape),
    'notes': [
        'Dataset 100% SINTÉTICO con fines académicos; no refleja métricas reales.',
        'Evitar fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST.'
    ]
}

with open(OUTDIR / 'processed_schema.json', 'w', encoding='utf-8') as f:
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# Lista de columnas finales para referencia de modelado
with open(OUTDIR / 'features_columns.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train.columns))

print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / 'processed_schema.json')
print(" -", OUTDIR / 'features_columns.txt')

print("\n✔ Listo. Recuerda: este dataset es sintético para práctica académica.")
