# Titanic EDA Script
# Exporta todas las gráficas como .jpg en la carpeta "outputs"

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 0. Crear carpeta de salida
# -----------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------
# 2. Limpieza básica y medidas de dispersión y posición
# -----------------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Seleccionar solo columnas numéricas
df_numeric = df.select_dtypes(include="number")

print("\n==============================")
print("CÁLCULO AUTOMÁTICO CON MÉTODO .describe()")
print("==============================")

df_numeric = df.select_dtypes(include="number").drop(columns=["PassengerId"])
print(df_numeric.describe())

print("\n==============================")
print("COLUMNAS NUMÉRICAS")
print("==============================")
print(df_numeric.columns)

# -----------------------------
# Medidas de dispersión
# -----------------------------
print("\n==============================")
print("VARIANZA")
print("==============================")
print(df_numeric.var())

print("\n==============================")
print("DESVIACIÓN ESTÁNDAR")
print("==============================")
print(df_numeric.std())

Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

print("\n==============================")
print("RANGO INTERCUARTÍLICO (IQR)")
print("==============================")
print(IQR)

# -----------------------------
# Medidas de posición
# -----------------------------
print("\n==============================")
print("MEDIA")
print("==============================")
print(df_numeric.mean())

print("\n==============================")
print("MEDIANA")
print("==============================")
print(df_numeric.median())

print("\n==============================")
print("CUARTILES")
print("==============================")
print(df_numeric.quantile([0.25, 0.5, 0.75]))


# -----------------------------
# Detección de outliers en Age
# -----------------------------
Q1_age = df["Age"].quantile(0.25)
Q3_age = df["Age"].quantile(0.75)
IQR_age = Q3_age - Q1_age

limite_inferior = Q1_age - 1.5 * IQR_age
limite_superior = Q3_age + 1.5 * IQR_age

outliers_age = df[
    (df["Age"] < limite_inferior) |
    (df["Age"] > limite_superior)
]

print("\n==============================")
print("OUTLIERS EN AGE")
print("==============================")
print(f"Límite inferior: {limite_inferior}")
print(f"Límite superior: {limite_superior}")
print(f"Cantidad de outliers detectados: {len(outliers_age)}")
print("\nPrimeros outliers encontrados:")
print(outliers_age.head())

# -----------------------------
# 3. Histogramas
# -----------------------------
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

for col in numeric_cols:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hist_{col}.jpg")
    plt.close()

# -----------------------------
# 4. Boxplots
# -----------------------------
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_{col}.jpg")
    plt.close()

# -----------------------------
# 5. Gráficos de barras
# -----------------------------
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Supervivencia por Sexo")
plt.tight_layout()
plt.savefig(f"{output_dir}/bar_survival_sex.jpg")
plt.close()

plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Supervivencia por Clase")
plt.tight_layout()
plt.savefig(f"{output_dir}/bar_survival_pclass.jpg")
plt.close()

# -----------------------------
# 6. Scatter plots
# -----------------------------
plt.figure()
plt.scatter(df['Age'], df['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare")
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_age_fare.jpg")
plt.close()

plt.figure()
plt.scatter(df['Age'], df['Survived'])
plt.xlabel("Age")
plt.ylabel("Survived")
plt.title("Age vs Survived")
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_age_survived.jpg")
plt.close()

# -----------------------------
# 7. Correlation Heatmap
# -----------------------------
df_corr = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.jpg")
plt.close()

# -----------------------------
# 8. Transformación Log
# -----------------------------
df['Fare_log'] = np.log1p(df['Fare'])

plt.figure()
plt.hist(df['Fare_log'], bins=30)
plt.title("Distribución de Fare (Log)")
plt.tight_layout()
plt.savefig(f"{output_dir}/hist_fare_log.jpg")
plt.close()

# -----------------------------
# 9. Encoding y Scaling
# -----------------------------
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scale_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df_encoded[scale_cols] = scaler.fit_transform(df_encoded[scale_cols])

print("EDA finalizada. Todas las gráficas fueron exportadas en la carpeta 'outputs'.")
