# Titanic Survival Classification Model
# Basado en el ejemplo de Regresión Logística
# Objetivo: Predecir si un pasajero sobrevivió o no al hundimiento del Titanic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import reciprocal

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score
)

# =============================
# Configuración inicial
# =============================
random_state = 42
plt.rc('font', family='serif', size=12)

# =============================
# 1. Cargar el dataset
# =============================
print("=" * 50)
print("CARGANDO DATASET DEL TITANIC")
print("=" * 50)

df = pd.read_csv("../lect_04/Titanic-Dataset.csv")
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print("\nPrimeras filas:")
print(df.head())

# =============================
# 2. Limpieza y Preparación de Datos
# =============================
print("\n" + "=" * 50)
print("PREPARACIÓN DE DATOS")
print("=" * 50)

# Rellenar valores faltantes
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Información sobre valores faltantes
print("\nValores faltantes:")
print(df.isnull().sum())

# =============================
# 3. Seleccionar características y target
# =============================
# Características numéricas y categóricas relevantes
features_numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
features_categorical = ['Sex', 'Embarked']

# Crear dataset procesado
X = df[features_numeric + features_categorical].copy()
y = df['Survived']

print("\nCaracterísticas seleccionadas:")
print(f"Numéricas: {features_numeric}")
print(f"Categóricas: {features_categorical}")
print(f"\nTarget: Survived (0 = No sobrevivió, 1 = Sobrevivió)")
print(f"Distribución: {y.value_counts().to_dict()}")

# =============================
# 4. Codificar variables categóricas
# =============================
X = pd.get_dummies(X, columns=features_categorical, drop_first=True)

print(f"\nCaracterísticas después de encoding: {X.shape[1]}")
print(X.head())

# =============================
# 5. Dividir en conjuntos de entrenamiento y prueba
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)

print("\n" + "=" * 50)
print("DIVISIÓN DE DATOS")
print("=" * 50)
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# =============================
# 6. Definir el piepline del modelo
# =============================
print("\n" + "=" * 50)
print("CONSTRUCCIÓN DEL MODELO")
print("=" * 50)

# Pipeline con escalado y regresión logística
lr_base = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Distribuciones de parámetros para la búsqueda
param_distributions = {
    'classifier__C': reciprocal(1e-5, 1e5),
    'classifier__solver': ['lbfgs', 'liblinear']
}

# Modelo con búsqueda de hiperparámetros
lr = RandomizedSearchCV(
    lr_base,
    cv=5,
    param_distributions=param_distributions,
    n_iter=100,
    random_state=random_state,
    scoring='f1',
    n_jobs=-1
)

print("Modelo configurado con RandomizedSearchCV")
print(f"CV: 5")
print(f"Iteraciones: 100")

# =============================
# 7. Entrenar el modelo
# =============================
print("\n" + "=" * 50)
print("ENTRENANDO EL MODELO...")
print("=" * 50)

lr.fit(X_train, y_train)

print("¡Entrenamiento completado!")
print(f"\nMejores parámetros encontrados:")
for param, value in lr.best_params_.items():
    print(f"  {param}: {value}")

# =============================
# 8. Evaluación del modelo
# =============================
print("\n" + "=" * 50)
print("EVALUACIÓN DEL MODELO")
print("=" * 50)

# Predicciones
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Métricas en entrenamiento
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Métricas en prueba
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nMÉTRICAS DE ENTRENAMIENTO:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1 Score:  {train_f1:.4f}")

print("\nMÉTRICAS DE PRUEBA:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

# =============================
# 9. Matriz de Confusión
# =============================
print("\n" + "=" * 50)
print("MATRIZ DE CONFUSIÓN")
print("=" * 50)

cm = confusion_matrix(y_test, y_test_pred)
print(f"\nMatriz de confusión (datos de prueba):")
print(cm)

# Interpretar matriz de confusión
tn, fp, fn, tp = cm.ravel()
print(f"\nInterpretación:")
print(f"  Verdaderos Negativos (TN): {tn} - Predijo no sobrevivió correctamente")
print(f"  Falsos Positivos (FP):     {fp} - Predijo sobrevivió, pero no fue así")
print(f"  Falsos Negativos (FN):     {fn} - Predijo no sobrevivió, pero sí fue así")
print(f"  Verdaderos Positivos (TP): {tp} - Predijo sobrevivió correctamente")

# =============================
# 10. Visualización de la Matriz de Confusión
# =============================
disp = ConfusionMatrixDisplay(cm, display_labels=['No Sobrevivió', 'Sobrevivió'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title('Matriz de Confusión - Titanic Survival', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_titanic.png', dpi=150, bbox_inches='tight')
print("\n[OK] Matriz de confusion guardada como 'confusion_matrix_titanic.png'")

# =============================
# 11. Importancia de características
# =============================
print("\n" + "=" * 50)
print("IMPORTANCIA DE CARACTERÍSTICAS")
print("=" * 50)

# Obtener los coeficientes del modelo entrenado
lr_model = lr.best_estimator_
coefficients = lr_model.named_steps['classifier'].coef_[0]
feature_names = X.columns

# Crear dataframe with importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nCoeficientes del modelo (ordenados por importancia):")
print(importance_df.to_string(index=False))

# Visualizar importancia
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in importance_df['Coefficient']]
ax.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
ax.set_xlabel('Coeficiente', fontsize=12)
ax.set_title('Importancia de Características - Regresión Logística',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('feature_importance_titanic.png', dpi=150, bbox_inches='tight')
print("[OK] Grafico de importancia guardado como 'feature_importance_titanic.png'")

# =============================
# 12. Resumen final
# =============================
print("\n" + "=" * 50)
print("RESUMEN FINAL")
print("=" * 50)
print(f"""
El modelo de Regresion Logistica entrenado:

[OK] Alcanza una precision de {test_accuracy:.2%} en datos de prueba
[OK] Identifica correctamente el {test_recall:.2%} de los sobrevivientes
[OK] Tiene una precision de {test_precision:.2%} cuando predice que sobreviven

Caracteristicas mas influyentes en la prediccion:
1. {importance_df.iloc[0]['Feature']}: {importance_df.iloc[0]['Coefficient']:.4f}
2. {importance_df.iloc[1]['Feature']}: {importance_df.iloc[1]['Coefficient']:.4f}
3. {importance_df.iloc[2]['Feature']}: {importance_df.iloc[2]['Coefficient']:.4f}

Los valores positivos aumentan la probabilidad de supervivencia,
mientras que los valores negativos la disminuyen.
""")

print("\n[OK] Modelo completamente entrenado y evaluado")
print("=" * 50)
