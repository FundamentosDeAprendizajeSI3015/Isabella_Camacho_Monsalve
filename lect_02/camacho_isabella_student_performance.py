"""
Modelo de Predicción del Rendimiento Estudiantil (G3)
Dataset: Student Performance UCI
Modelo: Regresión Logística
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, precision_recall_curve, auc
)

import pickle

# Configuración
np.random.seed(42)
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(filename):
    """Guarda gráfica en outputs"""
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
print("\n" + "="*80)
print("CARGA DE DATOS")
print("="*80)

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo", "-q"])
    from ucimlrepo import fetch_ucirepo

student_performance = fetch_ucirepo(id=320)
X_raw = student_performance.data.features
y_raw = student_performance.data.targets
df = pd.concat([X_raw, y_raw], axis=1)

print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} variables")

# =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================
print("\n" + "="*80)
print("PREPROCESAMIENTO")
print("="*80)

# Eliminar duplicados
df = df.drop_duplicates()

# Codificación de variables
df_encoded = df.copy()

# Variables binarias (yes/no -> 1/0)
binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in binary_cols:
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map({'yes': 1, 'no': 0})

# One-Hot Encoding
nominal_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
df_encoded = pd.get_dummies(df_encoded, columns=[col for col in nominal_cols if col in df_encoded.columns], 
                           drop_first=True, dtype=int)

print(f"Variables codificadas: {df_encoded.shape[1]} columnas")

# =============================================================================
# 3. PREPARACIÓN DE DATOS
# =============================================================================
print("\n" + "="*80)
print("PREPARACIÓN DE DATOS")
print("="*80)

# Separar X e y (sin G1 y G2)
X = df_encoded.drop(['G1', 'G2', 'G3'], axis=1)
y = df_encoded['G3']

# Variable objetivo binaria: Aprobado (>=10) vs Reprobado (<10)
y_binary = (y >= 10).astype(int)

print(f"Características: {X.shape[1]}")
print(f"Reprobados: {(y_binary == 0).sum()} ({(y_binary == 0).sum()/len(y_binary)*100:.1f}%)")
print(f"Aprobados: {(y_binary == 1).sum()} ({(y_binary == 1).sum()/len(y_binary)*100:.1f}%)")

# División train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# =============================================================================
# 4. NORMALIZACIÓN
# =============================================================================
print("\n" + "="*80)
print("NORMALIZACIÓN")
print("="*80)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Datos normalizados con RobustScaler")

# =============================================================================
# 5. ENTRENAMIENTO - REGRESIÓN LOGÍSTICA
# =============================================================================
print("\n" + "="*80)
print("ENTRENAMIENTO - REGRESIÓN LOGÍSTICA")
print("="*80)

# Grid Search para optimizar hiperparámetros
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

print("Iniciando Grid Search...")
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nMejores parámetros:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Mejor F1-Score (CV): {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

# =============================================================================
# 6. VALIDACIÓN CRUZADA
# =============================================================================
print("\n" + "="*80)
print("VALIDACIÓN CRUZADA (10-Fold)")
print("="*80)

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='f1')
print(f"F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =============================================================================
# 7. EVALUACIÓN EN TEST
# =============================================================================
print("\n" + "="*80)
print("EVALUACIÓN EN TEST")
print("="*80)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\nMÉTRICAS:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC AUC:   {roc_auc:.4f}")

print(f"\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Reprobado', 'Aprobado']))

# =============================================================================
# 8. VISUALIZACIONES
# =============================================================================
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
           xticklabels=['Reprobado', 'Aprobado'],
           yticklabels=['Reprobado', 'Aprobado'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title(f'Matriz de Confusión\nAccuracy: {accuracy:.3f} | F1: {f1:.3f}')
save_plot('matriz_confusion.png')
print("  ✓ matriz_confusion.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_plot = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'Logistic Regression (AUC = {roc_auc_plot:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
save_plot('roc_curve.png')
print("  ✓ roc_curve.png")

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, linewidth=2, label=f'Logistic Regression (AUC = {pr_auc:.3f})')
baseline = y_test.sum() / len(y_test)
plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, label=f'Baseline ({baseline:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
save_plot('precision_recall_curve.png')
print("  ✓ precision_recall_curve.png")

# Importancia de coeficientes
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False).head(15)

plt.figure(figsize=(10, 6))
colors = ['red' if x < 0 else 'green' for x in coefficients['Coefficient']]
plt.barh(range(len(coefficients)), coefficients['Coefficient'], color=colors, edgecolor='black')
plt.yticks(range(len(coefficients)), coefficients['Feature'])
plt.xlabel('Coeficiente')
plt.title('Top 15 Coeficientes más Importantes')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
save_plot('coeficientes.png')
print("  ✓ coeficientes.png")

# =============================================================================
# 9. GUARDAR MODELO
# =============================================================================
print("\n" + "="*80)
print("GUARDAR MODELO")
print("="*80)

with open(os.path.join(OUTPUT_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("  ✓ model.pkl")
print("  ✓ scaler.pkl")

# Resumen
summary = pd.DataFrame([{
    'modelo': 'Logistic Regression',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'best_params': str(grid_search.best_params_)
}])

summary.to_csv(os.path.join(OUTPUT_DIR, 'resumen.csv'), index=False)
print("  ✓ resumen.csv")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"\nModelo: Regresión Logística")
print(f"Parámetros: {grid_search.best_params_}")
print(f"\nRendimiento en Test:")
print(f"  • Accuracy:  {accuracy:.2%}")
print(f"  • Precision: {precision:.2%}")
print(f"  • Recall:    {recall:.2%}")
print(f"  • F1-Score:  {f1:.2%}")
print(f"  • ROC AUC:   {roc_auc:.2%}")
print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
print(f"  • 4 gráficas PNG")
print(f"  • 2 archivos de modelo")
print(f"  • 1 resumen CSV")
print("\n" + "="*80)
print("COMPLETADO")
print("="*80 + "\n")