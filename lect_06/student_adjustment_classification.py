"""
Student-Career Adjustment Classification Model
SI3015 - Fundamentos de Aprendizaje Automático

Este script entrena un Random Forest y un Gradient Boosting Classifier
para predecir el ajuste académico de estudiantes de Ingeniería de Sistemas.

El problema es clasificar si un estudiante tiene un ajuste académico ALTO o BAJO
basado en sus características académicas, expectativas y percepciones.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    auc
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Definir random_state para reproducibilidad
random_state = 42

# Configurar matplotlib
plt.rc('font', family='serif', size=12)
sns.set_style("whitegrid")

# Ruta a los datos
data_path = Path(__file__).parent.parent / "Informe1" / "outputs"

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("=" * 80)
print("CARGANDO DATOS DE ENTRENAMIENTO, VALIDACIÓN Y PRUEBA")
print("=" * 80)

# Cargar conjuntos de datos
X_train = pd.read_parquet(data_path / "X_train.parquet").head(10)  # Temporal: solo 10 datos
X_val = pd.read_parquet(data_path / "X_val.parquet")
X_test = pd.read_parquet(data_path / "X_test.parquet")

y_train = pd.read_parquet(data_path / "y_train.parquet").squeeze().head(10)  # Temporal: solo 10 datos
y_val = pd.read_parquet(data_path / "y_val.parquet").squeeze()
y_test = pd.read_parquet(data_path / "y_test.parquet").squeeze()

print(f"Conjunto de entrenamiento: X_train shape = {X_train.shape}")
print(f"Conjunto de validación: X_val shape = {X_val.shape}")
print(f"Conjunto de prueba: X_test shape = {X_test.shape}")
print(f"\nDistribución de clases en entrenamiento:\n{y_train.value_counts()}")
print(f"\nDistribución de clases en validación:\n{y_val.value_counts()}")
print(f"\nDistribución de clases en prueba:\n{y_test.value_counts()}")

# ============================================================================
# 2. EXPLORACIÓN INICIAL DE DATOS
# ============================================================================

print("\n" + "=" * 80)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 80)

print("\nCaracterísticas del conjunto de entrenamiento:")
print(X_train.describe())

print("\nNombres de las características:")
print(list(X_train.columns))

# ============================================================================
# 3. DEFINICIÓN Y ENTRENAMIENTO DE MODELOS BASE
# ============================================================================

print("\n" + "=" * 80)
print("ENTRENAMIENTO DE MODELOS BASE")
print("=" * 80)

# Crear modelos base
rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
gb_base = GradientBoostingClassifier(random_state=random_state)

# Definir malla de hiperparámetros para GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 7, 10, None],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [2, 5, 10]
}

print("\nMalla de hiperparámetros a explorar:")
print(param_grid)

# GridSearchCV para Random Forest
print("\n[1/2] Entrenando Random Forest con GridSearchCV...")
rf_grid = GridSearchCV(
    rf_base,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
print(f"Mejores parámetros (RF): {rf_grid.best_params_}")
print(f"Mejor score en CV (RF): {rf_grid.best_score_:.4f}")

# GridSearchCV para Gradient Boosting
print("\n[2/2] Entrenando Gradient Boosting con GridSearchCV...")
gb_grid = GridSearchCV(
    gb_base,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(X_train, y_train)
print(f"Mejores parámetros (GB): {gb_grid.best_params_}")
print(f"Mejor score en CV (GB): {gb_grid.best_score_:.4f}")

# ============================================================================
# 4. EVALUACIÓN EN CONJUNTO DE VALIDACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("EVALUACIÓN EN CONJUNTO DE VALIDACIÓN")
print("=" * 80)

models = {'Random Forest': rf_grid, 'Gradient Boosting': gb_grid}
val_results = {}

for model_name, model in models.items():
    y_val_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    except:
        roc_auc = None
    
    val_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# 5. EVALUACIÓN EN CONJUNTO DE PRUEBA
# ============================================================================

print("\n" + "=" * 80)
print("EVALUACIÓN EN CONJUNTO DE PRUEBA")
print("=" * 80)

test_results = {}

for model_name, model in models.items():
    y_test_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except:
        roc_auc = None
    
    test_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_test_pred
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n{model_name} - Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    print(f"\n{model_name} - Reporte de Clasificación:")
    print(classification_report(y_test, y_test_pred))

# ============================================================================
# 6. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================

print("\n" + "=" * 80)
print("IMPORTANCIA DE CARACTERÍSTICAS")
print("=" * 80)

# Importancia en Random Forest
rf_model = rf_grid.best_estimator_
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 características más importantes (Random Forest):")
print(feature_importance_rf.head(10))

# Importancia en Gradient Boosting
gb_model = gb_grid.best_estimator_
feature_importance_gb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 características más importantes (Gradient Boosting):")
print(feature_importance_gb.head(10))

# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================

print("\n" + "=" * 80)
print("GENERANDO VISUALIZACIONES")
print("=" * 80)

# Crear directorio de outputs si no existe
output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 7.1 Importancia de características - Random Forest
fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance_rf.head(15)
ax.barh(range(len(top_features)), top_features['importance'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Importancia')
ax.set_title('Top 15 Características - Random Forest')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: feature_importance_rf.png")

# 7.2 Importancia de características - Gradient Boosting
fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance_gb.head(15)
ax.barh(range(len(top_features)), top_features['importance'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Importancia')
ax.set_title('Top 15 Características - Gradient Boosting')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_gb.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: feature_importance_gb.png")

# 7.3 Matrices de Confusión para Entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_train)
    cm = confusion_matrix(y_train, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                xticklabels=['Bajo', 'Alto'], yticklabels=['Bajo', 'Alto'],
                cbar_kws={'label': 'Cantidad'})
    axes[idx].set_title(f'Matriz de Confusión - {model_name}\n(Conjunto de ENTRENAMIENTO)', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Valor Real')
    axes[idx].set_xlabel('Predicción')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_train.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: confusion_matrix_train.png")

# 7.4 Matrices de Confusión para Validación
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[idx],
                xticklabels=['Bajo', 'Alto'], yticklabels=['Bajo', 'Alto'],
                cbar_kws={'label': 'Cantidad'})
    axes[idx].set_title(f'Matriz de Confusión - {model_name}\n(Conjunto de VALIDACIÓN)', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Valor Real')
    axes[idx].set_xlabel('Predicción')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_val.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: confusion_matrix_val.png")

# 7.5 Matrices de Confusión para Prueba
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Bajo', 'Alto'], yticklabels=['Bajo', 'Alto'],
                cbar_kws={'label': 'Cantidad'})
    axes[idx].set_title(f'Matriz de Confusión - {model_name}\n(Conjunto de PRUEBA)', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Valor Real')
    axes[idx].set_xlabel('Predicción')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_test.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: confusion_matrix_test.png")

# 7.6 Comparación de métricas
fig, ax = plt.subplots(figsize=(10, 6))

metrics_names = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(metrics_names))
width = 0.35

rf_scores = [test_results['Random Forest'][m] for m in metrics_names]
gb_scores = [test_results['Gradient Boosting'][m] for m in metrics_names]

ax.bar(x - width/2, rf_scores, width, label='Random Forest', alpha=0.8)
ax.bar(x + width/2, gb_scores, width, label='Gradient Boosting', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Comparación de Métricas de Rendimiento (Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: metrics_comparison.png")

# 7.7 Curvas ROC
fig, ax = plt.subplots(figsize=(8, 6))

for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: roc_curves.png")

# 7.8 Visualización del Árbol de Decisión (del primer árbol del Random Forest)
rf_first_tree = rf_model.estimators_[0]

fig, ax = plt.subplots(figsize=(25, 15))
plot_tree(
    rf_first_tree,
    feature_names=X_train.columns.tolist(),
    class_names=['Bajo', 'Alto'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
ax.set_title('Árbol de Decisión #1 - Random Forest (Profundidad limitada para visualización)', 
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'decision_tree_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: decision_tree_rf.png")

# 7.9 Árbol de Decisión Simple para Mejor Visualización
# Crear un árbol de decisión único con profundidad limitada
dt_simple = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=random_state
)
dt_simple.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(16, 10))
plot_tree(
    dt_simple,
    feature_names=X_train.columns.tolist(),
    class_names=['Bajo', 'Alto'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
ax.set_title('Árbol de Decisión Simple (Profundidad = 4) - Para Interpretabilidad', 
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'decision_tree_simple.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: decision_tree_simple.png")

# Evaluar el árbol simple
y_test_pred_dt = dt_simple.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_test_pred_dt)
print(f"\nRendimiento del Árbol Simple en Test Set: Accuracy = {dt_accuracy:.4f}")

# ============================================================================
# 8. RESUMEN Y CONCLUSIONES
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN DE RESULTADOS")

print("=" * 80)

summary_df = pd.DataFrame(test_results).T
summary_df = summary_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
print("\nMétricas en Conjunto de Prueba:")
print(summary_df.to_string())

# Guardar resumen en CSV
summary_df.to_csv(output_dir / 'model_performance_summary.csv')
print(f"\n✓ Guardado: model_performance_summary.csv")

# Guardar importancia de características
feature_importance_rf.to_csv(output_dir / 'feature_importance_rf.csv', index=False)
feature_importance_gb.to_csv(output_dir / 'feature_importance_gb.csv', index=False)
print(f"✓ Guardado: feature_importance_rf.csv")
print(f"✓ Guardado: feature_importance_gb.csv")

print("\n" + "=" * 80)
print("SCRIPT COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"\nArchivos guardados en: {output_dir}")
