"""
Implementación CORREGIDA de Random Forest para clasificación del dataset FIRE UdeA
Correcciones principales:
1. Manejo de desbalance de clases (causa raíz del problema)
2. Validación temporal (no aleatoria) para evitar data leakage
3. Umbral de clasificación ajustado
4. Métricas apropiadas para clases desbalanceadas
5. Eliminación de columnas no-features (anio, unidad)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FireDatasetRandomForest:
    def __init__(self, output_dir='outputs_modelo'):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.label_encoder = None
        self.output_dir = output_dir
        self.optimal_threshold = 0.5
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. CARGA DE DATOS
    # ------------------------------------------------------------------
    def load_data(self, data_path='outputs'):
        """Carga y combina todos los archivos parquet del directorio."""
        print(f"Cargando datos desde: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encuentra el directorio: {data_path}")

        parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
        if not parquet_files:
            raise FileNotFoundError(f"No se encontraron archivos .parquet en {data_path}")

        print(f"Archivos encontrados: {parquet_files}")
        dataframes = [pd.read_parquet(os.path.join(data_path, f)) for f in parquet_files]
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"Dataset combinado: {self.data.shape}")

    # ------------------------------------------------------------------
    # 2. PREPROCESAMIENTO — CORRECCIÓN PRINCIPAL
    # ------------------------------------------------------------------
    def preprocess_data(self, target_column='label', temporal_split=True, test_years=2):
        """
        Preprocesa los datos.

        CORRECCIONES vs versión original:
        - Elimina columnas de identificación (anio, unidad) que no son features
        - Usa validación TEMPORAL en lugar de split aleatorio para evitar data leakage
        - Reporta distribución de clases para detectar desbalance

        Args:
            target_column  : Nombre de la columna objetivo
            temporal_split : Si True, usa los últimos test_years años como test
            test_years     : Número de años finales reservados para test
        """
        print("\n" + "="*60)
        print("PREPROCESAMIENTO")
        print("="*60)

        # --- Distribución del target ANTES de cualquier proceso ---
        print(f"\nDistribución de '{target_column}':")
        vc = self.data[target_column].value_counts()
        print(vc)
        minority_pct = vc.min() / vc.sum() * 100
        print(f"Clase minoritaria: {minority_pct:.1f}%")
        if minority_pct < 20:
            print("⚠️  DESBALANCE DETECTADO — se aplicará class_weight='balanced'")

        # --- Columnas a eliminar (identificadores, no features) ---
        id_cols = [c for c in ['anio', 'unidad'] if c in self.data.columns]
        anio_col = 'anio' if 'anio' in self.data.columns else None

        # --- Split temporal ---
        if temporal_split and anio_col:
            years = sorted(self.data[anio_col].unique())
            cutoff = years[-test_years]
            print(f"\nValidación temporal: train < {cutoff}  |  test >= {cutoff}")
            train_mask = self.data[anio_col] < cutoff
            test_mask  = self.data[anio_col] >= cutoff

            train_df = self.data[train_mask].copy()
            test_df  = self.data[test_mask].copy()

            X_train = train_df.drop(columns=[target_column] + id_cols)
            y_train = train_df[target_column]
            X_test  = test_df.drop(columns=[target_column] + id_cols)
            y_test  = test_df[target_column]

        else:
            # Fallback: split aleatorio estratificado
            print("\nUsando split aleatorio estratificado (80/20)")
            from sklearn.model_selection import train_test_split
            X = self.data.drop(columns=[target_column] + id_cols)
            y = self.data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        # --- Imputación (mediana para numéricas) ---
        self.feature_names = X_train.columns.tolist()
        for col in self.feature_names:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)

        self.X_train = X_train.values
        self.X_test  = X_test.values
        self.y_train = y_train.values
        self.y_test  = y_test.values

        print(f"\nTrain: {self.X_train.shape[0]} muestras  |  positivos: {self.y_train.sum()}")
        print(f"Test : {self.X_test.shape[0]}  muestras  |  positivos: {self.y_test.sum()}")

        # NOTA: Random Forest no requiere escalado — se elimina StandardScaler
        # (el escalado no mejora los árboles y puede enmascarar interpretabilidad)

    # ------------------------------------------------------------------
    # 3. ENTRENAMIENTO — CORRECCIÓN PRINCIPAL
    # ------------------------------------------------------------------
    def train_model(self):
        """
        Entrena Random Forest con correcciones para desbalance de clases.

        CORRECCIONES vs versión original:
        - class_weight='balanced': penaliza más los errores en clase minoritaria
        - Métricas de CV: f1 y roc_auc en lugar de accuracy (accuracy es engañosa con desbalance)
        - Grid search reducido y enfocado en parámetros relevantes
        """
        print("\n" + "="*60)
        print("ENTRENAMIENTO")
        print("="*60)

        # ── Parámetros base con manejo de desbalance ──────────────────
        base_params = dict(
            class_weight='balanced',   # ← CORRECCIÓN CLAVE
            random_state=42,
            n_jobs=-1
        )

        param_grid = {
            'n_estimators'    : [200, 400],
            'max_depth'       : [5, 10, None],
            'min_samples_leaf': [1, 3, 5],
        }

        from sklearn.model_selection import GridSearchCV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        rf = RandomForestClassifier(**base_params)
        grid = GridSearchCV(
            rf, param_grid,
            cv=cv,
            scoring='roc_auc',   # ← métrica apropiada para desbalance
            n_jobs=-1,
            verbose=1
        )
        grid.fit(self.X_train, self.y_train)

        self.model = grid.best_estimator_
        print(f"\nMejores parámetros : {grid.best_params_}")
        print(f"Mejor ROC-AUC (CV) : {grid.best_score_:.4f}")

        # Validación cruzada con múltiples métricas
        for metric in ['roc_auc', 'f1', 'average_precision']:
            scores = cross_val_score(self.model, self.X_train, self.y_train,
                                     cv=cv, scoring=metric)
            print(f"CV {metric:20s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # ------------------------------------------------------------------
    # 4. UMBRAL ÓPTIMO — CORRECCIÓN PRINCIPAL
    # ------------------------------------------------------------------
    def find_optimal_threshold(self):
        """
        Calcula el umbral de clasificación óptimo según F1.

        CORRECCIÓN: El umbral por defecto (0.5) favorece a la clase mayoritaria.
        Bajarlo aumenta la sensibilidad hacia la clase positiva (tensión de caja).
        """
        y_proba = self.model.predict_proba(self.X_train)[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_train, y_proba)

        f1_scores = np.where(
            (precision + recall) == 0, 0,
            2 * precision * recall / (precision + recall)
        )
        best_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        print(f"\nUmbral óptimo (F1 máximo en train): {self.optimal_threshold:.3f}")

    # ------------------------------------------------------------------
    # 5. EVALUACIÓN
    # ------------------------------------------------------------------
    def evaluate_model(self):
        """Evalúa con métricas apropiadas para clases desbalanceadas."""
        print("\n" + "="*60)
        print("EVALUACIÓN")
        print("="*60)

        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Predicciones con umbral estándar y umbral óptimo
        y_pred_05  = (y_proba >= 0.5).astype(int)
        y_pred_opt = (y_proba >= self.optimal_threshold).astype(int)

        for label, y_pred in [("Umbral 0.50", y_pred_05), (f"Umbral óptimo ({self.optimal_threshold:.2f})", y_pred_opt)]:
            print(f"\n--- {label} ---")
            print(classification_report(self.y_test, y_pred,
                                        target_names=['Sin tensión (0)', 'Tensión (1)'],
                                        zero_division=0))

        auc  = roc_auc_score(self.y_test, y_proba)
        ap   = average_precision_score(self.y_test, y_proba)
        print(f"ROC-AUC            : {auc:.4f}")
        print(f"Average Precision  : {ap:.4f}")

        # Guardar reporte
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"ROC-AUC: {auc:.4f}\nAverage Precision: {ap:.4f}\n\n")
            f.write(f"Umbral óptimo: {self.optimal_threshold:.3f}\n\n")
            f.write(classification_report(self.y_test, y_pred_opt,
                                          target_names=['Sin tensión (0)', 'Tensión (1)'],
                                          zero_division=0))

        return y_pred_opt, y_proba

    # ------------------------------------------------------------------
    # 6. GRÁFICOS
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_pred):
        fig, ax = plt.subplots(figsize=(7, 5))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Sin tensión', 'Tensión'],
                    yticklabels=['Sin tensión', 'Tensión'], ax=ax)
        ax.set_title(f'Matriz de Confusión (umbral={self.optimal_threshold:.2f})')
        ax.set_ylabel('Etiqueta Real')
        ax.set_xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=150)
        plt.show()

    def plot_roc_and_pr(self, y_proba):
        """ROC y Precision-Recall juntos — ambos importantes con desbalance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)
        axes[0].plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].set(title='Curva ROC', xlabel='FPR', ylabel='TPR')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Precision-Recall (más informativa con desbalance)
        prec, rec, _ = precision_recall_curve(self.y_test, y_proba)
        ap = average_precision_score(self.y_test, y_proba)
        axes[1].plot(rec, prec, lw=2, label=f'AP = {ap:.3f}')
        baseline = self.y_test.mean()
        axes[1].axhline(baseline, linestyle='--', color='gray', label=f'Baseline ({baseline:.2f})')
        axes[1].set(title='Curva Precision-Recall', xlabel='Recall', ylabel='Precision')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle('Métricas de Clasificación — Tensión de Caja', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_pr_curves.png'), dpi=150)
        plt.show()

    def plot_feature_importance(self):
        fi = pd.DataFrame({
            'feature'   : self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        top = fi.head(15)
        sns.barplot(data=top, x='importance', y='feature', palette='viridis', ax=ax)
        ax.set_title('Top 15 — Importancia de Features (Random Forest)')
        ax.set_xlabel('Importancia (Gini)')
        ax.set_ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=150)
        plt.show()

        fi.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        return fi

    def plot_score_distribution(self, y_proba):
        """Distribución de probabilidades predichas por clase — detecta si el modelo discrimina."""
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, name, color in [(0, 'Sin tensión (0)', '#2196F3'), (1, 'Tensión (1)', '#F44336')]:
            mask = self.y_test == label
            ax.hist(y_proba[mask], bins=25, alpha=0.6, label=name, color=color, density=True)
        ax.axvline(self.optimal_threshold, color='black', linestyle='--', label=f'Umbral ({self.optimal_threshold:.2f})')
        ax.set(title='Distribución de Probabilidad Predicha por Clase',
               xlabel='P(tensión de caja)', ylabel='Densidad')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'), dpi=150)
        plt.show()

    # ------------------------------------------------------------------
    # 7. PIPELINE COMPLETO
    # ------------------------------------------------------------------
    def run_full_pipeline(self, data_path='outputs', target_column='label',
                          temporal_split=True, test_years=2):
        print("="*60)
        print("PIPELINE RANDOM FOREST — TENSIÓN DE CAJA")
        print("="*60)

        self.load_data(data_path)
        self.preprocess_data(target_column, temporal_split, test_years)
        self.train_model()
        self.find_optimal_threshold()
        y_pred, y_proba = self.evaluate_model()

        self.plot_confusion_matrix(y_pred)
        self.plot_roc_and_pr(y_proba)
        fi = self.plot_feature_importance()
        self.plot_score_distribution(y_proba)

        print(f"\nResultados en: {self.output_dir}/")
        return self.model, fi


# ──────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────
def main():
    fire_model = FireDatasetRandomForest(output_dir='outputs_modelo')
    model, fi = fire_model.run_full_pipeline(
        data_path='outputs',
        target_column='label',
        temporal_split=True,   # Cambiar a False si no hay columna 'anio'
        test_years=2
    )


if __name__ == "__main__":
    main()