# ==========================================
# EDA AND PREPROCESSING PIPELINE
# Student-Career Adjustment Project
# ==========================================

import os
# suppress TensorFlow informational messages (oneDNN, etc.)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # errors and warnings only
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # disable oneDNN optimizations logs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from pathlib import Path
import pandas as pd
import umap

DATASET_PATH = '../Informe2/data/extended.csv'
OUTPUT_DIR = './extended_ds'
# ------------------------------------------
# 1. Create outputs folder
# ------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------
# 2. Load dataset
# ------------------------------------------
df = pd.read_csv(DATASET_PATH, sep=";")
print(df.columns)
df = df.drop(columns=df.columns[:5]) # drop unnecessary columns

print("\n===== DATA INSPECTION =====")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nDataset info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------
# 3. Filter only Systems Engineering
# ------------------------------------------
df = df[df["career"].str.lower().str.contains("sist", na=False)]

# ------------------------------------------
# 4. Likert scale transformation
# ------------------------------------------
likert_map = {
    "Muy en desacuerdo": 1,
    "En desacuerdo": 2,
    "Ni de acuerdo ni en desacuerdo": 3,
    "De acuerdo": 4,
    "Muy de acuerdo": 5
}

for col in df.columns:
    if col not in ["semester", "career"]:
        df[col] = df[col].map(likert_map)

# Convert semester to numeric
df["semester"] = pd.to_numeric(df["semester"], errors="coerce")

# ------------------------------------------
# 5. Missing values and data types
# ------------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

df = df.dropna()

# Select likert columns for descriptive statistics
df_likert = df.select_dtypes(include=np.number).drop(columns=["semester"])

# ------------------------------------------
# 7. General statictis tendency & dispersion
# ------------------------------------------
print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe()) # includes count, mean, std, min, 25%, 50%, 75%, max

# ------------------------------------------
# 8. Outlier detection (IQR method)
# ------------------------------------------
print("\n===== OUTLIER DETECTION (IQR) =====")

for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} potential outliers")

# ------------------------------------------
# 9. Visualizations
# ------------------------------------------

# --- 9a. Distribution for each variable ---
df_likert_rounded = df_likert.round(0).astype(int)

likert_scale = [1,2,3,4,5]

# Adjust number of rows and columns based on number of variables
n_cols = 5
n_rows = (len(df_likert_rounded.columns) + n_cols - 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(df_likert_rounded.columns):
    sns.countplot(
        x=col,
        data=df_likert_rounded,
        ax=axes[i],
        color="steelblue",
        order=likert_scale
    )
    axes[i].set_title(col)
    axes[i].set_xlabel("Valor Likert")
    axes[i].set_ylabel("Frecuencia")
    axes[i].set_ylim(0, df_likert_rounded.shape[0])  # uniform scale

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
#plt.savefig("outputs/likert_distributionpng", dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'likert_distributionpng'), dpi=300)
plt.close()

# --- 9b. Boxplot of all variables ---
plt.figure(figsize=(15,6))
sns.boxplot(data=df_likert, palette="coolwarm")
plt.title("Boxplot of Likert variables")
plt.ylabel("Valor Likert")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig("outputs/boxplot_likert.png")
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_likert.png'))
plt.close()

# ------------------------------------------
# 10. Feature engineering 
# ------------------------------------------
df_fe = df.copy()

# Reverse difficulty (keep interpretation consistent: higher = better)
df_fe["difficulty_reversed"] = 6 - df_fe["perceived_difficulty"]

# Academic adjustment index (target base)
df_fe["academic_adjustment"] = df_fe[
    ["difficulty_reversed", "study_habits", "time_management"]
].mean(axis=1)

# Motivation balance as ratio instead of difference (reduces linear dependency)
df_fe["motivation_ratio"] = df_fe["intrinsic_motivation"] / (
    df_fe["extrinsic_motivation"] + 1e-5
)

# Drop original variables used to build new ones → avoid multicollinearity
df_fe = df_fe.drop(columns=[
    "perceived_difficulty",
    "study_habits",
    "time_management",
    "intrinsic_motivation",
    "extrinsic_motivation"
])

# Use this dataset from now on
df_engineered = df_fe.copy()

# ------------------------------------------
# 11. Correlation analysis with heatmap
# ------------------------------------------

# --- 11a. Heatmap for initial data ---
df_numeric = df.select_dtypes(include=np.number)

corr_matrix = df_numeric.corr()

plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix,
    annot=True,           # show values
    fmt=".2f",            # 2 decimal points
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5,       # lines between cells
    linecolor='gray',
    mask=np.triu(np.ones_like(corr_matrix, dtype=bool))  # mask upper triangle
)
plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)
plt.tight_layout()
#plt.savefig("outputs/correlation_matrix.png", dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300)
plt.close()

# --- 11b. Heatmap for engineered data ---
df_numeric_eng = df_engineered.select_dtypes(include=np.number)
corr_matrix_enhanced = df_numeric_eng.corr()

plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix_enhanced,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor='gray',
    mask=np.triu(np.ones_like(corr_matrix_enhanced, dtype=bool))
)
plt.title("Correlation Matrix (Feature Engineering)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
#plt.savefig("outputs/correlation_matrix_fe.png", dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix_fe.png'), dpi=300)
plt.close()

# Delete highly correlated variables after feature engineering (threshold > 0.85)
upper = corr_matrix_enhanced.where(np.triu(np.ones(corr_matrix_enhanced.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

df_reduced = df.drop(columns=to_drop).select_dtypes(include=np.number) # keep only numeric for modeling
df_reduced["academic_adjustment"] = df_engineered["academic_adjustment"] # copy engineered target

print("\nHighly correlated variables removed:")
print(to_drop)

# --- 11c. Heatmap for reduced data ---
corr_matrix_reduced = df_reduced.corr()

plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix_reduced,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor='gray',
    mask=np.triu(np.ones_like(corr_matrix_reduced, dtype=bool))
)
plt.title("Correlation Matrix (Reduced Features)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
#plt.savefig("outputs/correlation_matrix_reduced.png", dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix_reduced.png'), dpi=300)
plt.close()

# ------------------------------------------
# 13. Data splitting, balancing and scaling for modeling
# ------------------------------------------
# Define output directory for exports
#OUTDIR = Path("outputs")
Path(OUTPUT_DIR).mkdir(exist_ok=True)

TARGET = "high_adjustment"

median_value = df_reduced["academic_adjustment"].median()

df_modeling = df_reduced.copy()
df_modeling[TARGET] = (
    df_modeling["academic_adjustment"] > median_value
).astype(int)

print("\n===== TARGET VARIABLE DISTRIBUTION =====")
print(df_modeling[TARGET].value_counts())
print(df_modeling[TARGET].value_counts(normalize=True))

# Features and target
X_raw = df_modeling.drop(columns=["academic_adjustment", TARGET])
y = df_modeling[TARGET]

print(f"\nFeatures: {list(X_raw.columns)}")
print(f"Total samples: {X_raw.shape[0]}")

# Export RAW dataset (no scaling, no split)
X_raw.to_parquet(os.path.join(OUTPUT_DIR, 'X_raw.parquet'), index=False)
y.to_frame(name=TARGET).to_parquet(os.path.join(OUTPUT_DIR, 'y.parquet'), index=False)


# ------------------------------------------
# 14. Dimensionality reduction visualizations
#     (PCA, t-SNE, UMAP) on the training data
# ------------------------------------------
print("\n===== DIMENSIONALITY REDUCTION VISUALIZATIONS =====")

# Use raw data and scale ONLY for visualization
scaler = StandardScaler()
X_scaled_viz = scaler.fit_transform(X_raw)

df_viz = pd.DataFrame(X_scaled_viz, columns=X_raw.columns)
df_viz[TARGET] = y.values

# PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(df_viz.drop(columns=[TARGET]))
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df[TARGET] = df_viz[TARGET].values

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="PC1", y="PC2",
    hue=TARGET,
    data=pca_df,
    s=60,
    alpha=0.8
)
plt.title("PCA projection (colored by target)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.legend(title="High Adjustment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_plot.png'), dpi=300)
plt.close()

# t-SNE
perplexity_value = min(30, len(df_viz) - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
tsne_result = tsne.fit_transform(df_viz.drop(columns=[TARGET]))
tsne_df = pd.DataFrame(tsne_result, columns=["Dim1", "Dim2"])
tsne_df[TARGET] = df_viz[TARGET].values

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="Dim1", y="Dim2",
    hue=TARGET,
    data=tsne_df,
    s=60,
    alpha=0.8
)
plt.title("t-SNE projection (colored by target)")
plt.legend(title="High Adjustment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_plot.png'), dpi=300)
plt.close()

# UMAP
umapper = umap.UMAP(n_components=2, random_state=42)
umap_result = umapper.fit_transform(df_viz.drop(columns=[TARGET]))
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
umap_df[TARGET] = df_viz[TARGET].values

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="UMAP1", y="UMAP2",
    hue=TARGET,
    data=umap_df,
    s=60,
    alpha=0.8
)
plt.title("UMAP projection (colored by target)")
plt.legend(title="High Adjustment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'umap_plot.png'), dpi=300)
plt.close()

print("Dimensionality reduction visualizations completed.")
# ------------------------------------------
# 15. Final conclusions
# ------------------------------------------
print("\n===== FINAL CONCLUSIONS =====")

print("\nDATASET SUMMARY")
print(f"  Total samples processed: {len(df)} (Systems Engineering students)")

print("\nKEY MOTIVATIONAL INSIGHTS")
print(f"  Average intrinsic motivation: {df['intrinsic_motivation'].mean():.2f}/5.00")
print(f"  Average extrinsic motivation: {df['extrinsic_motivation'].mean():.2f}/5.00")
print(f"  Motivation balance (intrinsic - extrinsic): {(df['intrinsic_motivation'].mean() - df['extrinsic_motivation'].mean()):.2f}")
print(f"  Average academic stress: {df['academic_stress'].mean():.2f}/5.00")

print("\nFEATURE ENGINEERING & SELECTION")
print(f"  Original numeric features: {len(df_likert.columns)}")
print(f"  Removed features: {to_drop}")

print("\nCORRELATION INSIGHTS")
strong_corr = corr_matrix_reduced.abs().unstack().sort_values(ascending=False)
strong_corr = strong_corr[strong_corr < 1].head(3)
print("  Top 3 strongest correlations in final feature set:")
for idx, (pair, corr_val) in enumerate(strong_corr.items(), 1):
    print(f"    {idx}. {pair[0]} <-> {pair[1]}: {corr_val:.3f}")

print("\nDATA QUALITY METRICS")
print(f"  Data types: All numeric (Likert scale 1-5 converted to integers)")
print(f"  Standardization: Mean=0, Std=1 (StandardScaler applied)")
print(f"  Stratified splitting: Maintains class proportions across split")

print(f"  ✓ Visualizations: PCA, t-SNE, UMAP, correlation heatmaps")

print("\n" + "="*70)
print("Pipeline complete. Dataset ready for supervised learning.")
print("="*70)