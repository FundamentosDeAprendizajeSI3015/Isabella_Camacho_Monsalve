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
# ------------------------------------------
# 1. Create outputs folder
# ------------------------------------------
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------
# 2. Load dataset
# ------------------------------------------
df = pd.read_csv("survey.csv", sep=";")
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

# ------------------------------------------
# 6. Statistics for Likert variables
# ------------------------------------------
# Select likert columns for descriptive statistics
df_likert = df.select_dtypes(include=np.number).drop(columns=["semester"])

desc_stats = pd.DataFrame(index=df_likert.columns)
desc_stats['mean'] = df_likert.mean()
desc_stats['median'] = df_likert.median()
desc_stats['mode'] = df_likert.mode().iloc[0]
desc_stats['std'] = df_likert.std()
desc_stats['min'] = df_likert.min()
desc_stats['max'] = df_likert.max()
desc_stats['range'] = desc_stats['max'] - desc_stats['min']
desc_stats['25%'] = df_likert.quantile(0.25)
desc_stats['75%'] = df_likert.quantile(0.75)
desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']

# Save statistics to csv
desc_stats.to_csv("outputs/descriptive_statistics.csv")
print("\nDescriptive statistics saved to outputs/descriptive_statistics.csv")

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
plt.savefig("outputs/likert_distributionpng", dpi=300)
plt.close()

# --- 9b. Boxplot of all variables ---
plt.figure(figsize=(15,6))
sns.boxplot(data=df_likert, palette="coolwarm")
plt.title("Boxplot of Likert variables")
plt.ylabel("Valor Likert")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/boxplot_likert.png")
plt.close()

# ------------------------------------------
# 10. Feature engineering 
# ------------------------------------------

# Motivation index
df["motivation_net"] = df["intrinsic_motivation"] - df["extrinsic_motivation"]

# Reverse difficulty (since high difficulty reduces adjustment)
df["difficulty_reversed"] = 6 - df["perceived_difficulty"]

# Academic adjustment index
df["academic_adjustment"] = df[
    ["difficulty_reversed", "study_habits", "time_management"]
].mean(axis=1)

# Make a copy of the dataset with engineered features to drop the used columns 
df_engineered = df.copy()
df_engineered = df_engineered.drop(columns=[
    "intrinsic_motivation", "extrinsic_motivation", "perceived_difficulty", "difficulty_reversed", "study_habits", "time_management"
])

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
plt.savefig("outputs/correlation_matrix.png", dpi=300)
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
plt.savefig("outputs/correlation_matrix_fe.png", dpi=300)
plt.close()

# Delete highly correlated variables after feature engineering (threshold > 0.85)
upper = corr_matrix_enhanced.where(np.triu(np.ones(corr_matrix_enhanced.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

df_reduced = df.drop(columns=to_drop).select_dtypes(include=np.number) # keep only numeric for modeling

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
plt.savefig("outputs/correlation_matrix_reduced.png", dpi=300)
plt.close()

# ------------------------------------------
# 13. Data splitting, balancing and scaling for modeling
# ------------------------------------------
# Define output directory for exports
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

# Create label using academic_adjustment BEFORE dropping it from features
TARGET = "high_adjustment"
median_value = df_reduced["academic_adjustment"].median()

df_modeling = df_reduced.copy()
df_modeling[TARGET] = (
    df_modeling["academic_adjustment"] > median_value
).astype(int)

print("\n===== TARGET VARIABLE DISTRIBUTION =====")
print(df_modeling[TARGET].value_counts())
print(f"Class balance: {df_modeling[TARGET].value_counts(normalize=True)}")

# Define features (drop academic_adjustment and target variable)
X = df_modeling.drop(columns=["academic_adjustment", TARGET])
y = df_modeling[TARGET]

NUM_COLS = X.select_dtypes(include="number").columns.tolist()

print(f"\nFeatures: {list(X.columns)}")
print(f"Numeric features: {NUM_COLS}")

# Split into train (70%), val (15%) and test (15%) with stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print(f"\n===== DATA SPLIT =====")
print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Balance the training set by downsampling the majority class to match the minority class size
train_df = pd.concat([X_train, y_train], axis=1)

class_0 = train_df[train_df[TARGET] == 0]
class_1 = train_df[train_df[TARGET] == 1]

min_class = min(len(class_0), len(class_1))

print(f"\n===== CLASS BALANCE =====")
print(f"Before balancing - Class 0: {len(class_0)}, Class 1: {len(class_1)}")

class_0_bal = resample(
    class_0,
    replace=False,
    n_samples=min_class,
    random_state=42
)

class_1_bal = resample(
    class_1,
    replace=False,
    n_samples=min_class,
    random_state=42
)

train_balanced = pd.concat([class_0_bal, class_1_bal])

X_train = train_balanced.drop(columns=[TARGET])
y_train = train_balanced[TARGET]

print(f"After balancing - Class 0: {len(class_0_bal)}, Class 1: {len(class_1_bal)}")

# Scale numeric features with StandardScaler (mean=0, std=1)
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_val_scaled[NUM_COLS]   = scaler.transform(X_val[NUM_COLS])
X_test_scaled[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

# Export
X_train_scaled.to_parquet(OUTDIR / "X_train.parquet", index=False)
X_val_scaled.to_parquet(OUTDIR / "X_val.parquet", index=False)
X_test_scaled.to_parquet(OUTDIR / "X_test.parquet", index=False)

y_train.to_frame(name=TARGET).to_parquet(OUTDIR / "y_train.parquet", index=False)
y_val.to_frame(name=TARGET).to_parquet(OUTDIR / "y_val.parquet", index=False)
y_test.to_frame(name=TARGET).to_parquet(OUTDIR / "y_test.parquet", index=False)

print("\n Datasets exported correclty.")


# ------------------------------------------
# 14. Dimensionality reduction visualizations
#     (PCA, t-SNE, UMAP) on the training data
# ------------------------------------------
print("\n===== DIMENSIONALITY REDUCTION VISUALIZATIONS =====")

# Use the combined training+validation+test scaled data for visualization
df_all_scaled = pd.concat([X_train_scaled, X_val_scaled, X_test_scaled], ignore_index=True)

# PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(df_all_scaled)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

plt.figure(figsize=(8,6))
sns.scatterplot(x="PC1", y="PC2", data=pca_df, s=50, alpha=0.7)
plt.title("PCA projection (2 components)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.tight_layout()
plt.savefig("outputs/pca_plot.png", dpi=300)
plt.close()

# t-SNE
perplexity_value = min(30, len(df_all_scaled) - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
tsne_result = tsne.fit_transform(df_all_scaled)
tsne_df = pd.DataFrame(tsne_result, columns=["Dim1", "Dim2"])

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Dim1",
    y="Dim2",
    data=tsne_df,
    s=50,
    alpha=0.7
)
plt.title("t-SNE projection")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("outputs/tsne_plot.png", dpi=300)
plt.close()

# UMAP
umapper = umap.UMAP(n_components=2, random_state=42)
umap_result = umapper.fit_transform(df_all_scaled)
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])

plt.figure(figsize=(8,6))
sns.scatterplot(x="UMAP1", y="UMAP2", data=umap_df, s=50, alpha=0.7)
plt.title("UMAP projection")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig("outputs/umap_plot.png", dpi=300)
plt.close()

print("Dimensionality reduction visualizations completed.")

# ------------------------------------------
# 15. Final conclusions
# ------------------------------------------
print("\n===== FINAL CONCLUSIONS =====")

print("\nDATASET SUMMARY")
print(f"  Total samples processed: {len(df)} (Systems Engineering students)")
print(f"  Final training set: {X_train_scaled.shape[0]} samples | Validation: {X_val_scaled.shape[0]} samples | Test: {X_test_scaled.shape[0]} samples")
print(f"  Total features for modeling: {X_train_scaled.shape[1]}")

print("\nTARGET VARIABLE: Student-Career Adjustment Level")
print(f"  Definition: Based on academic_adjustment index (difficulty reversed + study habits + time management)")
print(f"  Threshold: Median value = {median_value:.2f}")
print(f"  Training set distribution (balanced):")
print(f"    - High adjustment (1): {(y_train == 1).sum()} students ({(y_train == 1).mean():.1%})")
print(f"    - Low adjustment (0): {(y_train == 0).sum()} students ({(y_train == 0).mean():.1%})")
print(f"  Original class balance: {df_modeling['high_adjustment'].value_counts(normalize=True).to_dict()}")

print("\nKEY MOTIVATIONAL INSIGHTS")
print(f"  Average intrinsic motivation: {df['intrinsic_motivation'].mean():.2f}/5.00")
print(f"  Average extrinsic motivation: {df['extrinsic_motivation'].mean():.2f}/5.00")
print(f"  Motivation balance (intrinsic - extrinsic): {(df['intrinsic_motivation'].mean() - df['extrinsic_motivation'].mean()):.2f}")
print(f"  Average academic stress: {df['academic_stress'].mean():.2f}/5.00")

print("\nFEATURE ENGINEERING & SELECTION")
print(f"  Original numeric features: {len(df_likert.columns)}")
print(f"  After removing highly correlated variables (r > 0.85): {len(X.columns)}")
print(f"  Removed features: {to_drop}")
print(f"  Final model features: {NUM_COLS}")

print("\nCORRELATION INSIGHTS")
strong_corr = corr_matrix_reduced.abs().unstack().sort_values(ascending=False)
strong_corr = strong_corr[strong_corr < 1].head(3)
print("  Top 3 strongest correlations in final feature set:")
for idx, (pair, corr_val) in enumerate(strong_corr.items(), 1):
    print(f"    {idx}. {pair[0]} <-> {pair[1]}: {corr_val:.3f}")

print("\nDATA QUALITY METRICS")
print(f"  Missing values after cleaning: {X_train.isnull().sum().sum()}")
print(f"  Data types: All numeric (Likert scale 1-5 converted to integers)")
print(f"  Standardization: Mean=0, Std=1 (StandardScaler applied)")
print(f"  Stratified splitting: Maintains class proportions across split")

print("\nDELIVERABLES")
print(f"  ✓ Training data: X_train.parquet ({X_train_scaled.shape[0]}x{X_train_scaled.shape[1]})")
print(f"  ✓ Validation data: X_val.parquet ({X_val_scaled.shape[0]}x{X_val_scaled.shape[1]})")
print(f"  ✓ Test data: X_test.parquet ({X_test_scaled.shape[0]}x{X_test_scaled.shape[1]})")
print(f"  ✓ Visualizations: PCA, t-SNE, UMAP, correlation heatmaps")
print(f"  ✓ Descriptive statistics: descriptive_statistics.csv")


print("\n" + "="*70)
print("Pipeline complete. Dataset ready for supervised learning.")
print("="*70)