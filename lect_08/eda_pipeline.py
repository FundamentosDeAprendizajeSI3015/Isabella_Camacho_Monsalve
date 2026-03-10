# ==========================================
# EDA AND PREPROCESSING PIPELINE
# FIRE UdeA Financial Dataset - Realistic Version
# ==========================================

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

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
import umap
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------
# 1. Create outputs folder
# ------------------------------------------
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------
# 2. Load dataset
# ------------------------------------------
df = pd.read_csv("datasets/dataset_sintetico_FIRE_UdeA_realista.csv")

print("\n===== DATA INSPECTION =====")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"\nColumns: {list(df.columns)}")
print("\nDataset info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------
# 3. Data type conversion and cleaning
# ------------------------------------------
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------------------------
# 4. Missing values and data types
# ------------------------------------------
print("\n===== MISSING VALUES =====")
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_summary)

print("\nHandling missing values...")
df = df.dropna(subset=['label'])

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col}: filled {df[col].isnull().sum()} missing values with median ({median_val:.2f})")

print(f"\nDataset after cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------------------------
# 5. Summary statistics
# ------------------------------------------
print("\n===== DESCRIPTIVE STATISTICS =====")

numeric_features = df.select_dtypes(include=np.number).drop(columns=['anio', 'label']).columns

desc_stats = pd.DataFrame()
desc_stats['mean'] = df[numeric_features].mean()
desc_stats['median'] = df[numeric_features].median()
desc_stats['std'] = df[numeric_features].std()
desc_stats['min'] = df[numeric_features].min()
desc_stats['25%'] = df[numeric_features].quantile(0.25)
desc_stats['75%'] = df[numeric_features].quantile(0.75)
desc_stats['max'] = df[numeric_features].max()
desc_stats['range'] = desc_stats['max'] - desc_stats['min']
desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']

desc_stats.to_csv("outputs/descriptive_statistics.csv")
print("\nDescriptive statistics saved to outputs/descriptive_statistics.csv")
print(desc_stats)

# ------------------------------------------
# 6. Dataset overview by unit and year
# ------------------------------------------
print("\n===== DATASET OVERVIEW BY UNIT AND YEAR =====")
print(f"Unique units: {df['unidad'].nunique()}")
print(f"Years: {sorted(df['anio'].unique())}")
print(f"\nTarget variable distribution (label):")
print(df['label'].value_counts())
print(f"Class balance: {(df['label'].value_counts() / len(df) * 100).round(2).to_dict()}")

# ------------------------------------------
# 7. Outlier detection (IQR method)
# ------------------------------------------
print("\n===== OUTLIER DETECTION (IQR) =====")

outlier_summary = {}
for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_summary[col] = len(outliers)
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} potential outliers")

print("\nOutliers are kept in the dataset for financial analysis (potential important events)")

# ------------------------------------------
# 8. TEMPORAL VISUALIZATIONS
# ------------------------------------------
print("\n===== GENERATING TEMPORAL VISUALIZATIONS =====")

# 8a. Time series of key financial metrics by unit
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Evolution of Key Financial Metrics Over Time by Unit", fontsize=16, y=1.00)

units = df['unidad'].unique()
colors = sns.color_palette("husl", len(units))

# 1. Revenue over time
ax = axes[0, 0]
for unit, color in zip(units, colors):
    unit_data = df[df['unidad'] == unit].sort_values('anio')
    ax.plot(unit_data['anio'], unit_data['ingresos_totales']/1e9, marker='o', label=unit, color=color, linewidth=2)
ax.set_xlabel('Year')
ax.set_ylabel('Total Revenue (Billions)')
ax.set_title('Ingresos Totales')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Personal expenses over time
ax = axes[0, 1]
for unit, color in zip(units, colors):
    unit_data = df[df['unidad'] == unit].sort_values('anio')
    ax.plot(unit_data['anio'], unit_data['gastos_personal']/1e9, marker='s', label=unit, color=color, linewidth=2)
ax.set_xlabel('Year')
ax.set_ylabel('Personal Expenses (Billions)')
ax.set_title('Gastos Personales')
ax.grid(True, alpha=0.3)

# 3. Liquidity over time
ax = axes[1, 0]
for unit, color in zip(units, colors):
    unit_data = df[df['unidad'] == unit].sort_values('anio')
    ax.plot(unit_data['anio'], unit_data['liquidez'], marker='^', label=unit, color=color, linewidth=2)
ax.set_xlabel('Year')
ax.set_ylabel('Liquidity Ratio')
ax.set_title('Liquidez')
ax.grid(True, alpha=0.3)

# 4. Indebtedness over time
ax = axes[1, 1]
for unit, color in zip(units, colors):
    unit_data = df[df['unidad'] == unit].sort_values('anio')
    ax.plot(unit_data['anio'], unit_data['endeudamiento'], marker='d', label=unit, color=color, linewidth=2)
ax.set_xlabel('Year')
ax.set_ylabel('Indebtedness Ratio')
ax.set_title('Endeudamiento')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/01_temporal_evolution_key_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

# 8b. Heatmap of revenue trends by unit over years
pivot_revenue = df.pivot_table(values='ingresos_totales', index='unidad', columns='anio', aggfunc='mean')/1e9

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot_revenue, annot=True, fmt='.0f', cmap='YlGn', ax=ax, cbar_kws={'label': 'Revenue (Billions)'})
ax.set_title('Average Revenue by Unit and Year (Heatmap)', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Unit')
plt.tight_layout()
plt.savefig("outputs/02_heatmap_revenue_by_unit_year.png", dpi=300, bbox_inches='tight')
plt.close()

# 8c. Liquidity heatmap
pivot_liquidity = df.pivot_table(values='liquidez', index='unidad', columns='anio', aggfunc='mean')

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot_liquidity, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, center=1.0, cbar_kws={'label': 'Liquidity Ratio'})
ax.set_title('Average Liquidity by Unit and Year (Heatmap)', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Unit')
plt.tight_layout()
plt.savefig("outputs/03_heatmap_liquidity_by_unit_year.png", dpi=300, bbox_inches='tight')
plt.close()

# 8d. Target variable distribution over time
fig, ax = plt.subplots(figsize=(12, 6))
label_by_year = df.groupby(['anio', 'label']).size().unstack(fill_value=0)
label_by_year.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_xlabel('Year')
ax.set_ylabel('Count')
ax.set_title('Distribution of Target Variable (Financial Health) Over Years', fontsize=12)
ax.set_xticklabels(label_by_year.index, rotation=0)
ax.legend(['Sustainable (0)', 'At Risk (1)'], title='Label')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("outputs/04_target_distribution_over_time.png", dpi=300)
plt.close()

print("Temporal visualizations completed")

# ------------------------------------------
# 9. Distribution visualizations
# ------------------------------------------
print("\n===== GENERATING DISTRIBUTION VISUALIZATIONS =====")

n_cols = 4
n_rows = (len(numeric_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(col)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)
    
    mean_val = df[col].mean()
    median_val = df[col].median()
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2e}')
    axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2e}')
    axes[i].legend(fontsize=8)

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("outputs/05_distributions_all_features.png", dpi=300, bbox_inches='tight')
plt.close()

# 9b. Boxplots of all numeric features by target variable
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    sns.boxplot(data=df, x='label', y=col, ax=axes[i], palette=['#FF6B6B', '#4ECDC4'])
    axes[i].set_title(f'{col} by Target Label')
    axes[i].set_xlabel('Label')
    axes[i].grid(True, alpha=0.3, axis='y')

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("outputs/06_boxplots_by_target.png", dpi=300, bbox_inches='tight')
plt.close()

print("Distribution visualizations completed")

# ------------------------------------------
# 10. Feature engineering (optional enhancements)
# ------------------------------------------
print("\n===== FEATURE ENGINEERING =====")

df_engineered = df.copy()

df_engineered['gastos_per_ingresos'] = df_engineered['gastos_personal'] / (df_engineered['ingresos_totales'] + 1e-6)
df_engineered['cfo_per_ingresos'] = df_engineered['cfo'] / (df_engineered['ingresos_totales'] + 1e-6)
df_engineered['dias_efectivo_ratio'] = df_engineered['dias_efectivo'] / 365.0

print("Created engineered features:")
print("  - gastos_per_ingresos: Personal Expenses / Total Revenue")
print("  - cfo_per_ingresos: Operating Cash Flow / Total Revenue")
print("  - dias_efectivo_ratio: Days of Cash / 365")

# ------------------------------------------
# 11. Correlation analysis with heatmap
# ------------------------------------------
print("\n===== CORRELATION ANALYSIS =====")

df_numeric = df.select_dtypes(include=np.number).drop(columns=['anio', 'label'])

corr_matrix = df_numeric.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor='gray',
    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
    annot_kws={'size': 8}
)
plt.title("Correlation Matrix - Financial Indicators", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/07_correlation_matrix_financial.png", dpi=300, bbox_inches='tight')
plt.close()

# Correlation of features with target variable
df_with_target = df_numeric.copy()
df_with_target['label'] = df['label']

corr_with_target = df_with_target.corr()['label'].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['green' if x > 0 else 'red' for x in corr_with_target.values[:-1]]
ax.barh(corr_with_target.index[:-1], corr_with_target.values[:-1], color=colors, alpha=0.7)
ax.set_xlabel('Correlation with Target (label)')
ax.set_title('Feature Correlation with Target Variable', fontsize=14)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("outputs/08_correlation_with_target.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nStrong correlations with target variable:")
strong_target_corr = corr_with_target[:-1].abs().sort_values(ascending=False)
for i, (var, corr) in enumerate(strong_target_corr.head(10).items(), 1):
    print(f"  {i}. {var}: {corr_with_target[var]:.4f}")

# Engineered features correlation
df_eng_numeric = df_engineered.select_dtypes(include=np.number).drop(columns=['anio', 'label'])
corr_engineered = df_eng_numeric.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_engineered,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor='gray',
    mask=np.triu(np.ones_like(corr_engineered, dtype=bool)),
    annot_kws={'size': 7}
)
plt.title("Correlation Matrix - Including Engineered Features", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/09_correlation_matrix_engineered.png", dpi=300, bbox_inches='tight')
plt.close()

# Remove highly correlated variables
print("\n===== REMOVING HIGHLY CORRELATED FEATURES =====")
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
highly_corr_pairs = []
to_drop = set()

for column in upper.columns:
    high_corr = upper[column][upper[column].abs() > 0.85]
    for idx in high_corr.index:
        if idx != column:
            highly_corr_pairs.append((idx, column, upper[column][idx]))
            to_drop.add(column)

to_drop = list(to_drop)
print(f"Highly correlated pairs (r > 0.85):")
for var1, var2, corr in highly_corr_pairs:
    print(f"  {var1} <-> {var2}: {corr:.4f}")
print(f"\nVariables to potentially remove: {to_drop}")

df_reduced = df_engineered.drop(columns=to_drop) if to_drop else df_engineered.copy()
df_reduced = df_reduced.select_dtypes(include=np.number)

print("Correlation analysis completed")

# ------------------------------------------
# 12. Data splitting and scaling
# ------------------------------------------
print("\n===== DATA PREPARATION FOR MODELING =====")

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

X = df_reduced.drop(columns=['label', 'anio'], errors='ignore')
y = df_reduced['label']

NUM_COLS = X.select_dtypes(include="number").columns.tolist()

print(f"\nFeature set shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Features: {list(X.columns)}")

print(f"\n===== TARGET VARIABLE BALANCE =====")
print(f"Class distribution:")
print(y.value_counts())
print(f"\nPercentages:")
print((y.value_counts(normalize=True) * 100).round(2))

if len(X) < 20:
    print("\nWarning: Small dataset size. Using 80% train, 20% test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    X_val, X_test, y_val, y_test = X_test, X_test, y_test, y_test
else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
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
print(f"Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} samples")

train_df = pd.concat([X_train, y_train], axis=1)
class_0 = train_df[train_df['label'] == 0]
class_1 = train_df[train_df['label'] == 1]

print(f"\n===== CLASS BALANCE IN TRAINING SET =====")
print(f"Before balancing:")
print(f"  Class 0: {len(class_0)} samples ({len(class_0)/len(train_df)*100:.1f}%)")
print(f"  Class 1: {len(class_1)} samples ({len(class_1)/len(train_df)*100:.1f}%)")

if len(class_0) > 0 and len(class_1) > 0:
    min_class = min(len(class_0), len(class_1))
    max_class = max(len(class_0), len(class_1))
    
    if max_class / min_class > 1.5:
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
        
        train_balanced = pd.concat([class_0_bal, class_1_bal]).sample(frac=1, random_state=42)
        X_train = train_balanced.drop(columns=['label'])
        y_train = train_balanced['label']
        
        print(f"After balancing (downsampling):")
        print(f"  Class 0: {len(class_0_bal)} samples")
        print(f"  Class 1: {len(class_1_bal)} samples")
    else:
        print(f"Dataset is relatively balanced (ratio: {max_class/min_class:.2f}). No resampling applied.")

print(f"\n===== SCALING FEATURES =====")
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_val_scaled[NUM_COLS] = scaler.transform(X_val[NUM_COLS])
X_test_scaled[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

print("Applied StandardScaler (mean=0, std=1)")

print(f"\n===== EXPORTING DATASETS =====")
X_train_scaled.to_parquet(OUTDIR / "X_train.parquet", index=False)
X_val_scaled.to_parquet(OUTDIR / "X_val.parquet", index=False)
X_test_scaled.to_parquet(OUTDIR / "X_test.parquet", index=False)

y_train.to_frame(name='label').to_parquet(OUTDIR / "y_train.parquet", index=False)
y_val.to_frame(name='label').to_parquet(OUTDIR / "y_val.parquet", index=False)
y_test.to_frame(name='label').to_parquet(OUTDIR / "y_test.parquet", index=False)

print("Datasets exported successfully")
print(f"  - X_train.parquet ({X_train_scaled.shape[0]}x{X_train_scaled.shape[1]})")
print(f"  - X_val.parquet ({X_val_scaled.shape[0]}x{X_val_scaled.shape[1]})")
print(f"  - X_test.parquet ({X_test_scaled.shape[0]}x{X_test_scaled.shape[1]})")

# ------------------------------------------
# 13. Dimensionality reduction visualizations
# ------------------------------------------
print("\n===== DIMENSIONALITY REDUCTION VISUALIZATIONS =====")

df_viz_train = X_train_scaled.copy()
df_viz_train['label'] = y_train.values

print("Computing PCA...")
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(df_viz_train.drop(columns=['label']))
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df['label'] = y_train.values

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#FF6B6B', '#4ECDC4']
for label in [0, 1]:
    mask = pca_df['label'] == label
    ax.scatter(pca_df[mask]['PC1'], pca_df[mask]['PC2'], 
              c=colors[label], label=f'Label {label}', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
ax.set_title("PCA Projection (2 components)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/10_pca_projection.png", dpi=300, bbox_inches='tight')
plt.close()

print("Computing t-SNE (this may take a moment)...")
perplexity_value = min(30, max(5, len(df_viz_train) - 1))
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=1000)
tsne_result = tsne.fit_transform(df_viz_train.drop(columns=['label']))
tsne_df = pd.DataFrame(tsne_result, columns=["Dim1", "Dim2"])
tsne_df['label'] = y_train.values

fig, ax = plt.subplots(figsize=(10, 8))
for label in [0, 1]:
    mask = tsne_df['label'] == label
    ax.scatter(tsne_df[mask]['Dim1'], tsne_df[mask]['Dim2'], 
              c=colors[label], label=f'Label {label}', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_title("t-SNE Projection")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/11_tsne_projection.png", dpi=300, bbox_inches='tight')
plt.close()

print("Computing UMAP (this may take a moment)...")
try:
    umapper = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(df_viz_train)-1), min_dist=0.1)
    umap_result = umapper.fit_transform(df_viz_train.drop(columns=['label']))
    umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
    umap_df['label'] = y_train.values

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in [0, 1]:
        mask = umap_df['label'] == label
        ax.scatter(umap_df[mask]['UMAP1'], umap_df[mask]['UMAP2'], 
                  c=colors[label], label=f'Label {label}', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/12_umap_projection.png", dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"  Warning: UMAP computation failed - {e}")

print("Dimensionality reduction visualizations completed")

# ------------------------------------------
# 14. Additional time series analysis by unit
# ------------------------------------------
print("\n===== UNIT-LEVEL TIME SERIES ANALYSIS =====")

unique_units = df['unidad'].unique()
n_units = len(unique_units)

fig, axes = plt.subplots(n_units, 1, figsize=(14, 4*n_units))
if n_units == 1:
    axes = [axes]

for idx, unit in enumerate(sorted(unique_units)):
    unit_data = df[df['unidad'] == unit].sort_values('anio')
    
    ax = axes[idx]
    ax2 = ax.twinx()
    
    line1 = ax.plot(unit_data['anio'], unit_data['ingresos_totales']/1e9, 
                    marker='o', color='green', linewidth=2, label='Revenue', alpha=0.7)
    line2 = ax.plot(unit_data['anio'], unit_data['gastos_personal']/1e9, 
                    marker='s', color='red', linewidth=2, label='Expenses', alpha=0.7)
    
    line3 = ax2.plot(unit_data['anio'], unit_data['liquidez'], 
                     marker='^', color='blue', linewidth=2, linestyle='--', label='Liquidity', alpha=0.7)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Revenue & Expenses (Billions)', color='black')
    ax2.set_ylabel('Liquidity Ratio', color='blue')
    ax.set_title(f'{unit}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=9)

plt.suptitle('Time Series Analysis by Academic Unit', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig("outputs/13_time_series_by_unit.png", dpi=300, bbox_inches='tight')
plt.close()

print("Unit-level time series analysis completed")

# ------------------------------------------
# 15. Final summary and conclusions
# ------------------------------------------
print("\n" + "="*70)
print("EDA AND PREPROCESSING PIPELINE - FINAL SUMMARY")
print("="*70)

print("\nDATASET SUMMARY")
print(f"  - Financial Health Indicators (FIRE UdeA)")
print(f"  - Time period: {df['anio'].min():.0f} - {df['anio'].max():.0f}")
print(f"  - Academic units: {df['unidad'].nunique()}")
print(f"  - Total observations: {len(df)}")
print(f"  - Final features: {X_train_scaled.shape[1]}")

print("\nTARGET VARIABLE: Financial Health Status")
print(f"  - Label=0: Sustainable financial situation")
print(f"  - Label=1: At-risk financial situation")
print(f"  - Class distribution:")
print(f"    Class 0: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"    Class 1: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

print("\nFEATURE ENGINEERING")
print(f"  - Original numeric features: {len(numeric_features)}")
print(f"  - Engineered features: 3")
print(f"  - Highly correlated features removed: {len(to_drop)}")

print("\nDATA SPLITTING")
print(f"  - Training set: {X_train_scaled.shape[0]} samples ({X_train_scaled.shape[0]/len(X)*100:.1f}%)")
print(f"  - Validation set: {X_val_scaled.shape[0]} samples ({X_val_scaled.shape[0]/len(X)*100:.1f}%)")
print(f"  - Test set: {X_test_scaled.shape[0]} samples ({X_test_scaled.shape[0]/len(X)*100:.1f}%)")

print("\nPREPROCESSING APPLIED")
print(f"  - Missing value handling: Median imputation")
print(f"  - Scaling method: StandardScaler (mean=0, std=1)")
print(f"  - Stratification: Yes (maintains class proportions)")
print(f"  - Class balancing: {'Yes (downsampling)' if len(class_0_bal) > 0 else 'No'}")

print("\nVISUALIZATIONS GENERATED: 13 files")
print("  - Temporal evolution and heatmaps")
print("  - Feature distributions and relationships")
print("  - Correlation matrices")
print("  - Dimensionality reduction (PCA, t-SNE, UMAP)")
print("  - Unit-level time series analysis")

print("\nEXPORTED DATASETS")
print(f"  - X_train.parquet, X_val.parquet, X_test.parquet")
print(f"  - y_train.parquet, y_val.parquet, y_test.parquet")
print(f"  - descriptive_statistics.csv")

print("\n" + "="*70)
print("PIPELINE COMPLETE - Ready for machine learning models")
print("="*70 + "\n")
