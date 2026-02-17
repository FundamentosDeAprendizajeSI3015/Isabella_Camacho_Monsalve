# ==========================================
# EDA AND PREPROCESSING PIPELINE
# Student-Career Adjustment Project
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

# Delete highly correlated variables (threshold > 0.85)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
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
# 13. Standardization
# ------------------------------------------
features = df_reduced
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

df_scaled = pd.DataFrame(scaled_data, columns=features.columns)
df_scaled.to_csv("outputs/final_scaled_dataset.csv", index=False)

# ------------------------------------------
# 14. Final conclusions
# ------------------------------------------
print("\n===== FINAL CONCLUSIONS =====")

print(f"Final number of samples: {df_scaled.shape[0]}")
print(f"Final number of features: {df_scaled.shape[1]}")

print("\nKey descriptive insights:")

print(f"- Average intrinsic motivation: {df['intrinsic_motivation'].mean():.2f}")
print(f"- Average extrinsic motivation: {df['extrinsic_motivation'].mean():.2f}")
print(f"- Average academic stress: {df['academic_stress'].mean():.2f}")
print(f"- Average academic adjustment index: {df['academic_adjustment'].mean():.2f}")

print("\nCorrelation insights:")
strong_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
strong_corr = strong_corr[strong_corr < 1].head(3)

print("Top correlated variable pairs:")
print(strong_corr)

print("\nMulticollinearity control:")
print(f"- Highly correlated variables removed: {to_drop}")

print("\nData successfully cleaned, transformed, engineered and standardized.")
print("Dataset is now ready for supervised or unsupervised modeling.")
print("All visualizations exported to 'outputs/' folder.")