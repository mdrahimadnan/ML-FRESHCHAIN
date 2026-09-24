# ============================================================
# FRESHCHAIN
# MODULE 2
# EXPLORATORY DATA ANALYSIS (EDA)
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. FILE SETTINGS
# ============================================================

DATASET_PATH = "../dataset/fruit_spoilage.csv"
OUTPUT_FOLDER = "../output/M2_EDA"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================

df_original = pd.read_csv(DATASET_PATH)

# Work on a copy
df = df_original.copy(deep=True)

print("=" * 70)
print("FRESHCHAIN - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\nDataset Shape:", df.shape)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

# ============================================================
# 4. NUMERICAL SUMMARY
# ============================================================

summary = df.describe(include="all")

print("\nStatistical Summary")
print(summary)

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "EDA_Summary.csv"
    )
)

# ============================================================
# 5. HISTOGRAMS
# ============================================================

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:

    plt.figure(figsize=(7,5))

    plt.hist(
        df[column].dropna(),
        bins=20
    )

    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{column}_Histogram.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 6. BOXPLOTS
# ============================================================

for column in numeric_columns:

    plt.figure(figsize=(6,5))

    sns.boxplot(
        y=df[column]
    )

    plt.title(column)

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{column}_Boxplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

if len(numeric_columns) > 1:

    correlation = df[numeric_columns].corr()

    correlation.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "Correlation_Matrix.csv"
        )
    )

    plt.figure(figsize=(10,8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Correlation_Heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

if "Spoilage_Status" in df.columns:

    plt.figure(figsize=(6,5))

    sns.countplot(
        x="Spoilage_Status",
        data=df
    )

    plt.title("Spoilage Status Distribution")

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Target_Distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 9. VERIFY ORIGINAL DATASET
# ============================================================

check = pd.read_csv(DATASET_PATH)

if df_original.equals(check):

    print("\nOriginal dataset was NOT modified.")

else:

    print("\nWARNING: Original dataset changed!")

# ============================================================
# 10. FINISH
# ============================================================

print("\n" + "=" * 70)
print("MODULE 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutputs saved in:")
print(OUTPUT_FOLDER)