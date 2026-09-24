# ============================================================
# FRESHCHAIN
# MODULE 1
# DATASET LOADING AND INSPECTION
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import pandas as pd

# ============================================================
# 1. FILE SETTINGS
# ============================================================

DATASET_PATH = "../dataset/fruit_spoilage.csv"

OUTPUT_FOLDER = "../output/M1_Dataset_Loading"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(

        f"\nDataset not found:\n{DATASET_PATH}"

    )

# ============================================================
# 3. LOAD DATASET
# ============================================================

df_original = pd.read_csv(DATASET_PATH)

# ============================================================
# 4. CREATE COPY
# ============================================================

df = df_original.copy(deep=True)

# ============================================================
# 5. DISPLAY DATASET
# ============================================================

print("=" * 70)
print("FRESHCHAIN DATASET")
print("=" * 70)

print("\nFirst Five Records")
print(df.head())

print("\nLast Five Records")
print(df.tail())

# ============================================================
# 6. SHAPE
# ============================================================

print("\nDataset Shape")
print(df.shape)

rows = df.shape[0]
columns = df.shape[1]

print("\nRows :", rows)
print("Columns :", columns)

# ============================================================
# 7. COLUMN NAMES
# ============================================================

print("\nColumn Names")

for column in df.columns:
    print(column)

# ============================================================
# 8. DATA TYPES
# ============================================================

print("\nData Types")

print(df.dtypes)

# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\nMissing Values")

missing = df.isnull().sum()

print(missing)

# ============================================================
# 10. DUPLICATE VALUES
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Records")

print(duplicates)

# ============================================================
# 11. DATASET INFORMATION
# ============================================================

print("\nDataset Information")

df.info()

# ============================================================
# 12. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary")

print(df.describe(include="all"))

# ============================================================
# 13. SAVE DATASET INFORMATION
# ============================================================

summary_file = os.path.join(

    OUTPUT_FOLDER,

    "dataset_summary.txt"

)

with open(
        summary_file,
        "w",
        encoding="utf-8"
) as file:

    file.write("FRESHCHAIN DATASET SUMMARY\n")

    file.write("=" * 70)

    file.write("\n\nShape\n")

    file.write(str(df.shape))

    file.write("\n\nColumns\n")

    for column in df.columns:
        file.write(column + "\n")

    file.write("\n\nData Types\n")

    file.write(str(df.dtypes))

    file.write("\n\nMissing Values\n")

    file.write(str(missing))

    file.write("\n\nDuplicate Records\n")

    file.write(str(duplicates))

# ============================================================
# 14. SAVE DATASET COPY
# ============================================================

copy_file = os.path.join(

    OUTPUT_FOLDER,

    "dataset_copy.csv"

)

df.to_csv(
    copy_file,
    index=False
)

# ============================================================
# 15. VERIFY ORIGINAL DATASET
# ============================================================

check = pd.read_csv(DATASET_PATH)

if df_original.equals(check):

    print("\nOriginal dataset was NOT modified.")

else:

    print("\nWARNING : Original dataset changed!")

# ============================================================
# 16. FINISH
# ============================================================

print("\n" + "=" * 70)

print("MODULE 1 COMPLETED SUCCESSFULLY")

print("=" * 70)

print("\nGenerated Files")

print("dataset_summary.txt")

print("dataset_copy.csv")