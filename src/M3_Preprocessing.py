# ============================================================
# FRESHCHAIN
# MODULE 3
# DATA PREPROCESSING
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. FILE SETTINGS
# ============================================================

DATASET_PATH = "../dataset/fruit_spoilage.csv"

OUTPUT_FOLDER = "../output/M3_Preprocessing"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# 2. LOAD DATASET
# ============================================================

df_original = pd.read_csv(DATASET_PATH)

# Work on copy only

df = df_original.copy(deep=True)

print("=" * 70)
print("FRESHCHAIN PREPROCESSING")
print("=" * 70)

# ============================================================
# 3. RENAME COLUMNS
# ============================================================

df.rename(

    columns={

        "Humid (%)": "Humidity",

        "Light (Fux)": "Light",

        "CO2 (pmm)": "CO2",

        "Class": "Spoilage_Status"

    },

    inplace=True

)

print("\nColumns after renaming")

print(df.columns.tolist())

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

duplicate_rows = df.duplicated().sum()

print("\nDuplicate Rows :", duplicate_rows)

df = df.drop_duplicates()

print("Rows after removing duplicates :", len(df))

# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values")

print(df.isnull().sum())

# ============================================================
# 6. LABEL ENCODING
# ============================================================

fruit_encoder = LabelEncoder()

status_encoder = LabelEncoder()

df["Fruit"] = fruit_encoder.fit_transform(

    df["Fruit"]

)

df["Spoilage_Status"] = status_encoder.fit_transform(

    df["Spoilage_Status"]

)

print("\nFruit Encoding")

for name, value in zip(

        fruit_encoder.classes_,

        fruit_encoder.transform(

            fruit_encoder.classes_

        )):

    print(name, "->", value)

print("\nSpoilage Status Encoding")

for name, value in zip(

        status_encoder.classes_,

        status_encoder.transform(

            status_encoder.classes_

        )):

    print(name, "->", value)

# ============================================================
# 7. FEATURE SCALING
# ============================================================

feature_columns = [

    "Fruit",

    "Temp",

    "Humidity",

    "Light",

    "CO2"

]

scaler = StandardScaler()

df[feature_columns] = scaler.fit_transform(

    df[feature_columns]

)

print("\nFeature Scaling Completed")

# ============================================================
# 8. SAVE PREPROCESSED DATASET
# ============================================================

preprocessed_file = os.path.join(

    OUTPUT_FOLDER,

    "final_preprocessed_M3.csv"

)

df.to_csv(

    preprocessed_file,

    index=False

)

print("\nPreprocessed Dataset Saved")

print(preprocessed_file)

# ============================================================
# 9. SAVE ENCODING DETAILS
# ============================================================

fruit_mapping = pd.DataFrame({

    "Fruit": fruit_encoder.classes_,

    "Encoded_Value":

        fruit_encoder.transform(

            fruit_encoder.classes_

        )

})

fruit_mapping.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "fruit_encoding.csv"

    ),

    index=False

)

status_mapping = pd.DataFrame({

    "Status": status_encoder.classes_,

    "Encoded_Value":

        status_encoder.transform(

            status_encoder.classes_

        )

})

status_mapping.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "status_encoding.csv"

    ),

    index=False

)

# ============================================================
# 10. SAVE PREPROCESSING SUMMARY
# ============================================================

summary_file = os.path.join(

    OUTPUT_FOLDER,

    "preprocessing_summary.txt"

)

with open(

        summary_file,

        "w",

        encoding="utf-8"

) as file:

    file.write(

        "FRESHCHAIN PREPROCESSING SUMMARY\n"

    )

    file.write("=" * 70)

    file.write(

        "\n\nRows : "

        + str(df.shape[0])

    )

    file.write(

        "\nColumns : "

        + str(df.shape[1])

    )

    file.write(

        "\nDuplicate Rows Removed : "

        + str(duplicate_rows)

    )

    file.write(

        "\n\nColumns\n"

    )

    for column in df.columns:

        file.write(column + "\n")

# ============================================================
# 11. VERIFY ORIGINAL DATASET
# ============================================================

check = pd.read_csv(DATASET_PATH)

if df_original.equals(check):

    print("\nOriginal dataset NOT modified.")

else:

    print("\nWARNING : Original dataset changed!")

# ============================================================
# 12. FINISH
# ============================================================

print("\n" + "=" * 70)

print("MODULE 3 COMPLETED SUCCESSFULLY")

print("=" * 70)