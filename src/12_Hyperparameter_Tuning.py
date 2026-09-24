# ============================================================
# FRESHCHAIN
# MODULE 12
# HYPERPARAMETER TUNING
#
# GridSearchCV + Cross Validation
#
# Original Dataset Never Modified
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ============================================================
# FILE SETTINGS
# ============================================================

DATASET_PATH = "../output/M3_Preprocessing/final_preprocessed_M3.csv"

OUTPUT_FOLDER = "../output/M12_Hyperparameter_Tuning"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

print("\nDataset Shape :", df.shape)

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Spoilage_Status"]
)

y = df["Spoilage_Status"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    random_state=42

)

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

parameter_grid = {

    "n_estimators": [

        50,

        100,

        200

    ],

    "max_depth": [

        None,

        10,

        20,

        30

    ],

    "min_samples_split": [

        2,

        5,

        10

    ],

    "min_samples_leaf": [

        1,

        2,

        4

    ],

    "criterion": [

        "gini",

        "entropy"

    ]

}

print("\nTotal Hyperparameter Combinations :")

total = (

    len(parameter_grid["n_estimators"])

    * len(parameter_grid["max_depth"])

    * len(parameter_grid["min_samples_split"])

    * len(parameter_grid["min_samples_leaf"])

    * len(parameter_grid["criterion"])

)

print(total)

# ============================================================
# GRID SEARCH WITH CROSS VALIDATION
# ============================================================

grid_search = GridSearchCV(

    estimator=model,

    param_grid=parameter_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)

print("\nRunning GridSearchCV...")

grid_search.fit(

    X_train,

    y_train

)

print("\nGrid Search Completed Successfully.")

# ============================================================
# BEST PARAMETERS
# ============================================================

print("\nBest Parameters")

print("-" * 40)

print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy")

print("-" * 40)

print(round(grid_search.best_score_, 4))

best_model = grid_search.best_estimator_


