# ============================================================
# FRESHCHAIN
# MODULE 6
# RANDOM FOREST CLASSIFIER
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# FILE SETTINGS
# ============================================================

DATASET_PATH = "../output/M3_Preprocessing/final_preprocessed_M3.csv"

OUTPUT_FOLDER = "../output/M6_Random_Forest"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("RANDOM FOREST CLASSIFIER")
print("=" * 70)

print("\nDataset Shape :", df.shape)

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Spoilage_Status"]
)

y = df["Spoilage_Status"]

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

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=100,

    criterion="gini",

    random_state=42,

    n_jobs=-1

)

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed.")

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1
    ]

})

metrics.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "metrics.csv"

    ),

    index=False

)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction = X_test.copy()

prediction["Actual"] = y_test.values

prediction["Predicted"] = y_pred

prediction.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "prediction_results.csv"

    ),

    index=False

)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm

)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "confusion_matrix.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

importance.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "feature_importance.csv"

    ),

    index=False

)

plt.figure(figsize=(12,6))

plt.bar(

    importance["Feature"],

    importance["Importance"]

)

plt.xticks(rotation=90)

plt.ylabel("Importance")

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "feature_importance.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    model,

    os.path.join(

        OUTPUT_FOLDER,

        "random_forest_model.pkl"

    )

)

# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("MODULE 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files:")

print("- metrics.csv")
print("- prediction_results.csv")
print("- confusion_matrix.png")
print("- feature_importance.csv")
print("- feature_importance.png")
print("- random_forest_model.pkl")

print("\nOutput Folder:")
print(os.path.abspath(OUTPUT_FOLDER))