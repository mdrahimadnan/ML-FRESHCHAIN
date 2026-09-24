# ============================================================
# FRESHCHAIN
# MODULE 9
# SUPPORT VECTOR MACHINE (SVM)
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

OUTPUT_FOLDER = "../output/M9_SVM"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("SUPPORT VECTOR MACHINE")
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
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================================
# CREATE SVM MODEL
# ============================================================

model = SVC(

    kernel="rbf",

    C=1.0,

    gamma="scale",

    probability=True,

    random_state=42

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
# MODEL EVALUATION
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

prediction = pd.DataFrame(X_test)

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

plt.title("Support Vector Machine")

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
# SUPPORT VECTORS INFORMATION
# ============================================================

support_vectors = pd.DataFrame(
    model.support_vectors_
)

support_vectors.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "support_vectors.csv"

    ),

    index=False

)

support_count = pd.DataFrame({

    "Support_Vectors":[
        len(model.support_)
    ]

})

support_count.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "support_vector_count.csv"

    ),

    index=False

)

# ============================================================
# SUPPORT VECTOR GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    ["Support Vectors"],

    [len(model.support_)]

)

plt.ylabel("Count")

plt.title("Number of Support Vectors")

plt.grid(axis="y")

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "support_vector_graph.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(

    scaler,

    os.path.join(

        OUTPUT_FOLDER,

        "scaler.pkl"

    )

)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    model,

    os.path.join(

        OUTPUT_FOLDER,

        "svm_model.pkl"

    )

)

# ============================================================
# FINISH
# ============================================================

print("\n" + "="*70)
print("MODULE 9 COMPLETED SUCCESSFULLY")
print("="*70)

print("\nGenerated Files")

print("- metrics.csv")
print("- prediction_results.csv")
print("- confusion_matrix.png")
print("- support_vectors.csv")
print("- support_vector_count.csv")
print("- support_vector_graph.png")
print("- scaler.pkl")
print("- svm_model.pkl")

print("\nOutput Folder")
print(os.path.abspath(OUTPUT_FOLDER))