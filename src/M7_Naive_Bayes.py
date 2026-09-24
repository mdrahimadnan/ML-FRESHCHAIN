# ============================================================
# FRESHCHAIN
# MODULE 7
# NAIVE BAYES CLASSIFIER
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

OUTPUT_FOLDER = "../output/M7_Naive_Bayes"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("NAIVE BAYES CLASSIFIER")
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
# MODEL
# ============================================================

model = GaussianNB()

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

accuracy = accuracy_score(y_test, y_pred)

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

plt.title("Naive Bayes Confusion Matrix")

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
# FEATURE STATISTICS
# ============================================================

mean_df = pd.DataFrame(
    model.theta_,
    columns=X.columns
)

mean_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "feature_means.csv"
    ),

    index=False

)

variance_df = pd.DataFrame(
    model.var_,
    columns=X.columns
)

variance_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "feature_variances.csv"
    ),

    index=False

)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    model,

    os.path.join(
        OUTPUT_FOLDER,
        "naive_bayes_model.pkl"
    )

)

# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("MODULE 7 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files:")

print("- metrics.csv")
print("- prediction_results.csv")
print("- confusion_matrix.png")
print("- feature_means.csv")
print("- feature_variances.csv")
print("- naive_bayes_model.pkl")

print("\nOutput Folder:")
print(os.path.abspath(OUTPUT_FOLDER))