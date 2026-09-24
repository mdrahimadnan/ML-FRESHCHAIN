# ============================================================
# FRESHCHAIN
# MODULE 4
# LOGISTIC REGRESSION
#
# Binary Classification
# Original Dataset is NEVER Modified
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. FILE SETTINGS
# ============================================================

DATASET_PATH = "../output/M3_Preprocessing/final_preprocessed_M3.csv"

OUTPUT_FOLDER = "../output/M4_Logistic_Regression"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Spoilage_Status"]
)

y = df["Spoilage_Status"]

# ============================================================
# 4. TRAIN TEST SPLIT
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
# 5. MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed.")

# ============================================================
# 6. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(X_test)

# ============================================================
# 7. METRICS
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

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# 8. SAVE METRICS
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
# 9. SAVE PREDICTIONS
# ============================================================

prediction = X_test.copy()

prediction["Actual"] = y_test.values

prediction["Predicted"] = y_pred

prediction["Predicted_Probability"] = y_prob.max(axis=1)

prediction.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "prediction_results.csv"

    ),

    index=False

)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred

)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

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
# 12. SAVE MODEL
# ============================================================

joblib.dump(

    model,

    os.path.join(

        OUTPUT_FOLDER,

        "logistic_regression_model.pkl"

    )

)

# ============================================================
# 13. FINISH
# ============================================================

print("\n" + "=" * 70)

print("MODULE 4 COMPLETED SUCCESSFULLY")

print("=" * 70)