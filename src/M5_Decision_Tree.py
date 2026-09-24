# ============================================================
# FRESHCHAIN
# MODULE 5
# DECISION TREE CLASSIFIER
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

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

OUTPUT_FOLDER = "../output/M5_Decision_Tree"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

print("=" * 70)
print("DECISION TREE")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


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
# CREATE MODEL
# ============================================================

model = DecisionTreeClassifier(

    criterion="gini",

    random_state=42

)

model.fit(

    X_train,

    y_train

)

print("\nDecision Tree Model Trained Successfully.")




# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


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

print("\nmetrics.csv saved.")


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

print("prediction_results.csv saved.")


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

plt.title(
    "Decision Tree Confusion Matrix"
)

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "confusion_matrix.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("confusion_matrix.png saved.")



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

plt.xticks(

    rotation=90

)

plt.ylabel("Importance")

plt.title("Decision Tree Feature Importance")

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

print("feature_importance.png saved.")


# ============================================================
# DECISION TREE
# ============================================================

plt.figure(figsize=(20,10))

plot_tree(

    model,

    feature_names=X.columns,

    class_names=[str(i) for i in sorted(y.unique())],

    filled=True,

    rounded=True,

    fontsize=8

)

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "decision_tree.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("decision_tree.png saved.")



# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    model,

    os.path.join(

        OUTPUT_FOLDER,

        "decision_tree_model.pkl"

    )

)

print("decision_tree_model.pkl saved.")



# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("MODULE 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput Folder:")
print(os.path.abspath(OUTPUT_FOLDER))