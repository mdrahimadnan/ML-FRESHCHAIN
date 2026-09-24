# ============================================================
# FRESHCHAIN
# MODULE 10
# ENSEMBLE LEARNING
#
# Hard Voting
# Soft Voting
# Stacking
#
# Original Dataset is NEVER Modified
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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

OUTPUT_FOLDER = "../output/M10_Ensemble_Learning"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("ENSEMBLE LEARNING")
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

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ============================================================
# BASE MODELS
# ============================================================

logistic = LogisticRegression(
    max_iter=1000
)

decision_tree = DecisionTreeClassifier(
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

naive_bayes = GaussianNB()

knn = KNeighborsClassifier(
    n_neighbors=5
)

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

print("\nBase Models Created Successfully.")


# ============================================================
# HARD VOTING CLASSIFIER
# ============================================================

hard_voting = VotingClassifier(

    estimators=[

        ("lr", logistic),

        ("dt", decision_tree),

        ("rf", random_forest),

        ("nb", naive_bayes),

        ("knn", knn),

        ("svm", svm)

    ],

    voting="hard"

)

hard_voting.fit(

    X_train_scaled,

    y_train

)

print("\nHard Voting Model Trained Successfully.")

# ============================================================
# SOFT VOTING CLASSIFIER
# ============================================================

soft_voting = VotingClassifier(

    estimators=[

        ("lr", logistic),

        ("dt", decision_tree),

        ("rf", random_forest),

        ("nb", naive_bayes),

        ("knn", knn),

        ("svm", svm)

    ],

    voting="soft"

)

soft_voting.fit(

    X_train_scaled,

    y_train

)

print("Soft Voting Model Trained Successfully.")

# ============================================================
# STACKING CLASSIFIER
# ============================================================

stacking = StackingClassifier(

    estimators=[

        ("lr", logistic),

        ("dt", decision_tree),

        ("rf", random_forest),

        ("nb", naive_bayes),

        ("knn", knn),

        ("svm", svm)

    ],

    final_estimator=LogisticRegression(),

    cv=5,

    n_jobs=-1

)

stacking.fit(

    X_train_scaled,

    y_train

)

print("Stacking Model Trained Successfully.")

# ============================================================
# PREDICTIONS
# ============================================================

hard_prediction = hard_voting.predict(

    X_test_scaled

)

soft_prediction = soft_voting.predict(

    X_test_scaled

)

stack_prediction = stacking.predict(

    X_test_scaled

)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(

        model_name,

        y_true,

        y_pred

):

    accuracy = accuracy_score(

        y_true,

        y_pred

    )

    precision = precision_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    print("\n" + "=" * 60)

    print(model_name)

    print("=" * 60)

    print("Accuracy :", round(accuracy,4))
    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1 Score :", round(f1,4))

    print("\nClassification Report\n")

    print(

        classification_report(

            y_true,

            y_pred,

            zero_division=0

        )

    )

    return [

        accuracy,

        precision,

        recall,

        f1

    ]

# ============================================================
# EVALUATE ALL MODELS
# ============================================================

hard_result = evaluate_model(

    "Hard Voting",

    y_test,

    hard_prediction

)

soft_result = evaluate_model(

    "Soft Voting",

    y_test,

    soft_prediction

)

stack_result = evaluate_model(

    "Stacking",

    y_test,

    stack_prediction

)



# ============================================================
# SAVE METRICS
# ============================================================

comparison = pd.DataFrame({

    "Model":[

        "Hard Voting",

        "Soft Voting",

        "Stacking"

    ],

    "Accuracy":[

        hard_result[0],

        soft_result[0],

        stack_result[0]

    ],

    "Precision":[

        hard_result[1],

        soft_result[1],

        stack_result[1]

    ],

    "Recall":[

        hard_result[2],

        soft_result[2],

        stack_result[2]

    ],

    "F1 Score":[

        hard_result[3],

        soft_result[3],

        stack_result[3]

    ]

})

comparison.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "ensemble_metrics.csv"

    ),

    index=False

)

print("\nensemble_metrics.csv saved.")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction = X_test.copy()

prediction["Actual"] = y_test.values

prediction["Hard_Voting"] = hard_prediction

prediction["Soft_Voting"] = soft_prediction

prediction["Stacking"] = stack_prediction

prediction.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "prediction_results.csv"

    ),

    index=False

)

print("prediction_results.csv saved.")

# ============================================================
# CONFUSION MATRIX FUNCTION
# ============================================================

def save_confusion_matrix(

        y_true,

        y_pred,

        title,

        filename

):

    cm = confusion_matrix(

        y_true,

        y_pred

    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm

    )

    disp.plot()

    plt.title(title)

    plt.savefig(

        os.path.join(

            OUTPUT_FOLDER,

            filename

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

save_confusion_matrix(

    y_test,

    hard_prediction,

    "Hard Voting Confusion Matrix",

    "hard_voting_confusion_matrix.png"

)

save_confusion_matrix(

    y_test,

    soft_prediction,

    "Soft Voting Confusion Matrix",

    "soft_voting_confusion_matrix.png"

)

save_confusion_matrix(

    y_test,

    stack_prediction,

    "Stacking Confusion Matrix",

    "stacking_confusion_matrix.png"

)

print("Confusion Matrix Images Saved.")

# ============================================================
# PERFORMANCE GRAPH
# ============================================================

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score"

]

x = range(len(metrics))

plt.figure(figsize=(10,6))

plt.plot(

    x,

    hard_result,

    marker="o",

    label="Hard Voting"

)

plt.plot(

    x,

    soft_result,

    marker="o",

    label="Soft Voting"

)

plt.plot(

    x,

    stack_result,

    marker="o",

    label="Stacking"

)

plt.xticks(

    x,

    metrics

)

plt.ylim(0,1.05)

plt.ylabel("Score")

plt.title("Ensemble Learning Model Comparison")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "ensemble_model_comparison.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("ensemble_model_comparison.png saved.")

# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(

    hard_voting,

    os.path.join(

        OUTPUT_FOLDER,

        "hard_voting_model.pkl"

    )

)

joblib.dump(

    soft_voting,

    os.path.join(

        OUTPUT_FOLDER,

        "soft_voting_model.pkl"

    )

)

joblib.dump(

    stacking,

    os.path.join(

        OUTPUT_FOLDER,

        "stacking_model.pkl"

    )

)

joblib.dump(

    scaler,

    os.path.join(

        OUTPUT_FOLDER,

        "scaler.pkl"

    )

)

print("Models Saved Successfully.")

# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)

print("MODULE 10 COMPLETED SUCCESSFULLY")

print("=" * 70)

print("\nGenerated Files:")

print("- ensemble_metrics.csv")
print("- prediction_results.csv")
print("- hard_voting_confusion_matrix.png")
print("- soft_voting_confusion_matrix.png")
print("- stacking_confusion_matrix.png")
print("- ensemble_model_comparison.png")
print("- hard_voting_model.pkl")
print("- soft_voting_model.pkl")
print("- stacking_model.pkl")
print("- scaler.pkl")

print("\nOutput Folder:")

print(os.path.abspath(OUTPUT_FOLDER))

