# ============================================================
# FRESHCHAIN
# MODULE 11
# MODEL COMPARISON
#
# Compare All Machine Learning Models
#
# Original Dataset Never Modified
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier
)

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score

)

# ============================================================
# FILE SETTINGS
# ============================================================

DATASET_PATH = "../output/M3_Preprocessing/final_preprocessed_M3.csv"

OUTPUT_FOLDER = "../output/M11_Model_Comparison"

os.makedirs(

    OUTPUT_FOLDER,

    exist_ok=True

)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("="*70)

print("MODEL COMPARISON")

print("="*70)

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
# SPLIT
# ============================================================

X_train,X_test,y_train,y_test=train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(

    X_train

)

X_test_scaled = scaler.transform(

    X_test

)

print()

print("Training Samples :",len(X_train))

print("Testing Samples  :",len(X_test))


# ============================================================
# CREATE MODELS
# ============================================================

logistic = LogisticRegression(

    max_iter=1000,

    random_state=42

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

    random_state=42

)

hard_voting = VotingClassifier(

    estimators=[

        ("lr", logistic),

        ("dt", decision_tree),

        ("rf", random_forest)

    ],

    voting="hard"

)

stacking = StackingClassifier(

    estimators=[

        ("lr", logistic),

        ("dt", decision_tree),

        ("rf", random_forest)

    ],

    final_estimator=LogisticRegression()

)

# ============================================================
# STORE MODELS
# ============================================================

models = {

    "Logistic Regression": logistic,

    "Decision Tree": decision_tree,

    "Random Forest": random_forest,

    "Naive Bayes": naive_bayes,

    "KNN": knn,

    "SVM": svm,

    "Hard Voting": hard_voting,

    "Stacking": stacking

}

print("\nModels Created Successfully.")

# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    if model_name in [

        "Logistic Regression",

        "KNN",

        "SVM"

    ]:

        model.fit(

            X_train_scaled,

            y_train

        )

        prediction = model.predict(

            X_test_scaled

        )

    else:

        model.fit(

            X_train,

            y_train

        )

        prediction = model.predict(

            X_test

        )

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    precision = precision_score(

        y_test,

        prediction,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_test,

        prediction,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        prediction,

        average="weighted",

        zero_division=0

    )

    results.append([

        model_name,

        accuracy,

        precision,

        recall,

        f1

    ])

    print("Completed.")



# ============================================================
# CREATE RESULT TABLE
# ============================================================

comparison = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison)

# ============================================================
# SAVE CSV
# ============================================================

comparison_file = os.path.join(

    OUTPUT_FOLDER,

    "model_comparison.csv"

)

comparison.to_csv(

    comparison_file,

    index=False

)

print("\nComparison CSV Saved.")

# ============================================================
# BEST MODEL
# ============================================================

best_model = comparison.loc[

    comparison["Accuracy"].idxmax()

]

print("\nBest Model")

print("-----------------------------")

print("Model    :", best_model["Model"])

print("Accuracy :", round(best_model["Accuracy"], 4))

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10,6))

plt.bar(

    comparison["Model"],

    comparison["Accuracy"]

)

plt.xticks(

    rotation=30,

    ha="right"

)

plt.ylabel("Accuracy")

plt.title("Accuracy Comparison of ML Models")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "accuracy_comparison.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# PRECISION RECALL F1 GRAPH
# ============================================================

x = range(len(comparison))

width = 0.25

plt.figure(figsize=(12,6))

plt.bar(

    [i-width for i in x],

    comparison["Precision"],

    width,

    label="Precision"

)

plt.bar(

    x,

    comparison["Recall"],

    width,

    label="Recall"

)

plt.bar(

    [i+width for i in x],

    comparison["F1 Score"],

    width,

    label="F1 Score"

)

plt.xticks(

    x,

    comparison["Model"],

    rotation=30,

    ha="right"

)

plt.ylabel("Score")

plt.title("Precision, Recall and F1 Comparison")

plt.legend()

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "precision_recall_f1.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# SAVE BEST MODEL INFO
# ============================================================

with open(

    os.path.join(

        OUTPUT_FOLDER,

        "best_model.txt"

    ),

    "w",

    encoding="utf-8"

) as file:

    file.write("FRESHCHAIN\n")

    file.write("Best Machine Learning Model\n\n")

    file.write(f"Model : {best_model['Model']}\n")

    file.write(f"Accuracy : {best_model['Accuracy']:.4f}\n")

    file.write(f"Precision : {best_model['Precision']:.4f}\n")

    file.write(f"Recall : {best_model['Recall']:.4f}\n")

    file.write(f"F1 Score : {best_model['F1 Score']:.4f}\n")

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print("MODULE 11 COMPLETED SUCCESSFULLY")

print("=" * 70)

print("\nGenerated Files:")

print("1. model_comparison.csv")

print("2. accuracy_comparison.png")

print("3. precision_recall_f1.png")

print("4. best_model.txt")

print("\nOutput Folder:")

print(os.path.abspath(OUTPUT_FOLDER))


