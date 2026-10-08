import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# AI CUSTOMER SUPPORT TICKET TRIAGE
# ============================================================

print("=" * 60)
print("AI CUSTOMER SUPPORT TICKET TRIAGE")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "tickets.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found!")
    print(f"Expected location: {DATASET_PATH}")
    exit()


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset loaded from: {DATASET_PATH}")
print(f"Total tickets: {len(df)}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ticket_id",
    "ticket_text",
    "category",
    "urgency",
    "resolution"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)
    exit()


# ============================================================
# 5. DISPLAY DATA DISTRIBUTION
# ============================================================

print("\nCategories:")
print(df["category"].value_counts())

print("\nUrgency:")
print(df["urgency"].value_counts())


# ============================================================
# 6. REMOVE EMPTY TICKETS
# ============================================================

df = df.dropna(
    subset=[
        "ticket_text",
        "category",
        "urgency"
    ]
).reset_index(drop=True)

df["ticket_text"] = df["ticket_text"].astype(str)
df["category"] = df["category"].astype(str)
df["urgency"] = df["urgency"].astype(str)


# ============================================================
# 7. INPUT AND TARGETS
# ============================================================

X = df["ticket_text"]

category_y = df["category"]

urgency_y = df["urgency"]


# ============================================================
# 8. ONE COMMON TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# We create ONE split and use the exact same rows
# for both category and urgency models.

train_indices, test_indices = train_test_split(
    range(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=category_y
)

train_indices = list(train_indices)
test_indices = list(test_indices)


X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

category_train = category_y.iloc[train_indices]
category_test = category_y.iloc[test_indices]

urgency_train = urgency_y.iloc[train_indices]
urgency_test = urgency_y.iloc[test_indices]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 9. TF-IDF TEXT FEATURES
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed.")


# ============================================================
# 10. CATEGORY MODEL
# ============================================================

print("\nTraining Category Model...")

category_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

category_model.fit(
    X_train_tfidf,
    category_train
)

category_predictions = category_model.predict(
    X_test_tfidf
)


# ============================================================
# 11. URGENCY MODEL
# ============================================================

print("\nTraining Urgency Model...")

urgency_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

urgency_model.fit(
    X_train_tfidf,
    urgency_train
)

urgency_predictions = urgency_model.predict(
    X_test_tfidf
)


# ============================================================
# 12. CATEGORY PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("CATEGORY MODEL PERFORMANCE")
print("=" * 60)


category_accuracy = accuracy_score(
    category_test,
    category_predictions
)

category_precision = precision_score(
    category_test,
    category_predictions,
    average="weighted",
    zero_division=0
)

category_recall = recall_score(
    category_test,
    category_predictions,
    average="weighted",
    zero_division=0
)

category_f1 = f1_score(
    category_test,
    category_predictions,
    average="weighted",
    zero_division=0
)


print(f"\nAccuracy  : {category_accuracy:.2%}")
print(f"Precision : {category_precision:.2%}")
print(f"Recall    : {category_recall:.2%}")
print(f"F1 Score  : {category_f1:.2%}")


print("\nDetailed Classification Report:")

print(
    classification_report(
        category_test,
        category_predictions,
        zero_division=0
    )
)


# ============================================================
# 13. URGENCY PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("URGENCY MODEL PERFORMANCE")
print("=" * 60)


urgency_accuracy = accuracy_score(
    urgency_test,
    urgency_predictions
)

urgency_precision = precision_score(
    urgency_test,
    urgency_predictions,
    average="weighted",
    zero_division=0
)

urgency_recall = recall_score(
    urgency_test,
    urgency_predictions,
    average="weighted",
    zero_division=0
)

urgency_f1 = f1_score(
    urgency_test,
    urgency_predictions,
    average="weighted",
    zero_division=0
)


print(f"\nAccuracy  : {urgency_accuracy:.2%}")
print(f"Precision : {urgency_precision:.2%}")
print(f"Recall    : {urgency_recall:.2%}")
print(f"F1 Score  : {urgency_f1:.2%}")


print("\nDetailed Classification Report:")

print(
    classification_report(
        urgency_test,
        urgency_predictions,
        zero_division=0
    )
)


# ============================================================
# 14. SAVE CATEGORY MODEL
# ============================================================

category_model_path = os.path.join(
    MODEL_DIR,
    "category_model.pkl"
)

joblib.dump(
    category_model,
    category_model_path
)


# ============================================================
# 15. SAVE URGENCY MODEL
# ============================================================

urgency_model_path = os.path.join(
    MODEL_DIR,
    "urgency_model.pkl"
)

joblib.dump(
    urgency_model,
    urgency_model_path
)


# ============================================================
# 16. SAVE TF-IDF VECTORIZER
# ============================================================

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    vectorizer,
    vectorizer_path
)


# ============================================================
# 17. CATEGORY CONFUSION MATRIX
# ============================================================

category_cm = confusion_matrix(
    category_test,
    category_predictions,
    labels=category_model.classes_
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    category_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=category_model.classes_,
    yticklabels=category_model.classes_
)

plt.title(
    "Category Classification - Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

category_matrix_path = os.path.join(
    MODEL_DIR,
    "category_confusion_matrix.png"
)

plt.savefig(
    category_matrix_path,
    dpi=150
)

plt.close()


# ============================================================
# 18. URGENCY CONFUSION MATRIX
# ============================================================

urgency_cm = confusion_matrix(
    urgency_test,
    urgency_predictions,
    labels=urgency_model.classes_
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    urgency_cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=urgency_model.classes_,
    yticklabels=urgency_model.classes_
)

plt.title(
    "Urgency Classification - Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

urgency_matrix_path = os.path.join(
    MODEL_DIR,
    "urgency_confusion_matrix.png"
)

plt.savefig(
    urgency_matrix_path,
    dpi=150
)

plt.close()


# ============================================================
# 19. SAVE EVALUATION INFORMATION
# ============================================================

evaluation_data = {
    "dataset_size": int(len(df)),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),

    "category": {
        "accuracy": round(
            float(category_accuracy) * 100,
            2
        ),
        "precision": round(
            float(category_precision) * 100,
            2
        ),
        "recall": round(
            float(category_recall) * 100,
            2
        ),
        "f1_score": round(
            float(category_f1) * 100,
            2
        )
    },

    "urgency": {
        "accuracy": round(
            float(urgency_accuracy) * 100,
            2
        ),
        "precision": round(
            float(urgency_precision) * 100,
            2
        ),
        "recall": round(
            float(urgency_recall) * 100,
            2
        ),
        "f1_score": round(
            float(urgency_f1) * 100,
            2
        )
    }
}


evaluation_path = os.path.join(
    MODEL_DIR,
    "evaluation.json"
)

import json

with open(
    evaluation_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        evaluation_data,
        file,
        indent=4
    )


# ============================================================
# 20. SUCCESS MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("MODELS SAVED SUCCESSFULLY")
print("=" * 60)

print("\nCreated files:")

print(f"✓ {category_model_path}")

print(f"✓ {urgency_model_path}")

print(f"✓ {vectorizer_path}")

print(f"✓ {category_matrix_path}")

print(f"✓ {urgency_matrix_path}")

print(f"✓ {evaluation_path}")


print("\n")
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
print("=" * 60)