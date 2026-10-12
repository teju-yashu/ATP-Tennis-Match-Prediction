from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"


# ============================================================
# EXACT RANKING FEATURES
# ============================================================

RANKING_FEATURES = [
    "rank_difference",
    "rank_points_difference"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("BASELINE CONSISTENCY CHECK")
print("=" * 75)

print("\nLoading prepared datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train      : {len(train):,} rows")
print(f"Validation : {len(validation):,} rows")
print(f"Test       : {len(test):,} rows")


# ============================================================
# CHECK FEATURES
# ============================================================

print("\nChecking ranking features...")

for feature in RANKING_FEATURES:

    print(
        f"{feature:<30} "
        f"Train={feature in train.columns} "
        f"Validation={feature in validation.columns} "
        f"Test={feature in test.columns}"
    )


# ============================================================
# PREPARE X / y
# ============================================================

X_train = train[RANKING_FEATURES].copy()
y_train = train["target"].astype(int)

X_validation = validation[RANKING_FEATURES].copy()
y_validation = validation["target"].astype(int)

X_test = test[RANKING_FEATURES].copy()
y_test = test["target"].astype(int)


# ============================================================
# DATA CHECK
# ============================================================

print("\nMissing values:")

print(
    "Train:"
)

print(
    X_train.isna().sum()
)

print(
    "Validation:"
)

print(
    X_validation.isna().sum()
)

print(
    "Test:"
)

print(
    X_test.isna().sum()
)


# ============================================================
# BUILD ONE STANDARDIZED PIPELINE
# ============================================================

print("\nBuilding standardized Logistic Regression...")

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    X,
    y
):

    prediction = model.predict(X)

    probability = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        prediction
    )

    f1 = f1_score(
        y,
        prediction
    )

    roc_auc = roc_auc_score(
        y,
        probability
    )

    loss = log_loss(
        y,
        probability
    )

    cm = confusion_matrix(
        y,
        prediction
    )

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"Log Loss : {loss:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss
    }


# ============================================================
# EVALUATE
# ============================================================

train_results = evaluate(
    "RANKING BASELINE - TRAIN",
    X_train,
    y_train
)

validation_results = evaluate(
    "RANKING BASELINE - VALIDATION",
    X_validation,
    y_validation
)

test_results = evaluate(
    "RANKING BASELINE - TEST",
    X_test,
    y_test
)


# ============================================================
# DATASET IDENTITY CHECK
# ============================================================

print("\n" + "=" * 75)
print("DATASET IDENTITY CHECK")
print("=" * 75)

print(
    f"\nTrain rows      : {len(train):,}"
)

print(
    f"Validation rows : {len(validation):,}"
)

print(
    f"Test rows       : {len(test):,}"
)

print(
    "\nTarget distribution:"
)

print(
    "\nTRAIN"
)

print(
    y_train.value_counts(
        normalize=True
    ).sort_index()
)

print(
    "\nVALIDATION"
)

print(
    y_validation.value_counts(
        normalize=True
    ).sort_index()
)

print(
    "\nTEST"
)

print(
    y_test.value_counts(
        normalize=True
    ).sort_index()
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("CONSISTENCY CHECK SUMMARY")
print("=" * 75)

print(
    "\nRanking features used:"
)

for feature in RANKING_FEATURES:

    print(
        f"  - {feature}"
    )

print(
    "\nTest Accuracy : "
    f"{test_results['accuracy']:.4f}"
)

print(
    "Test ROC-AUC  : "
    f"{test_results['roc_auc']:.4f}"
)

print(
    "Test F1       : "
    f"{test_results['f1']:.4f}"
)

print(
    "Test Log Loss : "
    f"{test_results['log_loss']:.4f}"
)

print("\nDONE.")