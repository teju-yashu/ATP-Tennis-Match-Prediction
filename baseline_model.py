from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

RESULTS_PATH = DATA_DIR / "baseline_results.csv"


# ============================================================
# RANKING BASELINE FEATURES
# ============================================================

RANKING_FEATURES = [
    "rank_difference",
    "rank_points_difference"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("ATP TENNIS RANKING BASELINE")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading prepared datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

print(
    f"Train      : {len(train):,} rows"
)

print(
    f"Validation : {len(validation):,} rows"
)

print(
    f"Test       : {len(test):,} rows"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

print("\nChecking required columns...")

required_columns = (
    RANKING_FEATURES
    + ["target"]
)

for column in required_columns:

    if column not in train.columns:
        raise KeyError(
            f"Missing column in train dataset: {column}"
        )

    if column not in validation.columns:
        raise KeyError(
            f"Missing column in validation dataset: {column}"
        )

    if column not in test.columns:
        raise KeyError(
            f"Missing column in test dataset: {column}"
        )

print("All required columns are present.")


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X_train = train[
    RANKING_FEATURES
].copy()

y_train = train[
    "target"
].astype(int)

X_validation = validation[
    RANKING_FEATURES
].copy()

y_validation = validation[
    "target"
].astype(int)

X_test = test[
    RANKING_FEATURES
].copy()

y_test = test[
    "target"
].astype(int)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\nChecking missing values...")

print("\nTRAIN:")

print(
    X_train.isna().sum()
)

print("\nVALIDATION:")

print(
    X_validation.isna().sum()
)

print("\nTEST:")

print(
    X_test.isna().sum()
)


# ============================================================
# BUILD CANONICAL BASELINE PIPELINE
# ============================================================

print("\nBuilding ranking baseline pipeline...")

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
# TRAIN MODEL
# ============================================================

print("\nTraining ranking baseline...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    X,
    y
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    f1 = f1_score(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    logloss = log_loss(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    report = classification_report(
        y,
        predictions,
        target_names=[
            "Player 2 wins",
            "Player 1 wins"
        ],
        digits=2
    )

    print("\n" + "=" * 75)

    print(
        f"Ranking Baseline - {name}"
    )

    print("=" * 75)

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
        f"Log Loss : {logloss:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(report)

    return {
        "model": f"Ranking Baseline - {name}",
        "dataset": name,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": logloss
    }


# ============================================================
# EVALUATE TRAIN
# ============================================================

train_results = evaluate_model(
    "Train",
    X_train,
    y_train
)


# ============================================================
# EVALUATE VALIDATION
# ============================================================

validation_results = evaluate_model(
    "Validation",
    X_validation,
    y_validation
)


# ============================================================
# EVALUATE TEST
# ============================================================

test_results = evaluate_model(
    "Test",
    X_test,
    y_test
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        train_results,
        validation_results,
        test_results
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n" + "=" * 75)
print("RANKING BASELINE SUMMARY")
print("=" * 75)

print(
    results[
        [
            "model",
            "accuracy",
            "f1",
            "roc_auc",
            "log_loss"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nBaseline results saved to:")

print(
    RESULTS_PATH
)

print("\n" + "=" * 75)
print("BASELINE COMPLETED")
print("=" * 75)