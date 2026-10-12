# ================================================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ATP TENNIS MATCH PREDICTION
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train.csv"
)

VAL_PATH = os.path.join(
    DATA_DIR,
    "validation.csv"
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "test.csv"
)


# ================================================================
# FEATURES
# ================================================================

FEATURES = [
    "rank_difference",
    "rank_points_difference",
    "elo_difference",
    "surface_elo_difference",
    "overall_win_rate_difference",
    "win_rate_last_5_difference",
    "win_rate_last_10_difference",
    "win_rate_last_20_difference",
    "recency_weighted_form_difference",
    "win_rate_last_30_days_difference",
    "surface_win_rate_difference",
    "surface_recent_win_rate_difference",
    "h2h_matches",
    "h2h_win_rate_difference",
    "recent_h2h_win_rate_difference",
    "surface_h2h_matches",
    "surface_h2h_win_rate_difference",
    "tournament_win_rate_difference",
    "age_difference",
    "height_difference",
    "hand_difference",
    "ace_rate_difference",
    "double_fault_rate_difference",
    "first_serve_percentage_difference",
    "first_serve_win_percentage_difference",
    "second_serve_win_percentage_difference",
    "break_point_save_percentage_difference"
]

TARGET = "target"


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 75)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 75)

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train      : {len(train):,} rows")
print(f"Validation : {len(validation):,} rows")
print(f"Test       : {len(test):,} rows")


# ================================================================
# PREPARE DATA
# ================================================================

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_val = validation[FEATURES].copy()
y_val = validation[TARGET].astype(int)

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int)


# ================================================================
# DATA CHECKS
# ================================================================

print("\nChecking data...")

for name, X, y in [
    ("TRAIN", X_train, y_train),
    ("VALIDATION", X_val, y_val),
    ("TEST", X_test, y_test)
]:

    if X.isnull().sum().sum() > 0:
        raise ValueError(
            f"{name} contains missing values."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"{name} contains infinite values."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name} features and target have different lengths."
        )

print("Data checks passed.")


# ================================================================
# BASELINE
# ================================================================

baseline_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

print("\nTraining baseline RF...")

baseline_rf.fit(
    X_train,
    y_train
)

baseline_val_probability = (
    baseline_rf.predict_proba(X_val)[:, 1]
)

baseline_val_prediction = (
    baseline_val_probability >= 0.5
).astype(int)

baseline_accuracy = accuracy_score(
    y_val,
    baseline_val_prediction
)

baseline_f1 = f1_score(
    y_val,
    baseline_val_prediction
)

baseline_auc = roc_auc_score(
    y_val,
    baseline_val_probability
)

baseline_logloss = log_loss(
    y_val,
    baseline_val_probability
)

baseline_brier = brier_score_loss(
    y_val,
    baseline_val_probability
)


print("\n" + "=" * 75)
print("BASELINE VALIDATION")
print("=" * 75)

print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"F1       : {baseline_f1:.4f}")
print(f"ROC-AUC  : {baseline_auc:.4f}")
print(f"Log Loss : {baseline_logloss:.4f}")
print(f"Brier    : {baseline_brier:.4f}")


# ================================================================
# RANDOM SEARCH SPACE
# ================================================================

param_distributions = {

    "n_estimators": [
        300,
        400,
        500,
        600,
        800
    ],

    "max_depth": [
        8,
        10,
        12,
        14,
        16
    ],

    "min_samples_split": [
        5,
        8,
        10,
        12,
        15
    ],

    "min_samples_leaf": [
        2,
        3,
        5,
        7,
        10
    ],

    "max_features": [
        "sqrt",
        0.3,
        0.5,
        0.7
    ],

    "criterion": [
        "gini",
        "entropy",
        "log_loss"
    ]
}


# ================================================================
# RANDOMIZED SEARCH
# ================================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_distributions,

    n_iter=30,

    scoring="accuracy",

    cv=3,

    random_state=42,

    n_jobs=-1,

    verbose=2,

    return_train_score=False
)


print("\n" + "=" * 75)
print("STARTING RANDOMIZED SEARCH")
print("=" * 75)

print("\nTesting 30 configurations.")
print("Each configuration uses 3-fold CV.")
print("Test set will NOT be used during tuning.")

random_search.fit(
    X_train,
    y_train
)


# ================================================================
# BEST MODEL
# ================================================================

best_rf = random_search.best_estimator_


print("\n" + "=" * 75)
print("BEST RANDOM FOREST")
print("=" * 75)

print("\nBest parameters:")

for parameter, value in (
    random_search.best_params_.items()
):

    print(
        f"{parameter:25s}: {value}"
    )

print(
    f"\nBest CV accuracy: "
    f"{random_search.best_score_:.4f}"
)


# ================================================================
# VALIDATION EVALUATION
# ================================================================

val_probability = (
    best_rf.predict_proba(X_val)[:, 1]
)

val_prediction = (
    val_probability >= 0.5
).astype(int)


val_accuracy = accuracy_score(
    y_val,
    val_prediction
)

val_f1 = f1_score(
    y_val,
    val_prediction
)

val_auc = roc_auc_score(
    y_val,
    val_probability
)

val_logloss = log_loss(
    y_val,
    val_probability
)

val_brier = brier_score_loss(
    y_val,
    val_probability
)


print("\n" + "=" * 75)
print("TUNED RF - VALIDATION")
print("=" * 75)

print(f"Accuracy : {val_accuracy:.4f}")
print(f"F1       : {val_f1:.4f}")
print(f"ROC-AUC  : {val_auc:.4f}")
print(f"Log Loss : {val_logloss:.4f}")
print(f"Brier    : {val_brier:.4f}")


# ================================================================
# IMPROVEMENT OVER BASELINE
# ================================================================

print("\n" + "=" * 75)
print("IMPROVEMENT OVER BASELINE")
print("=" * 75)

print(
    f"\nAccuracy:"
    f"\n  Baseline : {baseline_accuracy:.4f}"
    f"\n  Tuned    : {val_accuracy:.4f}"
    f"\n  Change   : "
    f"{val_accuracy - baseline_accuracy:+.4f}"
)

print(
    f"\nF1:"
    f"\n  Baseline : {baseline_f1:.4f}"
    f"\n  Tuned    : {val_f1:.4f}"
    f"\n  Change   : "
    f"{val_f1 - baseline_f1:+.4f}"
)

print(
    f"\nROC-AUC:"
    f"\n  Baseline : {baseline_auc:.4f}"
    f"\n  Tuned    : {val_auc:.4f}"
    f"\n  Change   : "
    f"{val_auc - baseline_auc:+.4f}"
)

print(
    f"\nLog Loss:"
    f"\n  Baseline : {baseline_logloss:.4f}"
    f"\n  Tuned    : {val_logloss:.4f}"
    f"\n  Change   : "
    f"{val_logloss - baseline_logloss:+.4f}"
)

print(
    f"\nBrier:"
    f"\n  Baseline : {baseline_brier:.4f}"
    f"\n  Tuned    : {val_brier:.4f}"
    f"\n  Change   : "
    f"{val_brier - baseline_brier:+.4f}"
)


# ================================================================
# SAVE SEARCH RESULTS
# ================================================================

RESULTS_DIR = os.path.join(
    DATA_DIR,
    "random_forest_tuning"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


results = pd.DataFrame(
    random_search.cv_results_
)

results = results.sort_values(
    "rank_test_score"
)

results_path = os.path.join(
    RESULTS_DIR,
    "randomized_search_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ================================================================
# SAVE BEST PARAMETERS
# ================================================================

best_params_path = os.path.join(
    RESULTS_DIR,
    "best_parameters.txt"
)

with open(
    best_params_path,
    "w"
) as file:

    file.write(
        "BEST RANDOM FOREST PARAMETERS\n"
    )

    file.write(
        "=" * 50 + "\n"
    )

    for parameter, value in (
        random_search.best_params_.items()
    ):

        file.write(
            f"{parameter}: {value}\n"
        )

    file.write(
        f"\nBest CV Accuracy: "
        f"{random_search.best_score_:.6f}\n"
    )

    file.write(
        f"\nValidation Accuracy: "
        f"{val_accuracy:.6f}\n"
    )

    file.write(
        f"Validation F1: "
        f"{val_f1:.6f}\n"
    )

    file.write(
        f"Validation ROC-AUC: "
        f"{val_auc:.6f}\n"
    )

    file.write(
        f"Validation Log Loss: "
        f"{val_logloss:.6f}\n"
    )

    file.write(
        f"Validation Brier: "
        f"{val_brier:.6f}\n"
    )


print("\n" + "=" * 75)
print("TUNING COMPLETED")
print("=" * 75)

print(
    f"\nSearch results saved to:"
    f"\n{results_path}"
)

print(
    f"\nBest parameters saved to:"
    f"\n{best_params_path}"
)

print("\nIMPORTANT:")
print("The test set has NOT been used to select the model.")
print("Do not evaluate on the test set yet.")