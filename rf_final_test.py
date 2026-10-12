# rf_final_test.py

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# This script is inside:
# miniproject/src/
#
# Data is inside:
# miniproject/data/

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "train.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "validation.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "test.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "feature_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. FROZEN FEATURES
# ============================================================

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
    "surface_recent_win_rate_difference",
    "h2h_win_rate_difference",
    "surface_h2h_matches",
    "surface_h2h_win_rate_difference",
    "tournament_win_rate_difference",
    "age_difference",
    "height_difference",
    "ace_rate_difference",
    "double_fault_rate_difference",
    "first_serve_percentage_difference",
    "first_serve_win_percentage_difference",
    "second_serve_win_percentage_difference",
    "break_point_save_percentage_difference"
]


# ============================================================
# 3. FROZEN MODEL CONFIGURATION
# ============================================================

TARGET = "target"

THRESHOLD = 0.54

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "random_state": 42,
    "n_jobs": -1
}


# ============================================================
# 4. START
# ============================================================

print("=" * 75)
print("FINAL RANDOM FOREST EVALUATION - 2024 TEST SET")
print("=" * 75)

print("\nProject directory:")
print(BASE_DIR)

print("\nTraining file:")
print(TRAIN_PATH)

print("\nValidation file:")
print(VAL_PATH)

print("\nTest file:")
print(TEST_PATH)

print("\nFrozen threshold:")
print(THRESHOLD)


# ============================================================
# 5. CHECK FILES
# ============================================================

for path, name in [
    (TRAIN_PATH, "Training"),
    (VAL_PATH, "Validation"),
    (TEST_PATH, "Test")
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING DATA")
print("=" * 75)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nTraining rows   : {len(train_df):,}")
print(f"Validation rows : {len(val_df):,}")
print(f"Test rows       : {len(test_df):,}")


# ============================================================
# 7. CHECK FEATURES
# ============================================================

for dataset, name in [
    (train_df, "training"),
    (val_df, "validation"),
    (test_df, "test")
]:

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in dataset.columns
    ]

    if missing_features:

        raise ValueError(
            f"\nMissing features in {name} data:\n"
            + "\n".join(missing_features)
        )


# ============================================================
# 8. CHECK TARGET
# ============================================================

for dataset, name in [
    (train_df, "training"),
    (val_df, "validation"),
    (test_df, "test")
]:

    if TARGET not in dataset.columns:

        raise ValueError(
            f"\nTarget column '{TARGET}' "
            f"not found in {name} data."
        )


# ============================================================
# 9. COMBINE TRAIN + VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("PREPARING FINAL TRAINING DATA")
print("=" * 75)

# We have already used validation to select:
# - features
# - model configuration
# - threshold
#
# Therefore, we can now use both training and validation
# data to train the final model before testing on 2024.

final_train_df = pd.concat(
    [
        train_df,
        val_df
    ],
    ignore_index=True
)

X_train_final = final_train_df[FEATURES]
y_train_final = final_train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(
    f"\nFinal training rows : "
    f"{len(final_train_df):,}"
)

print(
    f"Final test rows     : "
    f"{len(test_df):,}"
)

print(
    f"Number of features  : "
    f"{len(FEATURES)}"
)


# ============================================================
# 10. TRAIN FINAL RANDOM FOREST
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 75)

print("\nFrozen parameters:")

for parameter, value in MODEL_PARAMS.items():

    print(
        f"{parameter:20s} = {value}"
    )

print(
    f"{'threshold':20s} = {THRESHOLD}"
)

model = RandomForestClassifier(
    **MODEL_PARAMS
)

print("\nTraining final model...")

model.fit(
    X_train_final,
    y_train_final
)

print("Final model training complete.")


# ============================================================
# 11. GENERATE TEST PROBABILITIES
# ============================================================

print("\nGenerating 2024 test probabilities...")

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

print("Test probabilities generated.")


# ============================================================
# 12. APPLY FROZEN THRESHOLD
# ============================================================

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(int)


# ============================================================
# 13. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


# ============================================================
# 14. PROBABILITY METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

logloss = log_loss(
    y_test,
    test_probabilities
)

brier = brier_score_loss(
    y_test,
    test_probabilities
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# 16. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL 2024 TEST RESULTS")
print("=" * 75)

print("\nClassification Metrics")
print("-" * 40)

print(
    f"Accuracy  : {accuracy:.6f}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1 Score  : {f1:.6f}"
)


print("\nProbability Metrics")
print("-" * 40)

print(
    f"ROC-AUC   : {roc_auc:.6f}"
)

print(
    f"Log Loss  : {logloss:.6f}"
)

print(
    f"Brier     : {brier:.6f}"
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print("\n                Predicted")
print("               0       1")
print(
    f"Actual 0    {tn:4d}    {fp:4d}"
)

print(
    f"Actual 1    {fn:4d}    {tp:4d}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 19. SURFACE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("SURFACE-WISE PERFORMANCE")
print("=" * 75)

# Try to locate the surface column.
surface_column = None

possible_surface_columns = [
    "surface",
    "Surface",
    "SURFACE"
]

for column in possible_surface_columns:

    if column in test_df.columns:

        surface_column = column
        break


if surface_column is None:

    print(
        "\nSurface column was not found."
    )

else:

    print(
        f"\nUsing surface column: "
        f"{surface_column}"
    )

    surface_results = []

    for surface in sorted(
        test_df[surface_column].dropna().unique()
    ):

        mask = (
            test_df[surface_column] == surface
        )

        surface_y = y_test[mask]
        surface_predictions = test_predictions[mask]
        surface_probabilities = test_probabilities[mask]

        surface_accuracy = accuracy_score(
            surface_y,
            surface_predictions
        )

        surface_f1 = f1_score(
            surface_y,
            surface_predictions,
            zero_division=0
        )

        # ROC-AUC requires both classes to exist.
        if len(np.unique(surface_y)) == 2:

            surface_auc = roc_auc_score(
                surface_y,
                surface_probabilities
            )

        else:

            surface_auc = np.nan

        surface_results.append({
            "surface": surface,
            "matches": len(surface_y),
            "accuracy": surface_accuracy,
            "f1": surface_f1,
            "roc_auc": surface_auc
        })

        print(
            f"\n{surface}:"
        )

        print(
            f"Matches  : {len(surface_y)}"
        )

        print(
            f"Accuracy : {surface_accuracy:.6f}"
        )

        print(
            f"F1       : {surface_f1:.6f}"
        )

        if not np.isnan(surface_auc):

            print(
                f"ROC-AUC  : {surface_auc:.6f}"
            )


    surface_results_df = pd.DataFrame(
        surface_results
    )

    surface_results_path = os.path.join(
        OUTPUT_DIR,
        "rf_final_surface_results.csv"
    )

    surface_results_df.to_csv(
        surface_results_path,
        index=False
    )


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df[
    "predicted_probability"
] = test_probabilities

predictions_df[
    "predicted_class"
] = test_predictions

predictions_df[
    "prediction_threshold"
] = THRESHOLD

predictions_path = os.path.join(
    OUTPUT_DIR,
    "rf_final_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 21. SAVE FINAL SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "rf_final_test_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "FINAL RANDOM FOREST TEST EVALUATION\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Training period: Train + Validation\n"
    )

    f.write(
        "Test period: 2024\n\n"
    )

    f.write(
        "Features: 22\n"
    )

    f.write(
        f"Threshold: {THRESHOLD}\n\n"
    )

    f.write(
        "Random Forest parameters:\n"
    )

    for parameter, value in MODEL_PARAMS.items():

        f.write(
            f"{parameter} = {value}\n"
        )

    f.write("\n")

    f.write(
        "CLASSIFICATION METRICS\n"
    )

    f.write(
        f"Accuracy = {accuracy:.6f}\n"
    )

    f.write(
        f"Precision = {precision:.6f}\n"
    )

    f.write(
        f"Recall = {recall:.6f}\n"
    )

    f.write(
        f"F1 = {f1:.6f}\n\n"
    )

    f.write(
        "PROBABILITY METRICS\n"
    )

    f.write(
        f"ROC-AUC = {roc_auc:.6f}\n"
    )

    f.write(
        f"Log Loss = {logloss:.6f}\n"
    )

    f.write(
        f"Brier Score = {brier:.6f}\n\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        f"TN = {tn}\n"
    )

    f.write(
        f"FP = {fp}\n"
    )

    f.write(
        f"FN = {fn}\n"
    )

    f.write(
        f"TP = {tp}\n"
    )


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("FINAL EVALUATION COMPLETE")
print("=" * 75)

print("\nSaved files:")

print(
    f"\nFinal predictions:"
    f"\n{predictions_path}"
)

print(
    f"\nFinal summary:"
    f"\n{summary_path}"
)

if surface_column is not None:

    print(
        f"\nSurface results:"
        f"\n{surface_results_path}"
    )

print("\n" + "=" * 75)
print("IMPORTANT")
print("=" * 75)
