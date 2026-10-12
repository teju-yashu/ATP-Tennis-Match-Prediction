# rf_error_analysis.py

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "feature_analysis",
    "rf_final_test_predictions.csv"
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
# 2. LOAD FINAL PREDICTIONS
# ============================================================

print("=" * 75)
print("FINAL RANDOM FOREST - ERROR ANALYSIS")
print("=" * 75)

print("\nLoading final predictions:")
print(PREDICTIONS_PATH)

if not os.path.exists(PREDICTIONS_PATH):
    raise FileNotFoundError(
        f"\nPrediction file not found:\n{PREDICTIONS_PATH}"
    )

df = pd.read_csv(PREDICTIONS_PATH)

print(f"\nRows loaded: {len(df):,}")


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "target",
    "predicted_probability",
    "predicted_class",
    "prediction_threshold"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 4. BASIC VARIABLES
# ============================================================

y_true = df["target"].astype(int)

y_pred = df["predicted_class"].astype(int)

probabilities = df["predicted_probability"].astype(float)

threshold = float(
    df["prediction_threshold"].iloc[0]
)

# Whether prediction was correct
df["correct"] = (
    df["target"] == df["predicted_class"]
).astype(int)

# Confidence of the predicted class
#
# If predicted class = 1:
# confidence = P(class 1)
#
# If predicted class = 0:
# confidence = 1 - P(class 1)

df["confidence"] = np.where(
    df["predicted_class"] == 1,
    df["predicted_probability"],
    1 - df["predicted_probability"]
)

df["error"] = (
    df["correct"] == 0
).astype(int)


# ============================================================
# 5. OVERALL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    probabilities
)

logloss = log_loss(
    y_true,
    probabilities
)

brier = brier_score_loss(
    y_true,
    probabilities
)


# ============================================================
# 6. BASIC COUNTS
# ============================================================

total_matches = len(df)

correct_predictions = int(
    df["correct"].sum()
)

incorrect_predictions = (
    total_matches -
    correct_predictions
)

predicted_class_0 = int(
    (y_pred == 0).sum()
)

predicted_class_1 = int(
    (y_pred == 1).sum()
)

actual_class_0 = int(
    (y_true == 0).sum()
)

actual_class_1 = int(
    (y_true == 1).sum()
)


# ============================================================
# 7. PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("OVERALL PERFORMANCE")
print("=" * 75)

print(f"\nThreshold              : {threshold:.2f}")
print(f"Total matches          : {total_matches}")
print(f"Correct predictions    : {correct_predictions}")
print(f"Incorrect predictions  : {incorrect_predictions}")

print(
    f"Error rate             : "
    f"{incorrect_predictions / total_matches:.4f}"
)

print("\nClassification metrics:")

print(
    f"Accuracy               : {accuracy:.6f}"
)

print(
    f"Precision              : {precision:.6f}"
)

print(
    f"Recall                 : {recall:.6f}"
)

print(
    f"F1                     : {f1:.6f}"
)

print("\nProbability metrics:")

print(
    f"ROC-AUC                : {roc_auc:.6f}"
)

print(
    f"Log Loss               : {logloss:.6f}"
)

print(
    f"Brier Score            : {brier:.6f}"
)


# ============================================================
# 8. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("PREDICTION DISTRIBUTION")
print("=" * 75)

print(
    f"\nActual class 0         : "
    f"{actual_class_0}"
)

print(
    f"Actual class 1         : "
    f"{actual_class_1}"
)

print(
    f"Predicted class 0      : "
    f"{predicted_class_0}"
)

print(
    f"Predicted class 1      : "
    f"{predicted_class_1}"
)

print(
    f"Actual class 1 rate    : "
    f"{actual_class_1 / total_matches:.4f}"
)

print(
    f"Predicted class 1 rate : "
    f"{predicted_class_1 / total_matches:.4f}"
)


# ============================================================
# 9. CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("CONFIDENCE ANALYSIS")
print("=" * 75)

confidence_bins = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00
]

confidence_results = []

for lower, upper in zip(
    confidence_bins[:-1],
    confidence_bins[1:]
):

    if upper == 1.00:

        mask = (
            (df["confidence"] >= lower) &
            (df["confidence"] <= upper)
        )

    else:

        mask = (
            (df["confidence"] >= lower) &
            (df["confidence"] < upper)
        )

    subset = df[mask]

    count = len(subset)

    if count > 0:

        accuracy_bin = subset["correct"].mean()

        avg_confidence = subset[
            "confidence"
        ].mean()

        errors = (
            1 -
            subset["correct"].mean()
        )

    else:

        accuracy_bin = np.nan
        avg_confidence = np.nan
        errors = np.nan

    confidence_results.append({
        "confidence_range":
            f"{lower:.2f}-{upper:.2f}",
        "matches": count,
        "average_confidence":
            avg_confidence,
        "accuracy":
            accuracy_bin,
        "error_rate":
            errors
    })


confidence_df = pd.DataFrame(
    confidence_results
)


print(
    "\n"
    + confidence_df.to_string(
        index=False,
        formatters={
            "average_confidence":
                lambda x: f"{x:.4f}"
                if pd.notna(x)
                else "N/A",

            "accuracy":
                lambda x: f"{x:.4f}"
                if pd.notna(x)
                else "N/A",

            "error_rate":
                lambda x: f"{x:.4f}"
                if pd.notna(x)
                else "N/A"
        }
    )
)


# ============================================================
# 10. HIGH-CONFIDENCE ERRORS
# ============================================================

print("\n" + "=" * 75)
print("HIGH-CONFIDENCE ERROR ANALYSIS")
print("=" * 75)

confidence_levels = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

high_confidence_summary = []

for level in confidence_levels:

    high_confidence = df[
        df["confidence"] >= level
    ]

    high_confidence_errors = high_confidence[
        high_confidence["correct"] == 0
    ]

    count = len(high_confidence)

    errors = len(
        high_confidence_errors
    )

    if count > 0:

        error_rate = errors / count

    else:

        error_rate = np.nan

    high_confidence_summary.append({
        "confidence_threshold": level,
        "predictions": count,
        "errors": errors,
        "error_rate": error_rate
    })

    print(
        f"\nConfidence >= {level:.2f}"
    )

    print(
        f"Predictions : {count}"
    )

    print(
        f"Errors      : {errors}"
    )

    if count > 0:

        print(
            f"Error rate  : "
            f"{error_rate:.4f}"
        )


high_confidence_df = pd.DataFrame(
    high_confidence_summary
)


# ============================================================
# 11. WORST HIGH-CONFIDENCE ERRORS
# ============================================================

print("\n" + "=" * 75)
print("WORST HIGH-CONFIDENCE ERRORS")
print("=" * 75)

error_df = df[
    df["correct"] == 0
].copy()

error_df = error_df.sort_values(
    "confidence",
    ascending=False
)

columns_to_show = [
    column
    for column in [
        "match_date",
        "tourney_name",
        "surface",
        "player1_id",
        "player2_id",
        "target",
        "predicted_class",
        "predicted_probability",
        "confidence",
        "rank_difference",
        "rank_points_difference",
        "elo_difference",
        "surface_elo_difference"
    ]
    if column in error_df.columns
]

worst_errors = error_df[
    columns_to_show
].head(20)


print(
    "\nTop 20 highest-confidence incorrect predictions:\n"
)

print(
    worst_errors.to_string(
        index=False
    )
)


# ============================================================
# 12. SURFACE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("SURFACE ERROR ANALYSIS")
print("=" * 75)

surface_results = []

if "surface" in df.columns:

    for surface in sorted(
        df["surface"].dropna().unique()
    ):

        subset = df[
            df["surface"] == surface
        ]

        surface_accuracy = (
            subset["correct"].mean()
        )

        surface_errors = (
            1 -
            surface_accuracy
        )

        surface_auc = roc_auc_score(
            subset["target"],
            subset["predicted_probability"]
        )

        surface_results.append({
            "surface": surface,
            "matches": len(subset),
            "correct": int(
                subset["correct"].sum()
            ),
            "errors": int(
                (1 - subset["correct"]).sum()
            ),
            "accuracy": surface_accuracy,
            "error_rate": surface_errors,
            "roc_auc": surface_auc
        })

        print(
            f"\n{surface}"
        )

        print(
            f"Matches     : {len(subset)}"
        )

        print(
            f"Correct     : "
            f"{int(subset['correct'].sum())}"
        )

        print(
            f"Errors      : "
            f"{int((1 - subset['correct']).sum())}"
        )

        print(
            f"Accuracy    : "
            f"{surface_accuracy:.6f}"
        )

        print(
            f"Error rate  : "
            f"{surface_errors:.6f}"
        )

        print(
            f"ROC-AUC     : "
            f"{surface_auc:.6f}"
        )

else:

    print(
        "\nSurface column not found."
    )


surface_df = pd.DataFrame(
    surface_results
)


# ============================================================
# 13. RANK-DIFFERENCE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("RANK-DIFFERENCE ANALYSIS")
print("=" * 75)

rank_results = []

if "rank_difference" in df.columns:

    df["rank_difference_group"] = pd.cut(
        df["rank_difference"],
        bins=[
            -np.inf,
            -50,
            -20,
            -10,
            0,
            10,
            20,
            50,
            np.inf
        ],
        labels=[
            "< -50",
            "-50 to -20",
            "-20 to -10",
            "-10 to 0",
            "0 to 10",
            "10 to 20",
            "20 to 50",
            "> 50"
        ]
    )

    grouped = df.groupby(
        "rank_difference_group",
        observed=True
    )

    for group, subset in grouped:

        if len(subset) == 0:
            continue

        group_accuracy = (
            subset["correct"].mean()
        )

        rank_results.append({
            "rank_difference_group": str(group),
            "matches": len(subset),
            "accuracy": group_accuracy,
            "error_rate":
                1 - group_accuracy
        })

        print(
            f"\nRank difference {group}"
        )

        print(
            f"Matches    : {len(subset)}"
        )

        print(
            f"Accuracy   : {group_accuracy:.6f}"
        )

        print(
            f"Error rate : "
            f"{1 - group_accuracy:.6f}"
        )

else:

    print(
        "\nrank_difference column not found."
    )


rank_df = pd.DataFrame(
    rank_results
)


# ============================================================
# 14. ECE - EXPECTED CALIBRATION ERROR
# ============================================================

print("\n" + "=" * 75)
print("CALIBRATION ERROR")
print("=" * 75)

# Confidence of predicted class
# compared against whether prediction was correct.

ece = 0.0

ece_bins = []

for lower, upper in zip(
    confidence_bins[:-1],
    confidence_bins[1:]
):

    if upper == 1.00:

        mask = (
            (df["confidence"] >= lower) &
            (df["confidence"] <= upper)
        )

    else:

        mask = (
            (df["confidence"] >= lower) &
            (df["confidence"] < upper)
        )

    subset = df[mask]

    if len(subset) == 0:
        continue

    mean_confidence = (
        subset["confidence"].mean()
    )

    observed_accuracy = (
        subset["correct"].mean()
    )

    proportion = (
        len(subset) /
        total_matches
    )

    contribution = (
        proportion *
        abs(
            mean_confidence -
            observed_accuracy
        )
    )

    ece += contribution

    ece_bins.append({
        "confidence_range":
            f"{lower:.2f}-{upper:.2f}",
        "matches": len(subset),
        "mean_confidence":
            mean_confidence,
        "observed_accuracy":
            observed_accuracy,
        "absolute_difference":
            abs(
                mean_confidence -
                observed_accuracy
            ),
        "ece_contribution":
            contribution
    })


ece_df = pd.DataFrame(
    ece_bins
)

print(
    f"\nExpected Calibration Error: "
    f"{ece:.6f}"
)


# ============================================================
# 15. SAVE FILES
# ============================================================

confidence_path = os.path.join(
    OUTPUT_DIR,
    "rf_confidence_analysis.csv"
)

high_confidence_path = os.path.join(
    OUTPUT_DIR,
    "rf_high_confidence_errors.csv"
)

surface_path = os.path.join(
    OUTPUT_DIR,
    "rf_surface_error_analysis.csv"
)

rank_path = os.path.join(
    OUTPUT_DIR,
    "rf_rank_difference_analysis.csv"
)

worst_errors_path = os.path.join(
    OUTPUT_DIR,
    "rf_worst_errors.csv"
)

ece_path = os.path.join(
    OUTPUT_DIR,
    "rf_ece_analysis.csv"
)

confidence_df.to_csv(
    confidence_path,
    index=False
)

high_confidence_df.to_csv(
    high_confidence_path,
    index=False
)

surface_df.to_csv(
    surface_path,
    index=False
)

rank_df.to_csv(
    rank_path,
    index=False
)

worst_errors.to_csv(
    worst_errors_path,
    index=False
)

ece_df.to_csv(
    ece_path,
    index=False
)


# ============================================================
# 16. SAVE TEXT SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "rf_error_analysis_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "FINAL RANDOM FOREST ERROR ANALYSIS\n"
    )

    f.write(
        "=" * 65 + "\n\n"
    )

    f.write(
        f"Threshold = {threshold:.2f}\n"
    )

    f.write(
        f"Total matches = {total_matches}\n"
    )

    f.write(
        f"Correct predictions = "
        f"{correct_predictions}\n"
    )

    f.write(
        f"Incorrect predictions = "
        f"{incorrect_predictions}\n\n"
    )

    f.write(
        "OVERALL METRICS\n"
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
        f"F1 = {f1:.6f}\n"
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
        "EXPECTED CALIBRATION ERROR\n"
    )

    f.write(
        f"ECE = {ece:.6f}\n\n"
    )

    f.write(
        "HIGH-CONFIDENCE ERRORS\n"
    )

    for _, row in high_confidence_df.iterrows():

        f.write(
            f"Confidence >= "
            f"{row['confidence_threshold']:.2f}: "
            f"{int(row['errors'])} errors / "
            f"{int(row['predictions'])} predictions"
            f"\n"
        )


# ============================================================
# 17. DONE
# ============================================================

print("\n" + "=" * 75)
print("ERROR ANALYSIS COMPLETE")
print("=" * 75)

print("\nFiles saved:")

print(
    f"\nConfidence analysis:"
    f"\n{confidence_path}"
)

print(
    f"\nHigh-confidence errors:"
    f"\n{high_confidence_path}"
)

print(
    f"\nSurface analysis:"
    f"\n{surface_path}"
)

print(
    f"\nRank-difference analysis:"
    f"\n{rank_path}"
)

print(
    f"\nWorst errors:"
    f"\n{worst_errors_path}"
)

print(
    f"\nECE analysis:"
    f"\n{ece_path}"
)

print(
    f"\nComplete summary:"
    f"\n{summary_path}"
)
