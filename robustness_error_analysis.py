"""
ATP TENNIS MATCH PREDICTION
ROBUSTNESS, ERROR & PROBABILITY ANALYSIS

Purpose:
1. Evaluate final Random Forest by court surface
2. Analyze prediction errors
3. Analyze prediction confidence
4. Check probability calibration
5. Calculate Brier score, Log Loss and Expected Calibration Error
6. Save detailed results and figures

IMPORTANT:
- Test data is NEVER used to train the model.
- Validation data is used only for probability calibration.
- Test data is used only for final evaluation.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "robustness")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ------------------------------------------------------------
# DATASET PATHS
# ------------------------------------------------------------
# Change these ONLY if your prepared dataset filenames are
# different.

POSSIBLE_TRAIN_PATHS = [
    os.path.join(DATA_DIR, "prepared", "train.csv"),
    os.path.join(DATA_DIR, "processed", "train.csv"),
    os.path.join(DATA_DIR, "train.csv"),
]

POSSIBLE_VALIDATION_PATHS = [
    os.path.join(DATA_DIR, "prepared", "validation.csv"),
    os.path.join(DATA_DIR, "processed", "validation.csv"),
    os.path.join(DATA_DIR, "validation.csv"),
    os.path.join(DATA_DIR, "val.csv"),
]

POSSIBLE_TEST_PATHS = [
    os.path.join(DATA_DIR, "prepared", "test.csv"),
    os.path.join(DATA_DIR, "processed", "test.csv"),
    os.path.join(DATA_DIR, "test.csv"),
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_existing_path(paths, dataset_name):
    """
    Find the first existing path from a list.
    """

    for path in paths:
        if os.path.exists(path):
            print(f"{dataset_name} dataset found:")
            print(path)
            print()
            return path

    print(f"\nERROR: Could not find {dataset_name} dataset.")
    print("Checked:")
    for path in paths:
        print("  ", path)

    raise FileNotFoundError(
        f"{dataset_name} dataset could not be located."
    )


def safe_roc_auc(y_true, probabilities):
    """
    ROC-AUC can fail if a subgroup contains only one class.
    """

    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        return roc_auc_score(y_true, probabilities)

    except Exception:
        return np.nan


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Lower ECE = better calibrated probabilities.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    rows = []

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (probabilities >= bin_edges[i]) &
                (probabilities <= bin_edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[i]) &
                (probabilities < bin_edges[i + 1])
            )

        count = np.sum(mask)

        if count == 0:
            continue

        mean_probability = probabilities[mask].mean()
        actual_rate = y_true[mask].mean()

        gap = abs(mean_probability - actual_rate)

        ece += (count / len(y_true)) * gap

        rows.append({
            "bin": i + 1,
            "lower": bin_edges[i],
            "upper": bin_edges[i + 1],
            "count": count,
            "mean_probability": mean_probability,
            "actual_positive_rate": actual_rate,
            "absolute_gap": gap
        })

    return ece, pd.DataFrame(rows)


def print_metrics(name, y_true, predictions, probabilities):

    accuracy = accuracy_score(y_true, predictions)

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted"
    )

    roc_auc = safe_roc_auc(
        y_true,
        probabilities
    )

    loss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1]
    )

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"Log Loss : {loss:.4f}")
    print(f"Brier    : {brier:.4f}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss,
        "brier_score": brier
    }


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("ATP TENNIS ROBUSTNESS & ERROR ANALYSIS")
print("=" * 70)
print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")
print()

train_path = find_existing_path(
    POSSIBLE_TRAIN_PATHS,
    "TRAIN"
)

validation_path = find_existing_path(
    POSSIBLE_VALIDATION_PATHS,
    "VALIDATION"
)

test_path = find_existing_path(
    POSSIBLE_TEST_PATHS,
    "TEST"
)


train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)


print("Dataset sizes:")
print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(validation_df):,}")
print(f"Test       : {len(test_df):,}")
print()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

TARGET = "target"
SURFACE = "surface"


required_columns = [
    TARGET,
    SURFACE
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Required column '{column}' missing from TRAIN."
        )

    if column not in validation_df.columns:
        raise ValueError(
            f"Required column '{column}' missing from VALIDATION."
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Required column '{column}' missing from TEST."
        )


# ============================================================
# FEATURE SELECTION
# ============================================================

# Columns that should NEVER be used as predictors.
EXCLUDE_COLUMNS = {
    TARGET,
    SURFACE,

    # identifiers
    "id",
    "match_id",

    # player names / direct identifiers
    "player1",
    "player2",
    "player_1",
    "player_2",

    # outcome-related columns
    "winner",
    "loser",
    "winner_name",
    "loser_name"
}


feature_columns = [
    column
    for column in train_df.columns
    if column not in EXCLUDE_COLUMNS
    and pd.api.types.is_numeric_dtype(train_df[column])
]


print("=" * 70)
print("FEATURE SET")
print("=" * 70)

print(f"Number of features: {len(feature_columns)}")
print()

for feature in feature_columns:
    print(" -", feature)

print()


# ============================================================
# VERIFY FEATURE CONSISTENCY
# ============================================================

missing_train = [
    f for f in feature_columns
    if f not in train_df.columns
]

missing_validation = [
    f for f in feature_columns
    if f not in validation_df.columns
]

missing_test = [
    f for f in feature_columns
    if f not in test_df.columns
]

if missing_train or missing_validation or missing_test:

    print("ERROR: Feature mismatch.")

    if missing_train:
        print("Missing from train:", missing_train)

    if missing_validation:
        print("Missing from validation:", missing_validation)

    if missing_test:
        print("Missing from test:", missing_test)

    raise ValueError("Dataset feature mismatch.")


# ============================================================
# PREPARE X AND y
# ============================================================

X_train = train_df[feature_columns].copy()
y_train = train_df[TARGET].astype(int)

X_validation = validation_df[feature_columns].copy()
y_validation = validation_df[TARGET].astype(int)

X_test = test_df[feature_columns].copy()
y_test = test_df[TARGET].astype(int)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print(
    "Train missing values      :",
    X_train.isna().sum().sum()
)

print(
    "Validation missing values :",
    X_validation.isna().sum().sum()
)

print(
    "Test missing values       :",
    X_test.isna().sum().sum()
)

if (
    X_train.isna().sum().sum() > 0
    or X_validation.isna().sum().sum() > 0
    or X_test.isna().sum().sum() > 0
):

    raise ValueError(
        "Missing feature values detected."
    )

print()


# ============================================================
# BUILD FINAL RANDOM FOREST
# ============================================================

print("=" * 70)
print("BUILDING FINAL RANDOM FOREST")
print("=" * 70)

print()
print("Using the best parameters from your previous tuning:")
print()
print("n_estimators     = 300")
print("max_depth        = 10")
print("min_samples_split = 10")
print("min_samples_leaf = 5")
print("max_features     = 0.5")
print()


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)


print("Training Random Forest...")
model.fit(
    X_train,
    y_train
)

print("Training completed.")
print()


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_probability = model.predict_proba(
    X_validation
)[:, 1]

validation_prediction = (
    validation_probability >= 0.5
).astype(int)


validation_metrics = print_metrics(
    "RANDOM FOREST - VALIDATION",
    y_validation,
    validation_prediction,
    validation_probability
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_probability = model.predict_proba(
    X_test
)[:, 1]

test_prediction = (
    test_probability >= 0.5
).astype(int)


test_metrics = print_metrics(
    "RANDOM FOREST - TEST",
    y_test,
    test_prediction,
    test_probability
)


# ============================================================
# CREATE TEST PREDICTION DATAFRAME
# ============================================================

predictions_df = test_df.copy()

predictions_df["predicted_probability_player1"] = test_probability

predictions_df["predicted_probability_player2"] = (
    1 - test_probability
)

predictions_df["predicted_class"] = test_prediction

predictions_df["correct_prediction"] = (
    test_prediction == y_test.values
)

predictions_df["confidence"] = np.maximum(
    test_probability,
    1 - test_probability
)

predictions_df["prediction_margin"] = (
    np.abs(test_probability - 0.5) * 2
)


predictions_path = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print()
print("Test predictions saved:")
print(predictions_path)


# ============================================================
# 1. SURFACE-WISE PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("SURFACE-WISE PERFORMANCE")
print("=" * 70)

surface_results = []


for surface in sorted(
    test_df[SURFACE].dropna().unique()
):

    mask = (
        test_df[SURFACE] == surface
    )

    y_surface = y_test[mask]

    pred_surface = test_prediction[mask]

    prob_surface = test_probability[mask]

    accuracy = accuracy_score(
        y_surface,
        pred_surface
    )

    f1 = f1_score(
        y_surface,
        pred_surface,
        average="weighted"
    )

    roc_auc = safe_roc_auc(
        y_surface,
        prob_surface
    )

    loss = log_loss(
        y_surface,
        prob_surface,
        labels=[0, 1]
    )

    brier = brier_score_loss(
        y_surface,
        prob_surface
    )

    surface_results.append({
        "surface": surface,
        "matches": len(y_surface),
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss,
        "brier_score": brier
    })


surface_results_df = pd.DataFrame(
    surface_results
)


print(
    surface_results_df.to_string(
        index=False
    )
)


surface_results_path = os.path.join(
    RESULTS_DIR,
    "surface_performance.csv"
)

surface_results_df.to_csv(
    surface_results_path,
    index=False
)


# ============================================================
# SURFACE ACCURACY PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    surface_results_df["surface"].astype(str),
    surface_results_df["accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Court Surface")
plt.title("Random Forest Accuracy by Court Surface")

plt.ylim(0, 100)

for i, value in enumerate(
    surface_results_df["accuracy"] * 100
):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "surface_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SURFACE ROC-AUC PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    surface_results_df["surface"].astype(str),
    surface_results_df["roc_auc"]
)

plt.ylabel("ROC-AUC")
plt.xlabel("Court Surface")
plt.title("Random Forest ROC-AUC by Court Surface")

plt.ylim(0.5, 1.0)

for i, value in enumerate(
    surface_results_df["roc_auc"]
):

    if not np.isnan(value):

        plt.text(
            i,
            value + 0.015,
            f"{value:.3f}",
            ha="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "surface_roc_auc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. ERROR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)


incorrect_df = predictions_df[
    predictions_df["correct_prediction"] == False
].copy()


correct_df = predictions_df[
    predictions_df["correct_prediction"] == True
].copy()


print()
print(f"Correct predictions     : {len(correct_df):,}")
print(f"Incorrect predictions   : {len(incorrect_df):,}")

print()
print(
    f"Error rate              : "
    f"{len(incorrect_df) / len(predictions_df):.4f}"
)


# ------------------------------------------------------------
# MOST CONFIDENT WRONG PREDICTIONS
# ------------------------------------------------------------

high_confidence_errors = incorrect_df.sort_values(
    "confidence",
    ascending=False
)


high_confidence_errors_path = os.path.join(
    RESULTS_DIR,
    "high_confidence_errors.csv"
)

high_confidence_errors.to_csv(
    high_confidence_errors_path,
    index=False
)


print()
print("Most confident incorrect predictions:")

display_columns = [
    SURFACE,
    TARGET,
    "predicted_class",
    "predicted_probability_player1",
    "confidence",
    "prediction_margin"
]

available_columns = [
    column
    for column in display_columns
    if column in high_confidence_errors.columns
]

print(
    high_confidence_errors[
        available_columns
    ].head(20).to_string(index=False)
)


# ------------------------------------------------------------
# ALL MISCLASSIFIED MATCHES
# ------------------------------------------------------------

misclassified_path = os.path.join(
    RESULTS_DIR,
    "misclassified_matches.csv"
)

incorrect_df.to_csv(
    misclassified_path,
    index=False
)


# ============================================================
# 3. CONFIDENCE ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)


# Confidence is always between 0.5 and 1.0.

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


predictions_df["confidence_bin"] = pd.cut(
    predictions_df["confidence"],
    bins=confidence_bins,
    include_lowest=True,
    right=True
)


confidence_results = []


for confidence_bin, group in predictions_df.groupby(
    "confidence_bin",
    observed=False
):

    if len(group) == 0:
        continue

    accuracy = group[
        "correct_prediction"
    ].mean()

    confidence_results.append({
        "confidence_bin": str(confidence_bin),
        "matches": len(group),
        "average_confidence": group[
            "confidence"
        ].mean(),
        "accuracy": accuracy,
        "error_rate": 1 - accuracy
    })


confidence_results_df = pd.DataFrame(
    confidence_results
)


print()
print(
    confidence_results_df.to_string(
        index=False
    )
)


confidence_results_path = os.path.join(
    RESULTS_DIR,
    "confidence_analysis.csv"
)

confidence_results_df.to_csv(
    confidence_results_path,
    index=False
)


# ============================================================
# CONFIDENCE VS ACCURACY PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    confidence_results_df["average_confidence"],
    confidence_results_df["accuracy"],
    marker="o",
    linewidth=2
)

plt.plot(
    [0.5, 1.0],
    [0.5, 1.0],
    linestyle="--"
)

plt.xlabel("Average Prediction Confidence")
plt.ylabel("Actual Accuracy")
plt.title("Prediction Confidence vs Actual Accuracy")

plt.xlim(0.5, 1.0)
plt.ylim(0.5, 1.0)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "confidence_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. PROBABILITY CALIBRATION
# ============================================================

print()
print("=" * 70)
print("PROBABILITY CALIBRATION")
print("=" * 70)


# ------------------------------------------------------------
# TEST CALIBRATION METRICS
# ------------------------------------------------------------

test_ece, calibration_bins = (
    expected_calibration_error(
        y_test,
        test_probability,
        n_bins=10
    )
)


print()
print(f"Test Brier Score : {test_metrics['brier_score']:.4f}")
print(f"Test Log Loss    : {test_metrics['log_loss']:.4f}")
print(f"Test ECE         : {test_ece:.4f}")


# ============================================================
# CALIBRATION CURVE
# ============================================================

fraction_positive, mean_predicted = (
    calibration_curve(
        y_test,
        test_probability,
        n_bins=10,
        strategy="uniform"
    )
)


plt.figure(figsize=(8, 8))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    linewidth=2,
    label="Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Random Forest Probability Calibration")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "calibration_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE CALIBRATION DATA
# ============================================================

calibration_metrics_df = pd.DataFrame({
    "metric": [
        "Brier Score",
        "Log Loss",
        "Expected Calibration Error"
    ],
    "value": [
        test_metrics["brier_score"],
        test_metrics["log_loss"],
        test_ece
    ]
})


calibration_metrics_path = os.path.join(
    RESULTS_DIR,
    "calibration_metrics.csv"
)

calibration_metrics_df.to_csv(
    calibration_metrics_path,
    index=False
)


# ============================================================
# CALIBRATION BIN TABLE
# ============================================================

calibration_bins_path = os.path.join(
    RESULTS_DIR,
    "calibration_bins.csv"
)

calibration_bins.to_csv(
    calibration_bins_path,
    index=False
)


print()
print("Calibration bins:")
print(
    calibration_bins.to_string(
        index=False
    )
)


# ============================================================
# VALIDATION-BASED PROBABILITY CALIBRATION
# ============================================================

print()
print("=" * 70)
print("VALIDATION-BASED PROBABILITY CALIBRATION")
print("=" * 70)

print()
print(
    "The validation set will be used to learn "
    "a probability calibration mapping."
)

print(
    "The test set remains untouched for final evaluation."
)


# ------------------------------------------------------------
# METHOD 1: PLATT / SIGMOID CALIBRATION
# ------------------------------------------------------------

platt_calibrator = LogisticRegression(
    solver="lbfgs"
)

validation_prob_2d = (
    validation_probability.reshape(-1, 1)
)

platt_calibrator.fit(
    validation_prob_2d,
    y_validation
)


calibrated_test_probability = (
    platt_calibrator.predict_proba(
        test_probability.reshape(-1, 1)
    )[:, 1]
)


calibrated_test_prediction = (
    calibrated_test_probability >= 0.5
).astype(int)


calibrated_metrics = print_metrics(
    "CALIBRATED RANDOM FOREST - TEST",
    y_test,
    calibrated_test_prediction,
    calibrated_test_probability
)


# ============================================================
# COMPARE RAW VS CALIBRATED PROBABILITIES
# ============================================================

raw_brier = brier_score_loss(
    y_test,
    test_probability
)

calibrated_brier = brier_score_loss(
    y_test,
    calibrated_test_probability
)

raw_log_loss = log_loss(
    y_test,
    test_probability,
    labels=[0, 1]
)

calibrated_log_loss = log_loss(
    y_test,
    calibrated_test_probability,
    labels=[0, 1]
)


calibration_comparison = pd.DataFrame({

    "model": [
        "Raw Random Forest",
        "Calibrated Random Forest"
    ],

    "accuracy": [
        test_metrics["accuracy"],
        calibrated_metrics["accuracy"]
    ],

    "f1": [
        test_metrics["f1"],
        calibrated_metrics["f1"]
    ],

    "roc_auc": [
        test_metrics["roc_auc"],
        calibrated_metrics["roc_auc"]
    ],

    "log_loss": [
        raw_log_loss,
        calibrated_log_loss
    ],

    "brier_score": [
        raw_brier,
        calibrated_brier
    ]
})


print()
print("=" * 70)
print("RAW VS CALIBRATED PROBABILITIES")
print("=" * 70)

print(
    calibration_comparison.to_string(
        index=False
    )
)


calibration_comparison_path = os.path.join(
    RESULTS_DIR,
    "raw_vs_calibrated.csv"
)

calibration_comparison.to_csv(
    calibration_comparison_path,
    index=False
)


# ============================================================
# SAVE CALIBRATED PREDICTIONS
# ============================================================

predictions_df[
    "calibrated_probability_player1"
] = calibrated_test_probability

predictions_df[
    "calibrated_prediction"
] = calibrated_test_prediction


predictions_df[
    "calibrated_correct_prediction"
] = (
    calibrated_test_prediction == y_test.values
)


predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("ROBUSTNESS & ERROR ANALYSIS SUMMARY")
print("=" * 70)

print()

print("FINAL RANDOM FOREST")
print("-------------------")

print(
    f"Test Accuracy : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test F1       : "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"Test ROC-AUC  : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"Test Log Loss : "
    f"{test_metrics['log_loss']:.4f}"
)

print(
    f"Test Brier    : "
    f"{test_metrics['brier_score']:.4f}"
)

print(
    f"Test ECE      : "
    f"{test_ece:.4f}"
)

print()

print(
    f"Correct predictions   : "
    f"{len(correct_df):,}"
)

print(
    f"Incorrect predictions : "
    f"{len(incorrect_df):,}"
)

print()

print("Output directory:")
print(RESULTS_DIR)

print()

print("=" * 70)
print("ROBUSTNESS ANALYSIS COMPLETED")
print("=" * 70)