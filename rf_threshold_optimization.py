# rf_threshold_optimization.py

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# Script is being run from:
# miniproject/src/
#
# Data is located in:
# miniproject/data/

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "validation.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "feature_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. FINAL 22 FEATURES
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
# 3. TARGET
# ============================================================

TARGET = "target"


# ============================================================
# 4. START
# ============================================================

print("=" * 70)
print("THRESHOLD OPTIMIZATION - RANDOM FOREST")
print("=" * 70)

print("\nProject directory:")
print(BASE_DIR)

print("\nTraining file:")
print(TRAIN_PATH)

print("\nValidation file:")
print(VAL_PATH)


# ============================================================
# 5. CHECK FILES EXIST
# ============================================================

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"\nTraining file not found:\n{TRAIN_PATH}"
    )

if not os.path.exists(VAL_PATH):
    raise FileNotFoundError(
        f"\nValidation file not found:\n{VAL_PATH}"
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"\nTraining rows   : {len(train_df):,}")
print(f"Validation rows : {len(val_df):,}")
print(f"Number features : {len(FEATURES)}")


# ============================================================
# 7. CHECK FEATURES
# ============================================================

missing_train = [
    feature for feature in FEATURES
    if feature not in train_df.columns
]

missing_val = [
    feature for feature in FEATURES
    if feature not in val_df.columns
]

if missing_train:
    raise ValueError(
        "\nMissing features in training data:\n"
        + "\n".join(missing_train)
    )

if missing_val:
    raise ValueError(
        "\nMissing features in validation data:\n"
        + "\n".join(missing_val)
    )


# ============================================================
# 8. CHECK TARGET
# ============================================================

if TARGET not in train_df.columns:
    raise ValueError(
        f"\nTarget column '{TARGET}' not found in training data."
    )

if TARGET not in val_df.columns:
    raise ValueError(
        f"\nTarget column '{TARGET}' not found in validation data."
    )


# ============================================================
# 9. CREATE X AND y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

print("\nParameters:")
print("n_estimators      = 300")
print("max_depth         = 10")
print("min_samples_split = 10")
print("min_samples_leaf  = 5")
print("max_features      = 0.5")
print("random_state      = 42")

print("\nTraining...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 11. VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

val_probabilities = model.predict_proba(X_val)[:, 1]

print("Probabilities generated.")


# ============================================================
# 12. THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

thresholds = np.arange(0.30, 0.701, 0.01)

results = []

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    f1 = f1_score(
        y_val,
        predictions
    )

    results.append({
        "threshold": round(threshold, 2),
        "accuracy": accuracy,
        "f1": f1
    })


results_df = pd.DataFrame(results)


# ============================================================
# 13. DEFAULT THRESHOLD
# ============================================================

default_predictions = (
    val_probabilities >= 0.50
).astype(int)

default_accuracy = accuracy_score(
    y_val,
    default_predictions
)

default_f1 = f1_score(
    y_val,
    default_predictions
)


print("\nDefault threshold (0.50):")
print(f"Accuracy : {default_accuracy:.6f}")
print(f"F1       : {default_f1:.6f}")


# ============================================================
# 14. BEST ACCURACY THRESHOLD
# ============================================================

best_accuracy_row = results_df.loc[
    results_df["accuracy"].idxmax()
]

best_accuracy_threshold = best_accuracy_row["threshold"]
best_accuracy = best_accuracy_row["accuracy"]


# ============================================================
# 15. BEST F1 THRESHOLD
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_f1_threshold = best_f1_row["threshold"]
best_f1 = best_f1_row["f1"]


# ============================================================
# 16. PRINT BEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)

print("\nBest threshold for ACCURACY:")
print(f"Threshold : {best_accuracy_threshold:.2f}")
print(f"Accuracy  : {best_accuracy:.6f}")

print("\nBest threshold for F1:")
print(f"Threshold : {best_f1_threshold:.2f}")
print(f"F1        : {best_f1:.6f}")


# ============================================================
# 17. PRINT ALL THRESHOLD RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ALL THRESHOLD RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "accuracy": "{:.6f}".format,
            "f1": "{:.6f}".format
        }
    )
)


# ============================================================
# 18. PROBABILITY METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_val,
    val_probabilities
)

logloss = log_loss(
    y_val,
    val_probabilities
)

brier = brier_score_loss(
    y_val,
    val_probabilities
)


print("\n" + "=" * 70)
print("PROBABILITY METRICS")
print("=" * 70)

print(f"ROC-AUC : {roc_auc:.6f}")
print(f"Log Loss: {logloss:.6f}")
print(f"Brier   : {brier:.6f}")


# ============================================================
# 19. SAVE CSV RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "rf_threshold_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 20. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "rf_threshold_summary.txt"
)

with open(summary_path, "w") as f:

    f.write("RANDOM FOREST THRESHOLD OPTIMIZATION\n")
    f.write("=" * 60 + "\n\n")

    f.write("Features: 22\n")
    f.write("Model: Random Forest\n\n")

    f.write("Parameters:\n")
    f.write("n_estimators = 300\n")
    f.write("max_depth = 10\n")
    f.write("min_samples_split = 10\n")
    f.write("min_samples_leaf = 5\n")
    f.write("max_features = 0.5\n")
    f.write("random_state = 42\n\n")

    f.write("Default threshold (0.50):\n")
    f.write(f"Accuracy = {default_accuracy:.6f}\n")
    f.write(f"F1 = {default_f1:.6f}\n\n")

    f.write("Best threshold for Accuracy:\n")
    f.write(f"Threshold = {best_accuracy_threshold:.2f}\n")
    f.write(f"Accuracy = {best_accuracy:.6f}\n\n")

    f.write("Best threshold for F1:\n")
    f.write(f"Threshold = {best_f1_threshold:.2f}\n")
    f.write(f"F1 = {best_f1:.6f}\n\n")

    f.write("Probability metrics:\n")
    f.write(f"ROC-AUC = {roc_auc:.6f}\n")
    f.write(f"Log Loss = {logloss:.6f}\n")
    f.write(f"Brier Score = {brier:.6f}\n")


# ============================================================
# 21. DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nResults saved to:")
print(results_path)

print("\nSummary saved to:")
print(summary_path)
