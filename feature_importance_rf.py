# ================================================================
# RANDOM FOREST FEATURE IMPORTANCE ANALYSIS
# ATP TENNIS MATCH PREDICTION
# ================================================================

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
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

RESULTS_DIR = os.path.join(
    DATA_DIR,
    "feature_analysis"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
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
print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
print("=" * 75)

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_val = validation[FEATURES].copy()
y_val = validation[TARGET].astype(int)

print(f"Train rows      : {len(train):,}")
print(f"Validation rows : {len(validation):,}")
print(f"Features        : {len(FEATURES)}")


# ================================================================
# DATA CHECK
# ================================================================

if X_train.isnull().sum().sum() > 0:
    raise ValueError("Training data contains missing values.")

if X_val.isnull().sum().sum() > 0:
    raise ValueError("Validation data contains missing values.")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training data contains infinite values.")

if not np.isfinite(X_val.to_numpy()).all():
    raise ValueError("Validation data contains infinite values.")

print("\nData checks passed.")


# ================================================================
# TRAIN BASELINE RANDOM FOREST
# ================================================================

print("\n" + "=" * 75)
print("TRAINING BASELINE RANDOM FOREST")
print("=" * 75)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

print("\nTraining...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ================================================================
# 1. BUILT-IN RF FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 75)
print("BUILT-IN RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 75)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

importance_df["rank"] = (
    np.arange(len(importance_df)) + 1
)

print("\nFeature importance ranking:\n")

print(
    importance_df[
        ["rank", "feature", "importance"]
    ].to_string(index=False)
)


# ================================================================
# SAVE BUILT-IN IMPORTANCE
# ================================================================

importance_path = os.path.join(
    RESULTS_DIR,
    "rf_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print(
    f"\nSaved to:\n{importance_path}"
)


# ================================================================
# 2. PERMUTATION IMPORTANCE
# ================================================================

print("\n" + "=" * 75)
print("PERMUTATION IMPORTANCE")
print("=" * 75)

print(
    "\nCalculating permutation importance on validation data..."
)

print(
    "This does NOT retrain the Random Forest."
)

permutation = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="accuracy",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)


# ================================================================
# CREATE PERMUTATION TABLE
# ================================================================

permutation_df = pd.DataFrame({

    "feature": FEATURES,

    "mean_importance":
        permutation.importances_mean,

    "std_importance":
        permutation.importances_std

})

permutation_df = permutation_df.sort_values(
    by="mean_importance",
    ascending=False
).reset_index(drop=True)

permutation_df["rank"] = (
    np.arange(len(permutation_df)) + 1
)


# ================================================================
# DISPLAY PERMUTATION IMPORTANCE
# ================================================================

print("\nPermutation importance ranking:\n")

print(
    permutation_df[
        [
            "rank",
            "feature",
            "mean_importance",
            "std_importance"
        ]
    ].to_string(index=False)
)


# ================================================================
# SAVE PERMUTATION IMPORTANCE
# ================================================================

permutation_path = os.path.join(
    RESULTS_DIR,
    "rf_permutation_importance.csv"
)

permutation_df.to_csv(
    permutation_path,
    index=False
)

print(
    f"\nSaved to:\n{permutation_path}"
)


# ================================================================
# TOP / WEAK FEATURES
# ================================================================

print("\n" + "=" * 75)
print("FEATURE SUMMARY")
print("=" * 75)

print("\nTOP 10 FEATURES:\n")

print(
    permutation_df[
        ["feature", "mean_importance"]
    ].head(10).to_string(index=False)
)


print("\nBOTTOM 10 FEATURES:\n")

print(
    permutation_df[
        ["feature", "mean_importance"]
    ].tail(10).to_string(index=False)
)


# ================================================================
# FEATURES WITH NEGATIVE PERMUTATION IMPORTANCE
# ================================================================

negative_features = permutation_df[
    permutation_df["mean_importance"] < 0
]

print(
    "\nFeatures with negative permutation importance:"
)

if len(negative_features) == 0:

    print("None.")

else:

    print(
        negative_features[
            ["feature", "mean_importance"]
        ].to_string(index=False)
    )


# ================================================================
# SAVE FEATURE SUMMARY
# ================================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "feature_summary.txt"
)

with open(
    summary_path,
    "w"
) as file:

    file.write(
        "RANDOM FOREST FEATURE ANALYSIS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "TOP 10 FEATURES\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for _, row in permutation_df.head(10).iterrows():

        file.write(
            f"{row['feature']}: "
            f"{row['mean_importance']:.6f}\n"
        )

    file.write(
        "\n\nBOTTOM 10 FEATURES\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for _, row in permutation_df.tail(10).iterrows():

        file.write(
            f"{row['feature']}: "
            f"{row['mean_importance']:.6f}\n"
        )

    file.write(
        "\n\nNEGATIVE IMPORTANCE FEATURES\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    if len(negative_features) == 0:

        file.write("None.\n")

    else:

        for _, row in negative_features.iterrows():

            file.write(
                f"{row['feature']}: "
                f"{row['mean_importance']:.6f}\n"
            )


print(
    f"\nSummary saved to:\n{summary_path}"
)


# ================================================================
# COMPLETION
# ================================================================

print("\n" + "=" * 75)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 75)

print(
    "\nNo feature has been removed automatically."
)

print(
    "Use these results to decide the next feature-ablation step."
)

print("\nDONE.")