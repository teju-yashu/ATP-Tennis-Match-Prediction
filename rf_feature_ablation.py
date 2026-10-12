# ================================================================
# RANDOM FOREST FEATURE ABLATION
# ATP TENNIS MATCH PREDICTION
# ================================================================

import os
import warnings

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
# ALL FEATURES
# ================================================================

ALL_FEATURES = [
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
print("RANDOM FOREST FEATURE ABLATION")
print("=" * 75)

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)

y_train = train[TARGET].astype(int)
y_val = validation[TARGET].astype(int)

print(f"Train rows      : {len(train):,}")
print(f"Validation rows : {len(validation):,}")
print(f"Total features  : {len(ALL_FEATURES)}")


# ================================================================
# FEATURE GROUPS
# ================================================================

# Features identified as negative or essentially useless
REMOVE_NEGATIVE = [
    "h2h_matches",
    "recent_h2h_win_rate_difference",
    "hand_difference",
    "win_rate_last_30_days_difference",
    "surface_win_rate_difference"
]


# Bottom 10 based on permutation importance
BOTTOM_10 = [
    "double_fault_rate_difference",
    "overall_win_rate_difference",
    "surface_h2h_matches",
    "win_rate_last_10_difference",
    "tournament_win_rate_difference",
    "h2h_matches",
    "recent_h2h_win_rate_difference",
    "hand_difference",
    "win_rate_last_30_days_difference",
    "surface_win_rate_difference"
]


# Top 15 based on permutation importance
TOP_15 = [
    "surface_elo_difference",
    "elo_difference",
    "rank_difference",
    "age_difference",
    "rank_points_difference",
    "recency_weighted_form_difference",
    "first_serve_win_percentage_difference",
    "ace_rate_difference",
    "first_serve_percentage_difference",
    "win_rate_last_20_difference",
    "break_point_save_percentage_difference",
    "second_serve_win_percentage_difference",
    "win_rate_last_5_difference",
    "surface_h2h_win_rate_difference",
    "h2h_win_rate_difference"
]


# ================================================================
# CREATE FEATURE SETS
# ================================================================

FEATURE_SETS = {

    "Full_27_features":
        ALL_FEATURES,

    "Remove_5_negative_features":
        [
            f for f in ALL_FEATURES
            if f not in REMOVE_NEGATIVE
        ],

    "Remove_bottom_10_features":
        [
            f for f in ALL_FEATURES
            if f not in BOTTOM_10
        ],

    "Top_15_features":
        TOP_15
}


# ================================================================
# MODEL EVALUATION
# ================================================================

def evaluate_model(model, X, y):

    probability = model.predict_proba(X)[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "accuracy":
            accuracy_score(y, prediction),

        "f1":
            f1_score(y, prediction),

        "roc_auc":
            roc_auc_score(y, probability),

        "log_loss":
            log_loss(y, probability),

        "brier":
            brier_score_loss(y, probability)
    }


# ================================================================
# RUN ABLATION
# ================================================================

results = []


for experiment_name, features in FEATURE_SETS.items():

    print("\n" + "=" * 75)

    print(
        f"EXPERIMENT: {experiment_name}"
    )

    print("=" * 75)

    print(
        f"Number of features: {len(features)}"
    )

    X_train = train[features].copy()
    X_val = validation[features].copy()

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=10,

        min_samples_split=10,

        min_samples_leaf=5,

        max_features=0.5,

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_val,
        y_val
    )

    print("\nValidation Results:")

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    print(
        f"Log Loss : {metrics['log_loss']:.4f}"
    )

    print(
        f"Brier    : {metrics['brier']:.4f}"
    )

    results.append({

        "experiment":
            experiment_name,

        "number_of_features":
            len(features),

        "accuracy":
            metrics["accuracy"],

        "f1":
            metrics["f1"],

        "roc_auc":
            metrics["roc_auc"],

        "log_loss":
            metrics["log_loss"],

        "brier":
            metrics["brier"]
    })


# ================================================================
# RESULTS TABLE
# ================================================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="accuracy",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 75)
print("FEATURE ABLATION RESULTS")
print("=" * 75)

print(
    "\n" +
    results_df.to_string(index=False)
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_path = os.path.join(
    RESULTS_DIR,
    "rf_feature_ablation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print(
    f"\nResults saved to:"
    f"\n{results_path}"
)


# ================================================================
# BEST FEATURE SET
# ================================================================

best = results_df.iloc[0]


print("\n" + "=" * 75)
print("BEST FEATURE SET")
print("=" * 75)

print(
    f"\nExperiment       : "
    f"{best['experiment']}"
)

print(
    f"Features         : "
    f"{int(best['number_of_features'])}"
)

print(
    f"Accuracy         : "
    f"{best['accuracy']:.4f}"
)

print(
    f"F1               : "
    f"{best['f1']:.4f}"
)

print(
    f"ROC-AUC          : "
    f"{best['roc_auc']:.4f}"
)

print(
    f"Log Loss         : "
    f"{best['log_loss']:.4f}"
)

print(
    f"Brier            : "
    f"{best['brier']:.4f}"
)


# ================================================================
# COMPLETION
# ================================================================

print("\n" + "=" * 75)
print("FEATURE ABLATION COMPLETED")
print("=" * 75)

print(
    "\nNo test data was used."
)

print(
    "No hyperparameter tuning was performed."
)

print(
    "The best feature set is selected using validation performance."
)

print("\nDONE.")