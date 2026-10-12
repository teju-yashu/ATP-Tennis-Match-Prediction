# rf_corrected_visualizations.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

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

PLOTS_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


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

TARGET = "target"


# ============================================================
# 3. FROZEN FINAL MODEL
# ============================================================

MODEL = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 4. START
# ============================================================

print("=" * 75)
print("CORRECTED FINAL VISUALIZATIONS")
print("=" * 75)

print("\nThis script will:")
print("1. Create absolute ranking-gap analysis")
print("2. Train the frozen final RF once")
print("3. Extract importance from the actual 22-feature model")
print("4. Create corrected plots")

print("\nNo hyperparameter tuning.")
print("No threshold optimization.")
print("No test-set evaluation.")


# ============================================================
# 5. CHECK FILES
# ============================================================

for path, name in [
    (TRAIN_PATH, "Training"),
    (VAL_PATH, "Validation"),
    (PREDICTIONS_PATH, "Final predictions")
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )


# ============================================================
# 6. PART A — ABSOLUTE RANK DIFFERENCE
# ============================================================

print("\n" + "=" * 75)
print("PART A — ABSOLUTE RANK DIFFERENCE")
print("=" * 75)

pred_df = pd.read_csv(
    PREDICTIONS_PATH
)

if "rank_difference" not in pred_df.columns:

    raise ValueError(
        "rank_difference is not present "
        "in the final prediction file."
    )

if "correct" not in pred_df.columns:

    # Recreate correctness if necessary
    pred_df["correct"] = (
        pred_df["target"] ==
        pred_df["predicted_class"]
    ).astype(int)


# Absolute ranking gap
pred_df["absolute_rank_difference"] = (
    pred_df["rank_difference"].abs()
)


# ------------------------------------------------------------
# Define meaningful ranking-gap groups
# ------------------------------------------------------------

bins = [
    -0.001,
    5,
    10,
    20,
    50,
    100,
    np.inf
]

labels = [
    "0–5",
    "6–10",
    "11–20",
    "21–50",
    "51–100",
    ">100"
]

pred_df["rank_gap_group"] = pd.cut(
    pred_df["absolute_rank_difference"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


rank_gap_results = []

for group in labels:

    subset = pred_df[
        pred_df["rank_gap_group"] == group
    ]

    if len(subset) == 0:
        continue

    group_accuracy = (
        subset["correct"].mean()
    )

    rank_gap_results.append({
        "absolute_rank_difference_group": group,
        "matches": len(subset),
        "accuracy": group_accuracy,
        "error_rate": 1 - group_accuracy
    })


rank_gap_df = pd.DataFrame(
    rank_gap_results
)


# ------------------------------------------------------------
# Print ranking-gap results
# ------------------------------------------------------------

print("\nAbsolute ranking-gap results:")

print(
    rank_gap_df.to_string(
        index=False,
        formatters={
            "accuracy":
                lambda x: f"{x:.4f}",

            "error_rate":
                lambda x: f"{x:.4f}"
        }
    )
)


# ------------------------------------------------------------
# Save ranking-gap results
# ------------------------------------------------------------

rank_gap_csv = os.path.join(
    OUTPUT_DIR,
    "rf_absolute_rank_gap_analysis.csv"
)

rank_gap_df.to_csv(
    rank_gap_csv,
    index=False
)


# ------------------------------------------------------------
# Create ranking-gap plot
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    rank_gap_df[
        "absolute_rank_difference_group"
    ],
    rank_gap_df["accuracy"],
    marker="o"
)

overall_accuracy = (
    pred_df["correct"].mean()
)

plt.axhline(
    overall_accuracy,
    linestyle="--",
    label=(
        f"Overall accuracy = "
        f"{overall_accuracy:.3f}"
    )
)

plt.ylim(
    0,
    1
)

plt.xlabel(
    "Absolute Ranking Difference"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Prediction Accuracy vs Absolute Ranking Difference — 2024 Test"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


rank_plot_path = os.path.join(
    PLOTS_DIR,
    "04_corrected_absolute_rank_difference.png"
)

plt.savefig(
    rank_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. PART B — ACTUAL FINAL MODEL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 75)
print("PART B — FINAL 22-FEATURE MODEL IMPORTANCE")
print("=" * 75)

print("\nLoading training and validation data...")

train_df = pd.read_csv(
    TRAIN_PATH
)

val_df = pd.read_csv(
    VAL_PATH
)


# ------------------------------------------------------------
# Check all 22 features
# ------------------------------------------------------------

for dataset, name in [
    (train_df, "training"),
    (val_df, "validation")
]:

    missing = [
        feature
        for feature in FEATURES
        if feature not in dataset.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing features in {name} data:\n"
            + "\n".join(missing)
        )


# ------------------------------------------------------------
# Combine train + validation
# ------------------------------------------------------------

final_train_df = pd.concat(
    [
        train_df,
        val_df
    ],
    ignore_index=True
)

X_final = final_train_df[
    FEATURES
]

y_final = final_train_df[
    TARGET
]

print(
    f"\nFinal training rows : "
    f"{len(final_train_df):,}"
)

print(
    f"Final features      : "
    f"{len(FEATURES)}"
)


# ============================================================
# 8. TRAIN EXACT FROZEN MODEL
# ============================================================

print("\nTraining exact frozen Random Forest...")

MODEL.fit(
    X_final,
    y_final
)

print(
    "Training complete."
)


# ============================================================
# 9. EXTRACT FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": MODEL.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# Print all 22
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL MODEL FEATURE IMPORTANCE")
print("=" * 75)

for index, row in importance_df.iterrows():

    print(
        f"{index + 1:2d}. "
        f"{row['feature']:45s} "
        f"{row['importance']:.6f}"
    )


# ============================================================
# 10. SAVE FEATURE IMPORTANCE
# ============================================================

importance_csv = os.path.join(
    OUTPUT_DIR,
    "rf_final_22_feature_importance.csv"
)

importance_df.to_csv(
    importance_csv,
    index=False
)


# ============================================================
# 11. CREATE CORRECTED FEATURE IMPORTANCE PLOT
# ============================================================

top_features = (
    importance_df
    .head(15)
    .sort_values(
        "importance",
        ascending=True
    )
)


plt.figure(
    figsize=(11, 8)
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features — Final 22-Feature Random Forest"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()


importance_plot_path = os.path.join(
    PLOTS_DIR,
    "05_corrected_final_feature_importance.png"
)

plt.savefig(
    importance_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. SAVE A TEXT SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "rf_corrected_visualization_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "CORRECTED FINAL VISUALIZATION SUMMARY\n"
    )

    f.write(
        "=" * 65 + "\n\n"
    )

    f.write(
        "Absolute Ranking Difference Analysis\n"
    )

    f.write(
        rank_gap_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "Final 22-Feature Importance\n"
    )

    f.write(
        importance_df.to_string(
            index=False
        )
    )


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("CORRECTED VISUALIZATIONS COMPLETE")
print("=" * 75)

print("\nCreated:")

print(
    "\n1. Corrected ranking-gap results:"
)

print(
    rank_gap_csv
)

print(
    "\n2. Corrected ranking-gap plot:"
)

print(
    rank_plot_path
)

print(
    "\n3. Final 22-feature importance:"
)

print(
    importance_csv
)

print(
    "\n4. Corrected feature-importance plot:"
)

print(
    importance_plot_path
)

print(
    "\n5. Summary:"
)

print(
    summary_path
)

print("\nDone.")
