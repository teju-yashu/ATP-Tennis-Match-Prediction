# rf_final_results_report.py

import os
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "feature_analysis"
)

OUTPUT_DIR = RESULTS_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. HELPER
# ============================================================

def load_csv(filename):
    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if os.path.exists(path):
        return pd.read_csv(path)

    print(f"WARNING: {filename} not found.")
    return None


# ============================================================
# 3. LOAD EXISTING RESULTS
# ============================================================

final_summary_path = os.path.join(
    RESULTS_DIR,
    "rf_final_test_summary.txt"
)

threshold_df = load_csv(
    "rf_threshold_results.csv"
)

ablation_df = load_csv(
    "rf_feature_ablation_results.csv"
)

absolute_rank_df = load_csv(
    "rf_absolute_rank_gap_analysis.csv"
)

surface_df = load_csv(
    "rf_surface_error_analysis.csv"
)

confidence_df = load_csv(
    "rf_confidence_analysis.csv"
)

importance_df = load_csv(
    "rf_final_22_feature_importance.csv"
)


# ============================================================
# 4. READ FINAL TEST SUMMARY
# ============================================================

if not os.path.exists(final_summary_path):

    raise FileNotFoundError(
        "rf_final_test_summary.txt was not found."
    )

with open(
    final_summary_path,
    "r"
) as f:

    final_summary = f.read()


# ============================================================
# 5. FINAL MODEL INFORMATION
# ============================================================

final_model_info = """
FINAL MODEL CONFIGURATION

Features:
22

Random Forest:
n_estimators = 300
max_depth = 10
min_samples_split = 10
min_samples_leaf = 5
max_features = 0.5
random_state = 42

Decision threshold:
0.54

Training data:
Train + Validation

Final test data:
2024
"""


# ============================================================
# 6. EXPERIMENTAL RESULTS
# ============================================================

# These are the results already obtained during the project.
# They are NOT being recomputed.

experiments = pd.DataFrame([
    {
        "Experiment": "Full 27 features",
        "Features": 27,
        "Accuracy": 0.650036,
        "F1": 0.637453,
        "ROC-AUC": 0.711594,
        "Log Loss": 0.620060,
        "Brier": 0.215862
    },

    {
        "Experiment": "Remove 5 negative features",
        "Features": 22,
        "Accuracy": 0.651482,
        "F1": 0.639222,
        "ROC-AUC": 0.711209,
        "Log Loss": 0.620129,
        "Brier": 0.215924
    },

    {
        "Experiment": "Remove bottom 10 features",
        "Features": 17,
        "Accuracy": 0.650398,
        "F1": 0.636603,
        "ROC-AUC": 0.712708,
        "Log Loss": 0.619245,
        "Brier": 0.215478
    },

    {
        "Experiment": "Top 15 features",
        "Features": 15,
        "Accuracy": 0.649675,
        "F1": 0.635303,
        "ROC-AUC": 0.712608,
        "Log Loss": 0.619565,
        "Brier": 0.215599
    }
])


# ============================================================
# 7. SAVE EXPERIMENT COMPARISON
# ============================================================

experiment_path = os.path.join(
    OUTPUT_DIR,
    "final_experiment_comparison.csv"
)

experiments.to_csv(
    experiment_path,
    index=False
)


# ============================================================
# 8. PRINT EXPERIMENT COMPARISON
# ============================================================

print("=" * 80)
print("FINAL EXPERIMENT COMPARISON")
print("=" * 80)

print(
    experiments.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
            "Log Loss": "{:.4f}".format,
            "Brier": "{:.4f}".format
        }
    )
)


# ============================================================
# 9. THRESHOLD SUMMARY
# ============================================================

threshold_summary = ""

if threshold_df is not None:

    best_accuracy = threshold_df.loc[
        threshold_df["accuracy"].idxmax()
    ]

    best_f1 = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    threshold_summary = f"""
THRESHOLD OPTIMIZATION

Default threshold:
0.50

Default validation accuracy:
{threshold_df.loc[
    threshold_df["threshold"] == 0.50,
    "accuracy"
].iloc[0]:.6f}

Default validation F1:
{threshold_df.loc[
    threshold_df["threshold"] == 0.50,
    "f1"
].iloc[0]:.6f}

Best accuracy threshold:
{best_accuracy["threshold"]:.2f}

Best validation accuracy:
{best_accuracy["accuracy"]:.6f}

Best F1 threshold:
{best_f1["threshold"]:.2f}

Best validation F1:
{best_f1["f1"]:.6f}
"""


# ============================================================
# 10. SURFACE SUMMARY
# ============================================================

surface_summary = ""

if surface_df is not None:

    surface_summary = "\nSURFACE PERFORMANCE\n\n"

    for _, row in surface_df.iterrows():

        surface_summary += (
            f"{row['surface']}:\n"
            f"  Matches   = {int(row['matches'])}\n"
            f"  Accuracy  = {row['accuracy']:.4f}\n"
            f"  Error     = {row['error_rate']:.4f}\n"
            f"  ROC-AUC   = {row['roc_auc']:.4f}\n\n"
        )


# ============================================================
# 11. RANKING-GAP SUMMARY
# ============================================================

rank_summary = ""

if absolute_rank_df is not None:

    rank_summary = (
        "\nABSOLUTE RANKING DIFFERENCE ANALYSIS\n\n"
    )

    for _, row in absolute_rank_df.iterrows():

        rank_summary += (
            f"{row['absolute_rank_difference_group']}:\n"
            f"  Matches   = {int(row['matches'])}\n"
            f"  Accuracy  = {row['accuracy']:.4f}\n"
            f"  Error     = {row['error_rate']:.4f}\n\n"
        )


# ============================================================
# 12. CONFIDENCE SUMMARY
# ============================================================

confidence_summary = ""

if confidence_df is not None:

    confidence_summary = (
        "\nCONFIDENCE VS ACCURACY\n\n"
    )

    for _, row in confidence_df.iterrows():

        if row["matches"] == 0:
            continue

        confidence_summary += (
            f"{row['confidence_range']}:\n"
            f"  Matches       = {int(row['matches'])}\n"
            f"  Avg confidence = "
            f"{row['average_confidence']:.4f}\n"
            f"  Accuracy      = "
            f"{row['accuracy']:.4f}\n\n"
        )


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance_summary = ""

if importance_df is not None:

    importance_summary = (
        "\nTOP 15 FINAL MODEL FEATURES\n\n"
    )

    top_features = importance_df.head(15)

    for i, (_, row) in enumerate(
        top_features.iterrows(),
        start=1
    ):

        importance_summary += (
            f"{i}. {row['feature']} "
            f"= {row['importance']:.6f}\n"
        )


# ============================================================
# 14. FINAL INTERPRETATION
# ============================================================

interpretation = """
FINAL INTERPRETATION

1. Overall performance
The final Random Forest achieved approximately 65% accuracy
and approximately 0.72 ROC-AUC on unseen 2024 ATP matches.
This indicates useful predictive capability, although the model
is not close to perfect.

2. Feature importance
Elo difference and surface-specific Elo difference are the two
dominant features in the final model. Ranking points difference
and ranking difference are also among the strongest predictors.

3. Surface dependence
The model performs better on hard courts than on clay courts.
This indicates that predictive performance depends on playing
surface.

4. Ranking-gap dependence
Prediction accuracy increases substantially when the absolute
ranking difference between players becomes large. Closely
matched players are considerably harder for the model to predict.

5. Confidence
Higher-confidence predictions generally have higher observed
accuracy. Therefore, the model's probability estimates contain
useful information about prediction reliability.

6. Feature reduction
Removing five low-performing features produced only a very small
improvement over the full 27-feature model. Therefore, the main
benefit of feature reduction was simplification rather than a
large performance improvement.

7. Threshold optimization
The validation data selected a threshold of 0.54 for maximum
accuracy. The improvement over the default 0.50 threshold was
very small, so threshold optimization did not fundamentally
change model performance.

8. Main limitation
The model struggles most with closely matched players and
performs less strongly on clay. This suggests that additional
contextual and surface-specific information may be necessary
for substantial future improvement.
"""


# ============================================================
# 15. CREATE COMPLETE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "FINAL_RESULTS_AND_DISCUSSION.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ATP TENNIS MATCH PREDICTION\n"
        "FINAL RESULTS AND DISCUSSION\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        final_model_info
    )

    f.write(
        "\n\n"
    )

    f.write(
        "FINAL 2024 TEST RESULTS\n"
    )

    f.write(
        "=" * 50 + "\n"
    )

    f.write(
        final_summary
    )

    f.write(
        "\n\n"
    )

    f.write(
        "MODEL EXPERIMENT COMPARISON\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        experiments.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        threshold_summary
    )

    f.write(
        surface_summary
    )

    f.write(
        rank_summary
    )

    f.write(
        confidence_summary
    )

    f.write(
        importance_summary
    )

    f.write(
        "\n"
    )

    f.write(
        interpretation
    )


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULTS REPORT CREATED")
print("=" * 80)

print(
    "\nExperiment comparison:"
)

print(
    experiment_path
)

print(
    "\nComplete Results & Discussion:"
)

print(
    report_path
)

print(
    "\nNo model was trained."
)

print(
    "No test predictions were changed."
)

print(
    "This script only consolidates existing results."
)