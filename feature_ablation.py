from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"

RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "feature_ablation_results.csv"
)


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {

    "Ranking": [
        "rank_difference",
        "rank_points_difference"
    ],

    "ELO": [
        "elo_difference",
        "surface_elo_difference"
    ],

    "Recent Form": [
        "overall_win_rate_difference",
        "win_rate_last_5_difference",
        "win_rate_last_10_difference",
        "win_rate_last_20_difference",
        "recency_weighted_form_difference",
        "win_rate_last_30_days_difference"
    ],

    "Surface Form": [
        "surface_win_rate_difference",
        "surface_recent_win_rate_difference"
    ],

    "Head-to-Head": [
        "h2h_matches",
        "h2h_win_rate_difference",
        "recent_h2h_win_rate_difference",
        "surface_h2h_matches",
        "surface_h2h_win_rate_difference"
    ],

    "Tournament": [
        "tournament_win_rate_difference"
    ],

    "Player Profile": [
        "age_difference",
        "height_difference",
        "hand_difference"
    ],

    "Serve Statistics": [
        "ace_rate_difference",
        "double_fault_rate_difference",
        "first_serve_percentage_difference",
        "first_serve_win_percentage_difference",
        "second_serve_win_percentage_difference",
        "break_point_save_percentage_difference"
    ]
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 75)
    print("FEATURE ABLATION STUDY")
    print("=" * 75)

    print("\nLoading datasets...")

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

    return train, validation, test


# ============================================================
# CHECK FEATURES
# ============================================================

def check_features(df):

    print("\nChecking feature columns...")

    missing = []

    for group_name, features in FEATURE_GROUPS.items():

        for feature in features:

            if feature not in df.columns:
                missing.append(feature)

    if missing:

        missing = sorted(set(missing))

        raise ValueError(
            "The following features are missing:\n"
            + "\n".join(missing)
        )

    print("All feature groups are available.")


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    random_state=42
                )
            )
        ]
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    X,
    y
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "f1": f1_score(
            y,
            predictions
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),

        "log_loss": log_loss(
            y,
            probabilities
        )
    }


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    name,
    features,
    train,
    validation,
    test
):

    print("\n" + "=" * 75)

    print(
        f"EXPERIMENT: {name}"
    )

    print("=" * 75)

    print(
        f"Features used: {len(features)}"
    )

    for feature in features:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X_train = train[
        features
    ]

    X_validation = validation[
        features
    ]

    X_test = test[
        features
    ]

    y_train = train[
        "target"
    ].astype(int)

    y_validation = validation[
        "target"
    ].astype(int)

    y_test = test[
        "target"
    ].astype(int)

    # --------------------------------------------------------
    # Build and train
    # --------------------------------------------------------

    model = build_model()

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    train_metrics = evaluate(
        model,
        X_train,
        y_train
    )

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nTRAIN")

    print(
        f"Accuracy : "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{train_metrics['roc_auc']:.4f}"
    )

    print("\nVALIDATION")

    print(
        f"Accuracy : "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print("\nTEST")

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"F1       : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Log Loss : "
        f"{test_metrics['log_loss']:.4f}"
    )

    return {
        "experiment": name,
        "feature_count": len(features),

        "train_accuracy":
            train_metrics["accuracy"],

        "validation_accuracy":
            validation_metrics["accuracy"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_f1":
            test_metrics["f1"],

        "test_roc_auc":
            test_metrics["roc_auc"],

        "test_log_loss":
            test_metrics["log_loss"]
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train, validation, test = load_data()

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    check_features(train)

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    current_features = []

    # ========================================================
    # EXPERIMENT 1
    # Ranking only
    # ========================================================

    current_features += FEATURE_GROUPS[
        "Ranking"
    ]

    results.append(
        run_experiment(
            "Ranking",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # EXPERIMENT 2
    # Ranking + ELO
    # ========================================================

    current_features += FEATURE_GROUPS[
        "ELO"
    ]

    results.append(
        run_experiment(
            "Ranking + ELO",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # EXPERIMENT 3
    # Add Recent Form
    # ========================================================

    current_features += FEATURE_GROUPS[
        "Recent Form"
    ]

    results.append(
        run_experiment(
            "Ranking + ELO + Recent Form",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # EXPERIMENT 4
    # Add Surface Form
    # ========================================================

    current_features += FEATURE_GROUPS[
        "Surface Form"
    ]

    results.append(
        run_experiment(
            "Ranking + ELO + Form + Surface",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # EXPERIMENT 5
    # Add H2H
    # ========================================================

    current_features += FEATURE_GROUPS[
        "Head-to-Head"
    ]

    results.append(
        run_experiment(
            "Ranking + ELO + Form + Surface + H2H",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # EXPERIMENT 6
    # Add Tournament
    # ========================================================

    current_features += FEATURE_GROUPS[
        "Tournament"
    ]

    results.append(
        run_experiment(
            "Previous + Tournament",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # EXPERIMENT 7
    # Add Player Profile
    # ========================================================

    current_features += FEATURE_GROUPS[
        "Player Profile"
    ]

    results.append(
        run_experiment(
            "Previous + Player Profile",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # EXPERIMENT 8
    # Full Model
    # ========================================================

    current_features += FEATURE_GROUPS[
        "Serve Statistics"
    ]

    results.append(
        run_experiment(
            "FULL MODEL",
            current_features.copy(),
            train,
            validation,
            test
        )
    )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 75)
    print("FEATURE ABLATION SUMMARY")
    print("=" * 75)

    display_columns = [
        "experiment",
        "feature_count",
        "test_accuracy",
        "test_f1",
        "test_roc_auc",
        "test_log_loss"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    print("\n" + "=" * 75)
    print("COMPARISON WITH RANKING BASELINE")
    print("=" * 75)

    baseline_accuracy = 0.6498
    baseline_auc = 0.6794

    results_df[
        "accuracy_vs_baseline"
    ] = (
        results_df["test_accuracy"]
        - baseline_accuracy
    )

    results_df[
        "auc_vs_baseline"
    ] = (
        results_df["test_roc_auc"]
        - baseline_auc
    )

    print(
        results_df[
            [
                "experiment",
                "test_accuracy",
                "accuracy_vs_baseline",
                "test_roc_auc",
                "auc_vs_baseline"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULT_PATH
    )

    print("\n" + "=" * 75)
    print("FEATURE ABLATION COMPLETED")
    print("=" * 75)