from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
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
    / "logistic_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOGISTIC REGRESSION - TENNIS MATCH PREDICTION")
    print("=" * 70)

    print("\nLoading datasets...")

    train = pd.read_csv(TRAIN_PATH)

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

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
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train,
    validation,
    test
):

    print("\nPreparing features...")

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in train.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features:\n"
            + "\n".join(missing_features)
        )

    X_train = train[
        FEATURE_COLUMNS
    ].copy()

    X_validation = validation[
        FEATURE_COLUMNS
    ].copy()

    X_test = test[
        FEATURE_COLUMNS
    ].copy()

    y_train = train[
        "target"
    ].astype(int)

    y_validation = validation[
        "target"
    ].astype(int)

    y_test = test[
        "target"
    ].astype(int)

    print(
        f"Number of features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\nBuilding Logistic Regression pipeline...")

    model = Pipeline(
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

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model_name,
    model,
    X,
    y,
    dataset_name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    f1 = f1_score(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    loss = log_loss(
        y,
        probabilities
    )

    print("\n" + "=" * 70)

    print(
        f"{model_name} - {dataset_name}"
    )

    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"Log Loss : {loss:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Player 2 wins",
                "Player 1 wins"
            ]
        )
    )

    return {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss
    }


# ============================================================
# FEATURE COEFFICIENTS
# ============================================================

def print_coefficients(model):

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION FEATURE COEFFICIENTS")
    print("=" * 70)

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = (
        classifier.coef_[0]
    )

    coefficient_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": coefficients
    })

    coefficient_df[
        "absolute_coefficient"
    ] = coefficient_df[
        "coefficient"
    ].abs()

    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute_coefficient",
            ascending=False
        )
    )

    print(
        coefficient_df[
            [
                "feature",
                "coefficient"
            ]
        ]
        .to_string(
            index=False
        )
    )

    return coefficient_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    results_df = pd.DataFrame(
        results
    )

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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        train,
        validation,
        test
    ) = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = prepare_features(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    train_result = evaluate_model(
        "Logistic Regression",
        model,
        X_train,
        y_train,
        "Train"
    )

    results.append(
        train_result
    )

    validation_result = evaluate_model(
        "Logistic Regression",
        model,
        X_validation,
        y_validation,
        "Validation"
    )

    results.append(
        validation_result
    )

    test_result = evaluate_model(
        "Logistic Regression",
        model,
        X_test,
        y_test,
        "Test"
    )

    results.append(
        test_result
    )

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    print_coefficients(
        model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Baseline comparison
    # --------------------------------------------------------

    baseline_accuracy = 0.6498
    baseline_roc_auc = 0.6794
    baseline_log_loss = 1.1507

    print("\n" + "=" * 70)
    print("COMPARISON WITH RANKING BASELINE")
    print("=" * 70)

    logistic_accuracy = (
        test_result["accuracy"]
    )

    logistic_roc_auc = (
        test_result["roc_auc"]
    )

    logistic_log_loss = (
        test_result["log_loss"]
    )

    print(
        f"\nRanking baseline accuracy : "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Logistic Regression       : "
        f"{logistic_accuracy:.4f}"
    )

    print(
        f"Accuracy improvement      : "
        f"{logistic_accuracy - baseline_accuracy:+.4f}"
    )

    print(
        f"\nRanking baseline ROC-AUC  : "
        f"{baseline_roc_auc:.4f}"
    )

    print(
        f"Logistic Regression       : "
        f"{logistic_roc_auc:.4f}"
    )

    print(
        f"ROC-AUC improvement       : "
        f"{logistic_roc_auc - baseline_roc_auc:+.4f}"
    )

    print(
        f"\nRanking baseline Log Loss : "
        f"{baseline_log_loss:.4f}"
    )

    print(
        f"Logistic Regression       : "
        f"{logistic_log_loss:.4f}"
    )

    print(
        f"Log Loss improvement      : "
        f"{baseline_log_loss - logistic_log_loss:+.4f}"
    )

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION COMPLETED")
    print("=" * 70)