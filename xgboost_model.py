# ================================================================
# XGBOOST MODEL - ATP TENNIS MATCH PREDICTION
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

RESULTS_DIR = os.path.join(
    DATA_DIR,
    "xgboost_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


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
# HEADER
# ================================================================

print("=" * 75)
print("XGBOOST - ATP TENNIS MATCH PREDICTION")
print("=" * 75)


# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train      : {len(train):,} rows")
print(f"Validation : {len(validation):,} rows")
print(f"Test       : {len(test):,} rows")


# ================================================================
# CHECK COLUMNS
# ================================================================

print("\nChecking required columns...")

required_columns = FEATURES + [TARGET]

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"\nERROR in {name}:")
        for col in missing:
            print(f"  - {col}")

        raise ValueError(
            f"{name} is missing required columns."
        )

print("All required columns are present.")


# ================================================================
# PREPARE DATA
# ================================================================

print("\nPreparing features...")

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_val = validation[FEATURES].copy()
y_val = validation[TARGET].astype(int)

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int)

print(f"Number of features : {len(FEATURES)}")


# ================================================================
# DATA QUALITY CHECK
# ================================================================

print("\nChecking missing and infinite values...")

for name, X in [
    ("TRAIN", X_train),
    ("VALIDATION", X_val),
    ("TEST", X_test)
]:

    missing = X.isnull().sum().sum()

    if missing > 0:
        raise ValueError(
            f"{name} contains {missing} missing values."
        )

    infinite = (
        ~np.isfinite(X.to_numpy())
    ).sum()

    if infinite > 0:
        raise ValueError(
            f"{name} contains {infinite} infinite values."
        )

print("Data checks passed.")


# ================================================================
# MODEL CONFIGURATION
# ================================================================

print("\n" + "=" * 75)
print("BUILDING XGBOOST MODEL")
print("=" * 75)

print("\nModel configuration:")

print("n_estimators     = 300")
print("max_depth        = 5")
print("learning_rate    = 0.05")
print("subsample        = 0.80")
print("colsample_bytree = 0.80")
print("min_child_weight = 5")
print("reg_lambda       = 1.0")
print("reg_alpha        = 0.0")


model = XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.80,

    colsample_bytree=0.80,

    min_child_weight=5,

    reg_lambda=1.0,

    reg_alpha=0.0,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# ================================================================
# TRAIN
# ================================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    verbose=False
)

print("Training completed.")


# ================================================================
# EVALUATION FUNCTION
# ================================================================

def evaluate_model(model, X, y, dataset_name):

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

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "=" * 75)
    print(f"XGBOOST - {dataset_name}")
    print("=" * 75)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"Log Loss : {loss:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Player 2 wins",
                "Player 1 wins"
            ],
            digits=2
        )
    )

    return {
        "model": f"XGBoost - {dataset_name}",
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss
    }


# ================================================================
# TRAIN EVALUATION
# ================================================================

train_results = evaluate_model(
    model,
    X_train,
    y_train,
    "TRAIN"
)


# ================================================================
# VALIDATION EVALUATION
# ================================================================

validation_results = evaluate_model(
    model,
    X_val,
    y_val,
    "VALIDATION"
)


# ================================================================
# TEST EVALUATION
# ================================================================

test_results = evaluate_model(
    model,
    X_test,
    y_test,
    "TEST"
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 75)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 75)

importance_df = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        model.feature_importances_

})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


print("\nTop 15 features:\n")

for _, row in importance_df.head(15).iterrows():

    print(
        f"{row['feature']:<45}"
        f"{row['importance']:.6f}"
    )


# ================================================================
# SAVE FEATURE IMPORTANCE
# ================================================================

importance_path = os.path.join(
    RESULTS_DIR,
    "xgboost_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print(
    f"\nFeature importance saved to:\n"
    f"{importance_path}"
)


# ================================================================
# SAVE MODEL RESULTS
# ================================================================

results_df = pd.DataFrame([

    train_results,

    validation_results,

    test_results

])

results_path = os.path.join(
    RESULTS_DIR,
    "xgboost_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nModel results saved to:\n"
    f"{results_path}"
)


# ================================================================
# CURRENT BENCHMARKS
# ================================================================

BASELINE_ACCURACY = 0.6421
BASELINE_F1 = 0.6290
BASELINE_ROC_AUC = 0.7043
BASELINE_LOG_LOSS = 0.6304

LOGISTIC_ACCURACY = 0.6528
LOGISTIC_F1 = 0.6496
LOGISTIC_ROC_AUC = 0.7216
LOGISTIC_LOG_LOSS = 0.6115

TUNED_RF_ACCURACY = 0.6559
TUNED_RF_F1 = 0.6575
TUNED_RF_ROC_AUC = 0.7191
TUNED_RF_LOG_LOSS = 0.6154


# ================================================================
# COMPARISON
# ================================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print("\nTEST PERFORMANCE")

comparison = pd.DataFrame({

    "Model": [
        "Ranking Baseline",
        "Logistic Regression",
        "Tuned Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        BASELINE_ACCURACY,
        LOGISTIC_ACCURACY,
        TUNED_RF_ACCURACY,
        test_results["accuracy"]
    ],

    "F1": [
        BASELINE_F1,
        LOGISTIC_F1,
        TUNED_RF_F1,
        test_results["f1"]
    ],

    "ROC-AUC": [
        BASELINE_ROC_AUC,
        LOGISTIC_ROC_AUC,
        TUNED_RF_ROC_AUC,
        test_results["roc_auc"]
    ],

    "Log Loss": [
        BASELINE_LOG_LOSS,
        LOGISTIC_LOG_LOSS,
        TUNED_RF_LOG_LOSS,
        test_results["log_loss"]
    ]

})

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# XGBOOST IMPROVEMENT OVER BASELINE
# ================================================================

print("\n" + "=" * 75)
print("XGBOOST vs RANKING BASELINE")
print("=" * 75)

print(
    f"\nAccuracy improvement : "
    f"{test_results['accuracy'] - BASELINE_ACCURACY:+.4f}"
)

print(
    f"F1 improvement       : "
    f"{test_results['f1'] - BASELINE_F1:+.4f}"
)

print(
    f"ROC-AUC improvement  : "
    f"{test_results['roc_auc'] - BASELINE_ROC_AUC:+.4f}"
)

print(
    f"Log Loss improvement : "
    f"{BASELINE_LOG_LOSS - test_results['log_loss']:+.4f}"
)


# ================================================================
# SAVE COMPARISON
# ================================================================

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print(
    f"\nComparison saved to:\n"
    f"{comparison_path}"
)


# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 75)
print("XGBOOST COMPLETED")
print("=" * 75)

print("\nGenerated files:")

print(
    f"- {results_path}"
)

print(
    f"- {importance_path}"
)

print(
    f"- {comparison_path}"
)

print("\nDONE.")