# ================================================================
# RANDOM FOREST MODEL - ATP TENNIS MATCH PREDICTION
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
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

RESULTS_DIR = os.path.join(DATA_DIR, "random_forest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# FEATURES
# ================================================================

# We deliberately exclude IDs, names and categorical text fields.
# These features are numerical and already engineered.

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

print("=" * 70)
print("RANDOM FOREST - ATP TENNIS MATCH PREDICTION")
print("=" * 70)


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
# CHECK REQUIRED COLUMNS
# ================================================================

print("\nChecking required columns...")

required_columns = FEATURES + [TARGET]

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        print(f"\nERROR in {name}:")
        print("Missing columns:")
        for col in missing:
            print(f"  - {col}")
        raise ValueError(f"{name} dataset is missing required columns.")

print("All required columns are present.")


# ================================================================
# CHECK MISSING VALUES
# ================================================================

print("\nChecking missing values...")

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    missing_count = df[FEATURES].isnull().sum().sum()

    print(f"{name:<12}: {missing_count} missing feature values")

    if missing_count > 0:
        print(df[FEATURES].isnull().sum())
        raise ValueError(
            f"{name} contains missing feature values. "
            "Fix the prepared dataset before training."
        )


# ================================================================
# CHECK INFINITE VALUES
# ================================================================

print("\nChecking infinite values...")

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    X_check = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    infinite_count = X_check.isnull().sum().sum()

    print(f"{name:<12}: {infinite_count} invalid numeric values")

    if infinite_count > 0:
        raise ValueError(
            f"{name} contains infinite/non-finite feature values."
        )


# ================================================================
# PREPARE FEATURES AND TARGET
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
# TARGET DISTRIBUTION
# ================================================================

print("\nTarget distribution:")

for name, y in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:
    print(f"\n{name}:")
    print(y.value_counts(normalize=True).sort_index())


# ================================================================
# BUILD RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("BUILDING RANDOM FOREST")
print("=" * 70)

print("\nModel configuration:")
print("n_estimators      = 300")
print("max_depth         = 12")
print("min_samples_split = 10")
print("min_samples_leaf  = 5")
print("max_features      = sqrt")
print("class_weight      = balanced")
print("random_state      = 42")


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ================================================================
# TRAIN
# ================================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# ================================================================
# EVALUATION FUNCTION
# ================================================================

def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    roc_auc = roc_auc_score(y, probabilities)
    loss = log_loss(y, probabilities)

    cm = confusion_matrix(y, predictions)

    print("\n" + "=" * 70)
    print(f"RANDOM FOREST - {dataset_name}")
    print("=" * 70)

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
        "model": f"Random Forest - {dataset_name}",
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss
    }


# ================================================================
# EVALUATE TRAIN
# ================================================================

train_results = evaluate_model(
    model,
    X_train,
    y_train,
    "TRAIN"
)


# ================================================================
# EVALUATE VALIDATION
# ================================================================

validation_results = evaluate_model(
    model,
    X_val,
    y_val,
    "VALIDATION"
)


# ================================================================
# EVALUATE TEST
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

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nTop 15 features:\n")

for _, row in importance_df.head(15).iterrows():
    print(
        f"{row['feature']:<45} "
        f"{row['importance']:.6f}"
    )


# ================================================================
# SAVE FEATURE IMPORTANCE
# ================================================================

importance_path = os.path.join(
    RESULTS_DIR,
    "random_forest_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print(f"\nFeature importance saved to:")
print(importance_path)


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
    "random_forest_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel results saved to:")
print(results_path)


# ================================================================
# COMPARE WITH CORRECTED BASELINE
# ================================================================

BASELINE_TEST_ACCURACY = 0.6421
BASELINE_TEST_F1 = 0.6290
BASELINE_TEST_ROC_AUC = 0.7043
BASELINE_TEST_LOG_LOSS = 0.6304

rf_accuracy = test_results["accuracy"]
rf_f1 = test_results["f1"]
rf_roc_auc = test_results["roc_auc"]
rf_log_loss = test_results["log_loss"]

print("\n" + "=" * 70)
print("RANDOM FOREST vs RANKING BASELINE")
print("=" * 70)

print("\nTEST PERFORMANCE")

print(
    f"\nAccuracy:"
    f"\n  Baseline       : {BASELINE_TEST_ACCURACY:.4f}"
    f"\n  Random Forest  : {rf_accuracy:.4f}"
    f"\n  Improvement    : {rf_accuracy - BASELINE_TEST_ACCURACY:+.4f}"
)

print(
    f"\nF1 Score:"
    f"\n  Baseline       : {BASELINE_TEST_F1:.4f}"
    f"\n  Random Forest  : {rf_f1:.4f}"
    f"\n  Improvement    : {rf_f1 - BASELINE_TEST_F1:+.4f}"
)

print(
    f"\nROC-AUC:"
    f"\n  Baseline       : {BASELINE_TEST_ROC_AUC:.4f}"
    f"\n  Random Forest  : {rf_roc_auc:.4f}"
    f"\n  Improvement    : {rf_roc_auc - BASELINE_TEST_ROC_AUC:+.4f}"
)

print(
    f"\nLog Loss:"
    f"\n  Baseline       : {BASELINE_TEST_LOG_LOSS:.4f}"
    f"\n  Random Forest  : {rf_log_loss:.4f}"
    f"\n  Improvement    : {BASELINE_TEST_LOG_LOSS - rf_log_loss:+.4f}"
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("RANDOM FOREST COMPLETED")
print("=" * 70)

print("\nFiles generated:")
print(f"- {results_path}")
print(f"- {importance_path}")

print("\nDONE.")