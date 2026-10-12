import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VALIDATION_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


# ============================================================
# FEATURES
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(text):
    print("\n" + "=" * 75)
    print(text)
    print("=" * 75)


def evaluate_model(model, X, y, model_name, split_name):
    """
    Evaluate one trained model.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    roc_auc = roc_auc_score(y, probabilities)
    loss = log_loss(y, probabilities)

    print(f"\n{model_name} - {split_name}")
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"Log Loss : {loss:.4f}")

    cm = confusion_matrix(y, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Player 2 wins", "Player 1 wins"],
            digits=4
        )
    )

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss,
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": cm
    }


def validate_columns(df, name):
    missing = [f for f in FEATURES + [TARGET] if f not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing these columns:\n{missing}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print_header("FINAL MODEL EVALUATION")

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train      : {len(train):,} rows")
print(f"Validation : {len(validation):,} rows")
print(f"Test       : {len(test):,} rows")


# ============================================================
# DATA CHECK
# ============================================================

print_header("DATA CHECK")

validate_columns(train, "Train")
validate_columns(validation, "Validation")
validate_columns(test, "Test")

print("All required columns are present.")

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    missing = df[FEATURES].isna().sum().sum()

    print(
        f"{name}: {missing} missing feature values"
    )

    if missing > 0:
        raise ValueError(
            f"{name} contains missing feature values."
        )


# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


print("\nFeature count:", len(FEATURES))


# ============================================================
# MODEL DEFINITIONS
# ============================================================

print_header("BUILDING MODELS")


# ------------------------------------------------------------
# 1. LOGISTIC REGRESSION
# ------------------------------------------------------------

logistic_model = Pipeline([
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
])


# ------------------------------------------------------------
# 2. RANDOM FOREST
# Best validation configuration from your tuning
# ------------------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# 3. XGBOOST
# Fixed configuration for final comparison
# ------------------------------------------------------------

if XGBOOST_AVAILABLE:

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

else:
    xgb_model = None
    print("\nWARNING: XGBoost is not installed.")
    print("Install it using:")
    print("pip install xgboost")


models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

if xgb_model is not None:
    models["XGBoost"] = xgb_model


# ============================================================
# TRAINING + VALIDATION
# ============================================================

print_header("TRAINING AND VALIDATION")

validation_results = []
trained_models = {}


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    trained_models[model_name] = model

    result = evaluate_model(
        model,
        X_val,
        y_val,
        model_name,
        "Validation"
    )

    validation_results.append(result)


# ============================================================
# VALIDATION COMPARISON
# ============================================================

print_header("VALIDATION MODEL COMPARISON")

comparison = pd.DataFrame([
    {
        "Model": r["model"],
        "Accuracy": r["accuracy"],
        "F1": r["f1"],
        "ROC_AUC": r["roc_auc"],
        "Log_Loss": r["log_loss"]
    }
    for r in validation_results
])

comparison = comparison.sort_values(
    by="ROC_AUC",
    ascending=False
)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# MODEL SELECTION
# ============================================================
#
# IMPORTANT:
# We select using VALIDATION ROC-AUC.
# The TEST set is NOT used here.
# ============================================================

best_model_name = comparison.iloc[0]["Model"]

print_header("FINAL MODEL SELECTION")

print("Selection criterion : Validation ROC-AUC")
print(f"Selected model      : {best_model_name}")
print(
    f"Validation ROC-AUC  : "
    f"{comparison.iloc[0]['ROC_AUC']:.4f}"
)

print(
    "\nThe test dataset has NOT been used to select the model."
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print_header("FINAL TEST EVALUATION")

final_model = trained_models[best_model_name]

test_result = evaluate_model(
    final_model,
    X_test,
    y_test,
    best_model_name,
    "FINAL TEST"
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_to_save = comparison.copy()

comparison_to_save.to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_model_validation_comparison.csv"
    ),
    index=False
)


# ============================================================
# SAVE FINAL TEST RESULT
# ============================================================

final_result_df = pd.DataFrame([{
    "model": best_model_name,
    "test_accuracy": test_result["accuracy"],
    "test_f1": test_result["f1"],
    "test_roc_auc": test_result["roc_auc"],
    "test_log_loss": test_result["log_loss"]
}])

final_result_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_test_results.csv"
    ),
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print_header("CREATING CONFUSION MATRIX")

plt.figure(figsize=(7, 6))

sns.heatmap(
    test_result["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Player 2 wins", "Player 1 wins"],
    yticklabels=["Player 2 wins", "Player 1 wins"]
)

plt.title(
    f"{best_model_name} - Final Test Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "final_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVES
# ============================================================

print("Creating ROC curve...")

plt.figure(figsize=(9, 7))

for model_name, model in trained_models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC={auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Model Comparison")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "roc_curve_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVES
# ============================================================

print("Creating Precision-Recall curve...")

plt.figure(figsize=(9, 7))

for model_name, model in trained_models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    ap = average_precision_score(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (AP={ap:.3f})"
    )


plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Model Comparison"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "precision_recall_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# MODEL COMPARISON - ACCURACY
# ============================================================

print("Creating model comparison charts...")

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"] * 100
)

plt.ylabel("Validation Accuracy (%)")
plt.xlabel("Model")

plt.title(
    "Validation Accuracy Comparison"
)

plt.xticks(
    rotation=20,
    ha="right"
)

for i, value in enumerate(comparison["Accuracy"] * 100):
    plt.text(
        i,
        value + 0.2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "validation_accuracy_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# MODEL COMPARISON - ROC-AUC
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["ROC_AUC"]
)

plt.ylabel("Validation ROC-AUC")
plt.xlabel("Model")

plt.title(
    "Validation ROC-AUC Comparison"
)

plt.xticks(
    rotation=20,
    ha="right"
)

for i, value in enumerate(comparison["ROC_AUC"]):
    plt.text(
        i,
        value + 0.002,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "validation_roc_auc_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("Creating feature importance chart...")

if best_model_name == "Random Forest":

    importances = final_model.feature_importances_

elif best_model_name == "XGBoost":

    importances = final_model.feature_importances_

elif best_model_name == "Logistic Regression":

    importances = np.abs(
        final_model.named_steps[
            "classifier"
        ].coef_[0]
    )

else:

    importances = None


if importances is not None:

    feature_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    feature_importance.to_csv(
        os.path.join(
            RESULTS_DIR,
            "final_feature_importance.csv"
        ),
        index=False
    )

    top_features = feature_importance.head(15)

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        f"Top 15 Features - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "final_feature_importance.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("FINAL MODEL SUMMARY")

print(f"Selected model : {best_model_name}")

print("\nValidation performance:")
print(
    comparison[
        comparison["Model"] == best_model_name
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nFINAL TEST PERFORMANCE:")

print(
    f"Accuracy : {test_result['accuracy']:.4f}"
)

print(
    f"F1 Score : {test_result['f1']:.4f}"
)

print(
    f"ROC-AUC  : {test_result['roc_auc']:.4f}"
)

print(
    f"Log Loss : {test_result['log_loss']:.4f}"
)


print_header("FILES CREATED")

print(
    os.path.join(
        RESULTS_DIR,
        "final_model_validation_comparison.csv"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "final_test_results.csv"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "final_feature_importance.csv"
    )
)

print("\nFigures:")

print(
    os.path.join(
        FIGURES_DIR,
        "final_confusion_matrix.png"
    )
)

print(
    os.path.join(
        FIGURES_DIR,
        "roc_curve_comparison.png"
    )
)

print(
    os.path.join(
        FIGURES_DIR,
        "precision_recall_comparison.png"
    )
)

print(
    os.path.join(
        FIGURES_DIR,
        "validation_accuracy_comparison.png"
    )
)

print(
    os.path.join(
        FIGURES_DIR,
        "validation_roc_auc_comparison.png"
    )
)

print(
    os.path.join(
        FIGURES_DIR,
        "final_feature_importance.png"
    )
)

print_header("FINAL EVALUATION COMPLETED")