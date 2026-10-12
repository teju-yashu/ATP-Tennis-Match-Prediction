import os
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)

# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)

# ============================================================
# 2. TRAIN FINAL RANDOM FOREST
# ============================================================

print("=" * 60)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 60)

rf.fit(X_train, y_train)

print("Training completed.")

# ============================================================
# 3. VALIDATION EVALUATION
# ============================================================

val_probability = rf.predict_proba(X_val)[:, 1]
val_prediction = (val_probability >= 0.5).astype(int)

val_accuracy = accuracy_score(y_val, val_prediction)
val_f1 = f1_score(y_val, val_prediction)
val_auc = roc_auc_score(y_val, val_probability)
val_logloss = log_loss(y_val, val_probability)
val_brier = brier_score_loss(y_val, val_probability)

print("\n" + "=" * 60)
print("FINAL RF - VALIDATION")
print("=" * 60)

print(f"Accuracy : {val_accuracy:.4f}")
print(f"F1       : {val_f1:.4f}")
print(f"ROC-AUC  : {val_auc:.4f}")
print(f"Log Loss : {val_logloss:.4f}")
print(f"Brier    : {val_brier:.4f}")

# ============================================================
# 4. SAVE MODEL
# ============================================================

model_path = os.path.join(MODEL_DIR, "random_forest.pkl")

joblib.dump(rf, model_path)

print(f"\nModel saved to: {model_path}")

# ============================================================
# 5. SAVE FEATURE ORDER
# ============================================================

feature_names = list(X_train.columns)

feature_path = os.path.join(MODEL_DIR, "feature_order.json")

with open(feature_path, "w") as f:
    json.dump(feature_names, f, indent=4)

print(f"Feature order saved to: {feature_path}")

# ============================================================
# 6. SAVE MODEL METADATA
# ============================================================

metadata = {
    "model": "RandomForestClassifier",

    "parameters": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "random_state": 42
    },

    "threshold": 0.5,

    "number_of_features": len(feature_names),

    "validation_metrics": {
        "accuracy": float(val_accuracy),
        "f1": float(val_f1),
        "roc_auc": float(val_auc),
        "log_loss": float(val_logloss),
        "brier_score": float(val_brier)
    }
}

metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved to: {metadata_path}")

# ============================================================
# 7. VERIFY SAVED MODEL
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING SAVED MODEL")
print("=" * 60)

loaded_model = joblib.load(model_path)

test_probability = loaded_model.predict_proba(X_val)[:, 1]

print("Saved model loaded successfully.")
print(f"Sample probability: {test_probability[0]:.4f}")

print("\n" + "=" * 60)
print("MODEL FREEZE COMPLETE")
print("=" * 60)