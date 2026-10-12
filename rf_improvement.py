from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)

param_distributions = {
    "n_estimators": [300, 400, 500, 600, 800],
    "max_depth": [8, 10, 12, 14, 16],
    "min_samples_split": [5, 8, 10, 12, 15],
    "min_samples_leaf": [2, 3, 5, 7, 10],
    "max_features": ["sqrt", 0.3, 0.5, 0.7],
    "criterion": ["gini", "entropy", "log_loss"]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,

    # ONLY 30 combinations
    n_iter=30,

    scoring="accuracy",

    # Keep this small because you already have
    # a separate validation set
    cv=3,

    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("\nStarting Randomized RF search...")
print("Testing only 30 configurations...")

random_search.fit(X_train, y_train)

print("\n" + "=" * 60)
print("BEST RANDOM FOREST")
print("=" * 60)

print("\nBest parameters:")
for key, value in random_search.best_params_.items():
    print(f"{key:20s}: {value}")

print(
    f"\nBest CV accuracy: "
    f"{random_search.best_score_:.4f}"
)

best_rf = random_search.best_estimator_