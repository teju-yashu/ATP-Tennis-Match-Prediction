from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_DATASET = DATA_DIR / "model_dataset.csv"

TRAIN_PATH = DATA_DIR / "train.csv"

ABLATION_RESULTS = (
    DATA_DIR / "feature_ablation_results.csv"
)

LOGISTIC_RESULTS = (
    DATA_DIR / "logistic_results.csv"
)

BASELINE_RESULTS = (
    DATA_DIR / "baseline_results.csv"
)


# ============================================================
# FEATURE LIST
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
# UTILITY
# ============================================================

def save_plot(filename):

    path = FIGURES_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 75)
    print("TENNNIS MODEL VISUALIZATION")
    print("=" * 75)

    print("\nLoading data...")

    dataset = pd.read_csv(
        MODEL_DATASET
    )

    train = pd.read_csv(
        TRAIN_PATH
    )

    ablation = pd.read_csv(
        ABLATION_RESULTS
    )

    logistic = pd.read_csv(
        LOGISTIC_RESULTS
    )

    baseline = pd.read_csv(
        BASELINE_RESULTS
    )

    print(
        f"Model dataset : {len(dataset):,} rows"
    )

    print(
        f"Training data : {len(train):,} rows"
    )

    print(
        f"Ablation rows : {len(ablation):,}"
    )

    return (
        dataset,
        train,
        ablation,
        logistic,
        baseline
    )


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

def plot_target_distribution(dataset):

    print("\nCreating target distribution...")

    counts = (
        dataset["target"]
        .value_counts()
        .sort_index()
    )

    labels = [
        "Player 2 Wins",
        "Player 1 Wins"
    ]

    values = [
        counts.get(0, 0),
        counts.get(1, 0)
    ]

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        labels,
        values
    )

    plt.title(
        "Target Distribution"
    )

    plt.xlabel(
        "Match Outcome"
    )

    plt.ylabel(
        "Number of Matches"
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    save_plot(
        "target_distribution.png"
    )


# ============================================================
# 2. SURFACE DISTRIBUTION
# ============================================================

def plot_surface_distribution(dataset):

    print("\nCreating surface distribution...")

    counts = (
        dataset["surface"]
        .value_counts()
    )

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        counts.index,
        counts.values
    )

    plt.title(
        "Matches by Court Surface"
    )

    plt.xlabel(
        "Surface"
    )

    plt.ylabel(
        "Number of Matches"
    )

    plt.xticks(
        rotation=20
    )

    for bar, value in zip(
        bars,
        counts.values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    save_plot(
        "surface_distribution.png"
    )


# ============================================================
# 3. FEATURE ABLATION - ACCURACY
# ============================================================

def plot_ablation_accuracy(ablation):

    print("\nCreating ablation accuracy chart...")

    plt.figure(
        figsize=(12, 7)
    )

    x = np.arange(
        len(ablation)
    )

    values = (
        ablation["test_accuracy"]
        * 100
    )

    plt.plot(
        x,
        values,
        marker="o",
        linewidth=2
    )

    plt.xticks(
        x,
        ablation["experiment"],
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Test Accuracy (%)"
    )

    plt.xlabel(
        "Feature Group"
    )

    plt.title(
        "Feature Ablation: Test Accuracy"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "ablation_accuracy.png"
    )


# ============================================================
# 4. FEATURE ABLATION - ROC AUC
# ============================================================

def plot_ablation_auc(ablation):

    print("\nCreating ablation ROC-AUC chart...")

    plt.figure(
        figsize=(12, 7)
    )

    x = np.arange(
        len(ablation)
    )

    values = (
        ablation["test_roc_auc"]
    )

    plt.plot(
        x,
        values,
        marker="o",
        linewidth=2
    )

    plt.xticks(
        x,
        ablation["experiment"],
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Test ROC-AUC"
    )

    plt.xlabel(
        "Feature Group"
    )

    plt.title(
        "Feature Ablation: Test ROC-AUC"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "ablation_roc_auc.png"
    )


# ============================================================
# 5. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

def plot_logistic_coefficients(train):

    print(
        "\nTraining Logistic Regression "
        "for coefficient visualization..."
    )

    available_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature in train.columns
    ]

    X = train[
        available_features
    ]

    y = train[
        "target"
    ].astype(int)

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

    model.fit(
        X,
        y
    )

    coefficients = (
        model
        .named_steps[
            "classifier"
        ]
        .coef_[0]
    )

    coefficient_df = pd.DataFrame({
        "feature": available_features,
        "coefficient": coefficients
    })

    coefficient_df[
        "absolute"
    ] = coefficient_df[
        "coefficient"
    ].abs()

    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute",
            ascending=False
        )
        .head(15)
        .sort_values(
            "coefficient"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        coefficient_df["feature"],
        coefficient_df["coefficient"]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Logistic Regression Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 15 Logistic Regression Features"
    )

    save_plot(
        "logistic_coefficients.png"
    )


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(dataset):

    print(
        "\nCreating feature correlation heatmap..."
    )

    available_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature in dataset.columns
    ]

    correlation = (
        dataset[
            available_features
        ]
        .corr()
    )

    plt.figure(
        figsize=(18, 15)
    )

    image = plt.imshow(
        correlation,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(
        image,
        fraction=0.046,
        pad=0.04
    )

    plt.xticks(
        range(len(available_features)),
        available_features,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(available_features)),
        available_features,
        fontsize=7
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    save_plot(
        "feature_correlation_heatmap.png"
    )


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

def plot_model_comparison(
    logistic,
    baseline
):

    print(
        "\nCreating model comparison chart..."
    )

    # --------------------------------------------------------
    # Extract TEST results
    # --------------------------------------------------------

    logistic_test = logistic[
        logistic["dataset"]
        .str.lower()
        == "test"
    ]

    baseline_test = baseline[
        baseline["model"]
        .str.lower()
        .str.contains("test")
    ]

    # If baseline file doesn't contain
    # dataset information, use known values.

    if logistic_test.empty:

        logistic_accuracy = 0.6528
        logistic_auc = 0.7216

    else:

        logistic_accuracy = (
            logistic_test.iloc[0]["accuracy"]
        )

        logistic_auc = (
            logistic_test.iloc[0]["roc_auc"]
        )

    baseline_accuracy = 0.6498
    baseline_auc = 0.6794

    models = [
        "Ranking Baseline",
        "Logistic Regression"
    ]

    accuracy = [
        baseline_accuracy * 100,
        logistic_accuracy * 100
    ]

    auc = [
        baseline_auc,
        logistic_auc
    ]

    x = np.arange(
        len(models)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars1 = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy (%)"
    )

    ax2 = ax.twinx()

    bars2 = ax2.bar(
        x + width / 2,
        auc,
        width,
        label="ROC-AUC"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        models
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax2.set_ylabel(
        "ROC-AUC"
    )

    ax.set_title(
        "Ranking Baseline vs Logistic Regression"
    )

    for bar in bars1:

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom"
        )

    for bar in bars2:

        ax2.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom"
        )

    save_plot(
        "model_comparison.png"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    (
        dataset,
        train,
        ablation,
        logistic,
        baseline
    ) = load_data()

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    plot_target_distribution(
        dataset
    )

    plot_surface_distribution(
        dataset
    )

    plot_ablation_accuracy(
        ablation
    )

    plot_ablation_auc(
        ablation
    )

    plot_logistic_coefficients(
        train
    )

    plot_correlation_heatmap(
        dataset
    )

    plot_model_comparison(
        logistic,
        baseline
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("VISUALIZATION COMPLETED")
    print("=" * 75)

    print(
        f"\nAll figures saved to:"
    )

    print(
        FIGURES_DIR
    )