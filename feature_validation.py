from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "model_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Model dataset not found:\n{INPUT_PATH}"
        )

    print("Loading model dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["match_date"]
    )

    print(f"Loaded {len(df):,} rows.")

    return df


# ============================================================
# BASIC CHECK
# ============================================================

def basic_check(df):

    print("\n" + "=" * 70)
    print("1. BASIC DATASET CHECK")
    print("=" * 70)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    print("\nDate range:")

    print(
        f"{df['match_date'].min().date()} "
        f"→ "
        f"{df['match_date'].max().date()}"
    )


# ============================================================
# TARGET CHECK
# ============================================================

def target_check(df):

    print("\n" + "=" * 70)
    print("2. TARGET CHECK")
    print("=" * 70)

    print("Target values:")

    print(
        df["target"].value_counts()
        .sort_index()
    )

    print("\nTarget percentages:")

    print(
        (
            df["target"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    invalid_targets = (
        ~df["target"].isin([0, 1])
    ).sum()

    print(
        f"\nInvalid target values: "
        f"{invalid_targets:,}"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

def duplicate_check(df):

    print("\n" + "=" * 70)
    print("3. DUPLICATE CHECK")
    print("=" * 70)

    exact_duplicates = df.duplicated().sum()

    print(
        f"Exact duplicate rows: "
        f"{exact_duplicates:,}"
    )

    # Same match + same player pair
    match_duplicates = df.duplicated(
        subset=[
            "tourney_id",
            "player1_id",
            "player2_id"
        ]
    ).sum()

    print(
        f"Duplicate match/player combinations: "
        f"{match_duplicates:,}"
    )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

def missing_value_check(df):

    print("\n" + "=" * 70)
    print("4. MISSING VALUE CHECK")
    print("=" * 70)

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if missing.empty:

        print("No missing values.")

        return

    result = pd.DataFrame({
        "Missing": missing,
        "Percentage": (
            missing / len(df) * 100
        ).round(3)
    })

    print(result.to_string())


# ============================================================
# INFINITE VALUE CHECK
# ============================================================

def infinite_value_check(df):

    print("\n" + "=" * 70)
    print("5. INFINITE VALUE CHECK")
    print("=" * 70)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinite_counts = np.isinf(
        numeric_df
    ).sum()

    infinite_counts = infinite_counts[
        infinite_counts > 0
    ]

    if infinite_counts.empty:

        print("No infinite values found.")

    else:

        print(infinite_counts)


# ============================================================
# FEATURE RANGE CHECK
# ============================================================

def feature_range_check(df):

    print("\n" + "=" * 70)
    print("6. FEATURE RANGE CHECK")
    print("=" * 70)

    feature_columns = [
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
        "h2h_win_rate_difference",
        "recent_h2h_win_rate_difference",
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

    existing = [
        col
        for col in feature_columns
        if col in df.columns
    ]

    summary = df[existing].describe().T

    print(
        summary[
            [
                "min",
                "max",
                "mean",
                "std"
            ]
        ].round(4).to_string()
    )


# ============================================================
# PROBABILITY FEATURE CHECK
# ============================================================

def probability_feature_check(df):

    print("\n" + "=" * 70)
    print("7. PROBABILITY/RATE FEATURE CHECK")
    print("=" * 70)

    rate_features = [
        "overall_win_rate_difference",
        "win_rate_last_5_difference",
        "win_rate_last_10_difference",
        "win_rate_last_20_difference",
        "recency_weighted_form_difference",
        "win_rate_last_30_days_difference",
        "surface_win_rate_difference",
        "surface_recent_win_rate_difference",
        "h2h_win_rate_difference",
        "recent_h2h_win_rate_difference",
        "surface_h2h_win_rate_difference",
        "tournament_win_rate_difference",
        "ace_rate_difference",
        "double_fault_rate_difference",
        "first_serve_percentage_difference",
        "first_serve_win_percentage_difference",
        "second_serve_win_percentage_difference",
        "break_point_save_percentage_difference"
    ]

    for column in rate_features:

        if column not in df.columns:
            continue

        series = df[column].dropna()

        invalid = (
            (series < -1)
            | (series > 1)
        ).sum()

        print(
            f"{column:45} "
            f"invalid = {invalid:,}"
        )


# ============================================================
# PLAYER CONSISTENCY CHECK
# ============================================================

def player_consistency_check(df):

    print("\n" + "=" * 70)
    print("8. PLAYER CONSISTENCY CHECK")
    print("=" * 70)

    same_players = (
        df["player1_id"]
        == df["player2_id"]
    ).sum()

    print(
        f"Player 1 == Player 2: "
        f"{same_players:,}"
    )

    print(
        f"Unique Player 1 IDs: "
        f"{df['player1_id'].nunique():,}"
    )

    print(
        f"Unique Player 2 IDs: "
        f"{df['player2_id'].nunique():,}"
    )


# ============================================================
# CHRONOLOGICAL CHECK
# ============================================================

def chronological_check(df):

    print("\n" + "=" * 70)
    print("9. CHRONOLOGICAL CHECK")
    print("=" * 70)

    sorted_dates = df["match_date"].is_monotonic_increasing

    print(
        f"Dataset chronologically sorted: "
        f"{sorted_dates}"
    )

    if not sorted_dates:

        print(
            "\nWARNING: Dataset is not sorted chronologically."
        )

        return

    print(
        "No chronological ordering problem detected."
    )


# ============================================================
# TRAIN / VALIDATION / TEST COUNTS
# ============================================================

def split_preview(df):

    print("\n" + "=" * 70)
    print("10. CHRONOLOGICAL SPLIT PREVIEW")
    print("=" * 70)

    train = df[
        df["match_date"].dt.year <= 2022
    ]

    validation = df[
        df["match_date"].dt.year == 2023
    ]

    test = df[
        df["match_date"].dt.year == 2024
    ]

    print(
        f"TRAIN      1997–2022 : {len(train):,}"
    )

    print(
        f"VALIDATION 2023      : {len(validation):,}"
    )

    print(
        f"TEST       2024      : {len(test):,}"
    )

    print(
        f"TOTAL                : "
        f"{len(train) + len(validation) + len(test):,}"
    )

    print("\nTarget distribution:")

    print(
        "\nTRAIN:"
    )

    print(
        train["target"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(
        "\nVALIDATION:"
    )

    print(
        validation["target"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(
        "\nTEST:"
    )

    print(
        test["target"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


# ============================================================
# CORRELATION CHECK
# ============================================================

def correlation_check(df):

    print("\n" + "=" * 70)
    print("11. FEATURE CORRELATION WITH TARGET")
    print("=" * 70)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    correlations = (
        numeric_df.corr()["target"]
        .drop("target")
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    print(
        correlations.head(15)
        .round(4)
        .to_string()
    )


# ============================================================
# CONSTANT FEATURE CHECK
# ============================================================

def constant_feature_check(df):

    print("\n" + "=" * 70)
    print("12. CONSTANT FEATURE CHECK")
    print("=" * 70)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    constant_columns = []

    for column in numeric_columns:

        if df[column].nunique(
            dropna=False
        ) <= 1:

            constant_columns.append(
                column
            )

    if not constant_columns:

        print(
            "No constant numeric features found."
        )

    else:

        print(
            "Constant features:"
        )

        for column in constant_columns:
            print(
                f"  - {column}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TENNIS MODEL DATASET VALIDATION")
    print("=" * 70)

    df = load_dataset()

    basic_check(df)

    target_check(df)

    duplicate_check(df)

    missing_value_check(df)

    infinite_value_check(df)

    feature_range_check(df)

    probability_feature_check(df)

    player_consistency_check(df)

    chronological_check(df)

    split_preview(df)

    correlation_check(df)

    constant_feature_check(df)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)