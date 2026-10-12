from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "model_dataset.csv"

TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

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
# SORT CHRONOLOGICALLY
# ============================================================

def sort_data(df):

    print("\nSorting chronologically...")

    df = df.sort_values(
        [
            "match_date",
            "tourney_id",
            "match_num"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(df):

    print("\nCreating chronological splits...")

    train = df[
        df["match_date"].dt.year <= 2022
    ].copy()

    validation = df[
        df["match_date"].dt.year == 2023
    ].copy()

    test = df[
        df["match_date"].dt.year == 2024
    ].copy()

    print(
        f"Train      : {len(train):,} rows "
        f"(1997–2022)"
    )

    print(
        f"Validation : {len(validation):,} rows "
        f"(2023)"
    )

    print(
        f"Test       : {len(test):,} rows "
        f"(2024)"
    )

    return train, validation, test


# ============================================================
# PLAYER-SPECIFIC HEIGHT IMPUTATION
# ============================================================

def build_player_height_map(df):

    print("\nBuilding player height map...")

    heights = {}

    # The original model dataset only contains height
    # differences, so we cannot reconstruct individual
    # heights perfectly here.
    #
    # Therefore this function is intentionally kept empty.
    #
    # Height differences will be imputed using the TRAINING
    # distribution only.

    return heights


# ============================================================
# CALCULATE TRAINING MEDIANS
# ============================================================

def calculate_training_medians(train):

    print("\nCalculating training-only imputation values...")

    columns = [
        "height_difference",
        "rank_difference",
        "rank_points_difference",
        "age_difference"
    ]

    medians = {}

    for column in columns:

        if column not in train.columns:
            continue

        median_value = train[column].median()

        medians[column] = median_value

        print(
            f"{column:30} "
            f"median = {median_value:.4f}"
        )

    return medians


# ============================================================
# IMPUTE MISSING VALUES
# ============================================================

def impute_missing_values(
    train,
    validation,
    test,
    medians
):

    print("\nHandling missing values...")

    datasets = {
        "TRAIN": train,
        "VALIDATION": validation,
        "TEST": test
    }

    for name, dataset in datasets.items():

        print(f"\n{name}:")

        for column, value in medians.items():

            before = dataset[column].isna().sum()

            dataset[column] = dataset[column].fillna(
                value
            )

            after = dataset[column].isna().sum()

            print(
                f"{column:30} "
                f"{before:5,} → {after:5,}"
            )

    return train, validation, test


# ============================================================
# CHECK FOR REMAINING MISSING VALUES
# ============================================================

def check_missing_values(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("MISSING VALUE CHECK")
    print("=" * 70)

    datasets = {
        "TRAIN": train,
        "VALIDATION": validation,
        "TEST": test
    }

    for name, dataset in datasets.items():

        missing = dataset.isna().sum()

        missing = missing[
            missing > 0
        ]

        print(f"\n{name}:")

        if missing.empty:

            print("No missing values.")

        else:

            print(
                missing.to_string()
            )


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

def check_infinite_values(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("INFINITE VALUE CHECK")
    print("=" * 70)

    datasets = {
        "TRAIN": train,
        "VALIDATION": validation,
        "TEST": test
    }

    for name, dataset in datasets.items():

        numeric = dataset.select_dtypes(
            include=np.number
        )

        infinite = np.isinf(
            numeric
        ).sum().sum()

        print(
            f"{name:12}: "
            f"{infinite:,} infinite values"
        )


# ============================================================
# TARGET CHECK
# ============================================================

def target_check(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    datasets = {
        "TRAIN": train,
        "VALIDATION": validation,
        "TEST": test
    }

    for name, dataset in datasets.items():

        distribution = (
            dataset["target"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )

        print(f"\n{name}:")

        for target, percentage in distribution.items():

            print(
                f"Target {target}: "
                f"{percentage:.2f}%"
            )


# ============================================================
# CHECK CHRONOLOGICAL ORDER
# ============================================================

def chronological_check(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL CHECK")
    print("=" * 70)

    print(
        "TRAIN:"
    )

    print(
        f"{train['match_date'].min().date()} "
        f"→ "
        f"{train['match_date'].max().date()}"
    )

    print(
        "VALIDATION:"
    )

    print(
        f"{validation['match_date'].min().date()} "
        f"→ "
        f"{validation['match_date'].max().date()}"
    )

    print(
        "TEST:"
    )

    print(
        f"{test['match_date'].min().date()} "
        f"→ "
        f"{test['match_date'].max().date()}"
    )

    train_max = train["match_date"].max()
    validation_min = validation["match_date"].min()
    validation_max = validation["match_date"].max()
    test_min = test["match_date"].min()

    if (
        train_max < validation_min
        and validation_max < test_min
    ):

        print(
            "\nPASS: No temporal overlap."
        )

    else:

        print(
            "\nWARNING: Temporal overlap detected."
        )


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(
    train,
    validation,
    test
):

    print("\nSaving prepared datasets...")

    train.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False
    )

    test.to_csv(
        TEST_PATH,
        index=False
    )

    print(
        f"\nTRAIN saved to:\n{TRAIN_PATH}"
    )

    print(
        f"\nVALIDATION saved to:\n{VALIDATION_PATH}"
    )

    print(
        f"\nTEST saved to:\n{TEST_PATH}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("FINAL PREPARED DATA SUMMARY")
    print("=" * 70)

    print(
        f"Training rows      : {len(train):,}"
    )

    print(
        f"Validation rows    : {len(validation):,}"
    )

    print(
        f"Test rows          : {len(test):,}"
    )

    print(
        f"Total rows         : "
        f"{len(train) + len(validation) + len(test):,}"
    )

    print(
        f"Training features  : "
        f"{len(train.columns):,}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TENNIS MODEL DATA PREPARATION")
    print("=" * 70)

    # 1. Load
    df = load_data()

    # 2. Sort
    df = sort_data(df)

    # 3. Split
    train, validation, test = split_data(df)

    # 4. Calculate imputation values ONLY from training data
    medians = calculate_training_medians(train)

    # 5. Apply the training-derived values
    train, validation, test = impute_missing_values(
        train,
        validation,
        test,
        medians
    )

    # 6. Validate
    check_missing_values(
        train,
        validation,
        test
    )

    check_infinite_values(
        train,
        validation,
        test
    )

    target_check(
        train,
        validation,
        test
    )

    chronological_check(
        train,
        validation,
        test
    )

    # 7. Save
    save_datasets(
        train,
        validation,
        test
    )

    # 8. Summary
    final_summary(
        train,
        validation,
        test
    )

    print("\n" + "=" * 70)
    print("MODEL DATA PREPARATION COMPLETED")
    print("=" * 70)