from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "atp_matches_1997_2024.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"

OUTPUT_PATH = OUTPUT_DIR / "atp_matches_clean.csv"


# Tournament levels we will use
#
# A = ATP Tour
# M = Masters
# G = Grand Slam
# F = Tour Finals
#
# We exclude:
# D = Davis Cup
# Other non-target competition levels

ALLOWED_LEVELS = ["A", "M", "G", "F"]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df):,} rows.")

    return df


# ============================================================
# DATE CLEANING
# ============================================================

def clean_dates(df):

    print("\nCleaning dates...")

    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"],
        errors="coerce"
    )

    invalid_dates = df["tourney_date"].isna().sum()

    print(f"Invalid dates found: {invalid_dates:,}")

    # Remove records with no valid match date
    df = df.dropna(
        subset=["tourney_date"]
    )

    # Year should come from actual tournament date
    df["year"] = df["tourney_date"].dt.year

    return df


# ============================================================
# PLAYER ID CLEANING
# ============================================================

def clean_player_ids(df):

    print("\nCleaning player IDs...")

    before = len(df)

    df = df.dropna(
        subset=["winner_id", "loser_id"]
    )

    # Winner and loser obviously cannot be the same person
    df = df[
        df["winner_id"] != df["loser_id"]
    ]

    removed = before - len(df)

    print(f"Rows removed: {removed:,}")

    return df


# ============================================================
# SURFACE CLEANING
# ============================================================

def clean_surface(df):

    print("\nCleaning surface values...")

    print("Before:")
    print(
        df["surface"]
        .value_counts(dropna=False)
    )

    # Surface is essential for our surface-specific features.
    # We cannot reliably infer a missing surface.
    before = len(df)

    df = df.dropna(
        subset=["surface"]
    )

    removed = before - len(df)

    print(
        f"\nRows removed because surface was missing: "
        f"{removed:,}"
    )

    return df


# ============================================================
# TOURNAMENT LEVEL FILTER
# ============================================================

def filter_tournament_levels(df):

    print("\nFiltering tournament levels...")

    print("Before filtering:")

    print(
        df["tourney_level"]
        .value_counts(dropna=False)
    )

    before = len(df)

    df = df[
        df["tourney_level"].isin(
            ALLOWED_LEVELS
        )
    ].copy()

    removed = before - len(df)

    print(
        f"\nRows removed: {removed:,}"
    )

    print("\nAfter filtering:")

    print(
        df["tourney_level"]
        .value_counts(dropna=False)
    )

    return df


# ============================================================
# RANKING CLEANING
# ============================================================

def clean_rankings(df):

    print("\nCleaning rankings...")

    ranking_columns = [
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points"
    ]

    for column in ranking_columns:

        # Convert to numeric
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Invalid ranking values become missing.
    # We DO NOT delete these rows.
    for column in [
        "winner_rank",
        "loser_rank"
    ]:

        invalid = (
            df[column].notna()
            & (df[column] <= 0)
        )

        print(
            f"{column}: "
            f"{invalid.sum():,} invalid values"
        )

        df.loc[
            invalid,
            column
        ] = pd.NA

    # Ranking points cannot be negative
    for column in [
        "winner_rank_points",
        "loser_rank_points"
    ]:

        invalid = (
            df[column].notna()
            & (df[column] < 0)
        )

        print(
            f"{column}: "
            f"{invalid.sum():,} invalid values"
        )

        df.loc[
            invalid,
            column
        ] = pd.NA

    return df


# ============================================================
# NUMERIC COLUMN CLEANING
# ============================================================

def convert_numeric_columns(df):

    print("\nConverting numeric columns...")

    numeric_columns = [
        "draw_size",
        "match_num",
        "winner_seed",
        "winner_ht",
        "winner_age",
        "loser_seed",
        "loser_ht",
        "loser_age",
        "minutes",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

def sort_data(df):

    print("\nSorting chronologically...")

    df = df.sort_values(
        by=[
            "tourney_date",
            "tourney_id",
            "match_num"
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# FINAL DATA CHECK
# ============================================================

def final_check(df):

    print("\n" + "=" * 70)
    print("FINAL CLEAN DATASET CHECK")
    print("=" * 70)

    print(
        f"Total matches : {len(df):,}"
    )

    print(
        f"Unique players: "
        f"{len(set(df['winner_id']) | set(df['loser_id'])):,}"
    )

    print(
        f"Date range    : "
        f"{df['tourney_date'].min().date()} → "
        f"{df['tourney_date'].max().date()}"
    )

    print("\nSurfaces:")

    print(
        df["surface"]
        .value_counts()
    )

    print("\nTournament levels:")

    print(
        df["tourney_level"]
        .value_counts()
    )

    print("\nMissing values in critical columns:")

    critical_columns = [
        "tourney_date",
        "winner_id",
        "loser_id",
        "surface",
        "tourney_level"
    ]

    for column in critical_columns:

        print(
            f"{column:20} "
            f"{df[column].isna().sum():,}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    df = clean_dates(df)

    df = clean_player_ids(df)

    df = clean_surface(df)

    df = filter_tournament_levels(df)

    df = clean_rankings(df)

    df = convert_numeric_columns(df)

    df = sort_data(df)

    final_check(df)

    # Create output directory
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    # Save cleaned dataset
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nClean dataset saved to:")

    print(OUTPUT_PATH)

    print("\nDATA CLEANING COMPLETED SUCCESSFULLY.")