from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "atp_matches_1997_2024.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# BASIC INFORMATION
# ============================================================

def basic_information(df):

    print("\n" + "=" * 70)
    print("1. BASIC DATASET INFORMATION")
    print("=" * 70)

    print(f"Rows                : {len(df):,}")
    print(f"Columns             : {len(df.columns):,}")

    print(
        f"Memory usage        : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    print("\nData types:")
    print(df.dtypes)


# ============================================================
# DATE AUDIT
# ============================================================

def date_audit(df):

    print("\n" + "=" * 70)
    print("2. DATE AUDIT")
    print("=" * 70)

    dates = pd.to_datetime(
        df["tourney_date"],
        errors="coerce"
    )

    print(f"Earliest match date : {dates.min()}")
    print(f"Latest match date   : {dates.max()}")

    print("\nMatches by year:")

    print(
        dates.dt.year.value_counts()
        .sort_index()
        .to_string()
    )

    invalid_dates = dates.isna().sum()

    print(
        f"\nInvalid/missing dates: {invalid_dates:,}"
    )


# ============================================================
# MISSING VALUE AUDIT
# ============================================================

def missing_value_audit(df):

    print("\n" + "=" * 70)
    print("3. MISSING VALUE AUDIT")
    print("=" * 70)

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if missing.empty:

        print("No missing values found.")

        return

    missing_percentage = (
        missing / len(df) * 100
    )

    result = pd.DataFrame({
        "Missing": missing,
        "Percentage": missing_percentage.round(2)
    })

    print(result.to_string())


# ============================================================
# DUPLICATE AUDIT
# ============================================================

def duplicate_audit(df):

    print("\n" + "=" * 70)
    print("4. DUPLICATE AUDIT")
    print("=" * 70)

    exact_duplicates = df.duplicated().sum()

    print(
        f"Exact duplicate rows: {exact_duplicates:,}"
    )

    # Check duplicates based on tournament + match number
    if "tourney_id" in df.columns and "match_num" in df.columns:

        match_duplicates = df.duplicated(
            subset=["tourney_id", "match_num"]
        ).sum()

        print(
            f"Duplicate tournament/match IDs: "
            f"{match_duplicates:,}"
        )


# ============================================================
# PLAYER AUDIT
# ============================================================

def player_audit(df):

    print("\n" + "=" * 70)
    print("5. PLAYER AUDIT")
    print("=" * 70)

    winner_players = set(
        df["winner_id"].dropna()
    )

    loser_players = set(
        df["loser_id"].dropna()
    )

    all_players = winner_players | loser_players

    print(
        f"Unique winner IDs : {len(winner_players):,}"
    )

    print(
        f"Unique loser IDs  : {len(loser_players):,}"
    )

    print(
        f"Unique players    : {len(all_players):,}"
    )

    missing_winner_id = df["winner_id"].isna().sum()
    missing_loser_id = df["loser_id"].isna().sum()

    print(
        f"\nMissing winner IDs: {missing_winner_id:,}"
    )

    print(
        f"Missing loser IDs : {missing_loser_id:,}"
    )


# ============================================================
# SURFACE AUDIT
# ============================================================

def surface_audit(df):

    print("\n" + "=" * 70)
    print("6. SURFACE AUDIT")
    print("=" * 70)

    print(
        df["surface"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        f"\nMissing surface values: "
        f"{df['surface'].isna().sum():,}"
    )


# ============================================================
# TOURNAMENT AUDIT
# ============================================================

def tournament_audit(df):

    print("\n" + "=" * 70)
    print("7. TOURNAMENT AUDIT")
    print("=" * 70)

    print("\nTournament levels:")

    print(
        df["tourney_level"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nUnique tournaments:")

    print(
        f"{df['tourney_name'].nunique():,}"
    )


# ============================================================
# RANKING AUDIT
# ============================================================

def ranking_audit(df):

    print("\n" + "=" * 70)
    print("8. RANKING AUDIT")
    print("=" * 70)

    ranking_columns = [
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points"
    ]

    for column in ranking_columns:

        if column not in df.columns:
            print(
                f"{column}: COLUMN NOT FOUND"
            )
            continue

        series = df[column]

        print(f"\n{column}")

        print(
            f"  Missing : {series.isna().sum():,}"
        )

        print(
            f"  Minimum : {series.min()}"
        )

        print(
            f"  Maximum : {series.max()}"
        )

        print(
            f"  Median  : {series.median()}"
        )


# ============================================================
# PLAYER ATTRIBUTE AUDIT
# ============================================================

def player_attribute_audit(df):

    print("\n" + "=" * 70)
    print("9. PLAYER ATTRIBUTE AUDIT")
    print("=" * 70)

    columns = [
        "winner_age",
        "loser_age",
        "winner_ht",
        "loser_ht",
        "winner_hand",
        "loser_hand"
    ]

    for column in columns:

        if column not in df.columns:
            continue

        print(
            f"\n{column}"
        )

        print(
            f"  Missing: {df[column].isna().sum():,}"
        )

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            print(
                f"  Min    : {df[column].min()}"
            )

            print(
                f"  Max    : {df[column].max()}"
            )

            print(
                f"  Median : {df[column].median()}"
            )


# ============================================================
# MATCH STATISTICS AUDIT
# ============================================================

def match_statistics_audit(df):

    print("\n" + "=" * 70)
    print("10. MATCH STATISTICS AUDIT")
    print("=" * 70)

    statistics = [
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

    for column in statistics:

        if column not in df.columns:
            continue

        missing = df[column].isna().sum()

        print(
            f"{column:12} "
            f"missing = {missing:,} "
            f"({missing / len(df) * 100:.2f}%)"
        )


# ============================================================
# DATA CONSISTENCY AUDIT
# ============================================================

def consistency_audit(df):

    print("\n" + "=" * 70)
    print("11. DATA CONSISTENCY AUDIT")
    print("=" * 70)

    # Winner and loser should not be the same player
    same_player = (
        df["winner_id"] == df["loser_id"]
    ).sum()

    print(
        f"Winner == Loser: {same_player:,}"
    )

    # Ranking should normally be positive
    for column in [
        "winner_rank",
        "loser_rank"
    ]:

        invalid = (
            df[column].notna()
            & (df[column] <= 0)
        ).sum()

        print(
            f"{column} <= 0: {invalid:,}"
        )

    # Ranking points should not be negative
    for column in [
        "winner_rank_points",
        "loser_rank_points"
    ]:

        invalid = (
            df[column].notna()
            & (df[column] < 0)
        ).sum()

        print(
            f"{column} < 0: {invalid:,}"
        )


# ============================================================
# FEATURE AVAILABILITY CHECK
# ============================================================

def important_columns_check(df):

    print("\n" + "=" * 70)
    print("12. IMPORTANT COLUMN CHECK")
    print("=" * 70)

    required_columns = [
        "tourney_date",
        "winner_id",
        "loser_id",
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points",
        "surface",
        "winner_age",
        "loser_age",
        "winner_ht",
        "loser_ht",
        "winner_hand",
        "loser_hand"
    ]

    for column in required_columns:

        if column in df.columns:

            print(f"[OK]      {column}")

        else:

            print(f"[MISSING] {column}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    basic_information(df)

    date_audit(df)

    missing_value_audit(df)

    duplicate_audit(df)

    player_audit(df)

    surface_audit(df)

    tournament_audit(df)

    ranking_audit(df)

    player_attribute_audit(df)

    match_statistics_audit(df)

    consistency_audit(df)

    important_columns_check(df)

    print("\n" + "=" * 70)
    print("DATA AUDIT COMPLETED")
    print("=" * 70)