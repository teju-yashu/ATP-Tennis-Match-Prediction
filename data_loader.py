from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "archive" / "tennis_atp"
OUTPUT_DIR = PROJECT_ROOT / "data"

START_YEAR = 1997
END_YEAR = 2024


# ---------------------------------------------------------
# LOAD ATP MATCH DATA
# ---------------------------------------------------------

def load_atp_matches(dataset_dir, start_year, end_year):

    all_matches = []

    for year in range(start_year, end_year + 1):

        filename = dataset_dir / f"atp_matches_{year}.csv"

        if not filename.exists():
            print(f"WARNING: {filename.name} not found")
            continue

        print(f"Loading {year}...")

        df = pd.read_csv(filename)

        df["year"] = year

        all_matches.append(df)

    if not all_matches:
        raise ValueError("No ATP match files were loaded.")

    matches = pd.concat(
        all_matches,
        ignore_index=True
    )

    return matches


# ---------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------

def clean_basic_data(df):

    df = df.copy()

    # Convert tournament date
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Sort chronologically
    df = df.sort_values(
        by=["tourney_date", "match_num"]
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------
# DATA SUMMARY
# ---------------------------------------------------------

def print_summary(df):

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Total matches : {len(df):,}")

    print(
        f"Date range    : "
        f"{df['tourney_date'].min().date()} → "
        f"{df['tourney_date'].max().date()}"
    )

    print(f"Unique players: {len(set(df['winner_id']) | set(df['loser_id'])):,}")

    print("\nSurface distribution:")
    print(df["surface"].value_counts(dropna=False))

    print("\nTournament level:")
    print(df["tourney_level"].value_counts(dropna=False))
    print("\nMissing values:")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(missing)

    print("\nNumber of columns:")
    print(len(df.columns))

    print("\nColumns:")
    print(list(df.columns))

    print("=" * 60)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Could not find dataset directory:\n{DATASET_DIR}"
        )

    print("Loading ATP tennis dataset...")

    matches = load_atp_matches(
        DATASET_DIR,
        START_YEAR,
        END_YEAR
    )

    print("\nCleaning basic data...")

    matches = clean_basic_data(matches)

    print_summary(matches)

    # Create output directory
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "atp_matches_1997_2024.csv"

    matches.to_csv(
        output_path,
        index=False
    )

    print("\nCombined dataset saved to:")

    print(output_path)

    print("\nData loading completed successfully.")