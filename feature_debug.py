from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DATA = PROJECT_ROOT / "data" / "atp_matches_clean.csv"
MODEL_DATA = PROJECT_ROOT / "data" / "model_dataset.csv"


# ============================================================
# LOAD
# ============================================================

def load_data():

    clean = pd.read_csv(
        CLEAN_DATA,
        parse_dates=["tourney_date"]
    )

    model = pd.read_csv(
        MODEL_DATA,
        parse_dates=["match_date"]
    )

    return clean, model


# ============================================================
# SOURCE DATA DUPLICATE CHECK
# ============================================================

def check_source_duplicates(clean):

    print("\n" + "=" * 70)
    print("1. SOURCE DATA MATCH-ID CHECK")
    print("=" * 70)

    print(
        f"Total source matches: {len(clean):,}"
    )

    duplicates = clean[
        clean.duplicated(
            subset=[
                "tourney_id",
                "match_num"
            ],
            keep=False
        )
    ]

    print(
        f"Duplicate tourney_id + match_num rows: "
        f"{len(duplicates):,}"
    )

    if duplicates.empty:

        print(
            "PASS: Every source match has a unique "
            "tourney_id + match_num."
        )

    else:

        print("\nWARNING: These need investigation:")

        print(
            duplicates[
                [
                    "tourney_date",
                    "tourney_id",
                    "match_num",
                    "tourney_name",
                    "winner_id",
                    "winner_name",
                    "loser_id",
                    "loser_name"
                ]
            ]
            .sort_values(
                [
                    "tourney_id",
                    "match_num"
                ]
            )
            .to_string(index=False)
        )


# ============================================================
# PLAYER-PAIR CHECK
# ============================================================

def check_player_pairs(clean):

    print("\n" + "=" * 70)
    print("2. PLAYER-PAIR REPEAT CHECK")
    print("=" * 70)

    df = clean.copy()

    df["player_low"] = df[
        ["winner_id", "loser_id"]
    ].min(axis=1)

    df["player_high"] = df[
        ["winner_id", "loser_id"]
    ].max(axis=1)

    repeated = (
        df.groupby(
            [
                "tourney_id",
                "player_low",
                "player_high"
            ]
        )
        .size()
    )

    repeated = repeated[
        repeated > 1
    ]

    print(
        f"Repeated player pairs inside same tournament: "
        f"{len(repeated):,}"
    )

    if repeated.empty:

        print(
            "No repeated player pairs."
        )

        return

    print(
        "\nThese are NOT automatically duplicates."
    )

    print(
        "We inspect their match_num and dates:"
    )

    for key in repeated.index:

        tournament = key[0]
        player1 = key[1]
        player2 = key[2]

        group = df[
            (df["tourney_id"] == tournament)
            & (
                df["player_low"] == player1
            )
            & (
                df["player_high"] == player2
            )
        ]

        print("\n" + "-" * 70)

        print(
            f"Tournament : {tournament}"
        )

        print(
            f"Players    : {player1} vs {player2}"
        )

        print(
            group[
                [
                    "tourney_date",
                    "match_num",
                    "winner_id",
                    "loser_id",
                    "winner_name",
                    "loser_name"
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# MODEL DATA MATCH-ID CHECK
# ============================================================

def check_model_duplicates(model):

    print("\n" + "=" * 70)
    print("3. MODEL DATA MATCH-ID CHECK")
    print("=" * 70)

    if "match_num" not in model.columns:

        print(
            "ERROR: match_num is missing from "
            "model_dataset.csv"
        )

        return

    duplicates = model[
        model.duplicated(
            subset=[
                "tourney_id",
                "match_num"
            ],
            keep=False
        )
    ]

    print(
        f"Duplicate model match IDs: "
        f"{len(duplicates):,}"
    )

    if duplicates.empty:

        print(
            "PASS: Every model row has a unique "
            "tourney_id + match_num."
        )

    else:

        print(
            "\nWARNING: Model dataset contains "
            "duplicate match IDs:"
        )

        print(
            duplicates[
                [
                    "match_date",
                    "tourney_id",
                    "match_num",
                    "player1_id",
                    "player2_id",
                    "target"
                ]
            ]
            .sort_values(
                [
                    "tourney_id",
                    "match_num"
                ]
            )
            .to_string(index=False)
        )


# ============================================================
# TARGET CONSISTENCY
# ============================================================

def check_target_consistency(model):

    print("\n" + "=" * 70)
    print("4. TARGET CONSISTENCY CHECK")
    print("=" * 70)

    grouped = (
        model.groupby(
            [
                "tourney_id",
                "match_num"
            ]
        )["target"]
        .nunique()
    )

    inconsistent = grouped[
        grouped > 1
    ]

    print(
        f"Match IDs with conflicting targets: "
        f"{len(inconsistent):,}"
    )

    if inconsistent.empty:

        print(
            "PASS: No match has conflicting targets."
        )

    else:

        print(
            "\nCRITICAL:"
        )

        # Get the problematic match IDs
        inconsistent_ids = (
            inconsistent
            .reset_index()
            [
                [
                    "tourney_id",
                    "match_num"
                ]
            ]
        )

        # Find the actual rows
        problematic_rows = model.merge(
            inconsistent_ids,
            on=[
                "tourney_id",
                "match_num"
            ],
            how="inner"
        )

        print(
            problematic_rows[
                [
                    "match_date",
                    "tourney_id",
                    "match_num",
                    "player1_id",
                    "player2_id",
                    "target"
                ]
            ]
            .sort_values(
                [
                    "tourney_id",
                    "match_num"
                ]
            )
            .to_string(index=False)
        )


# ============================================================
# FINAL COMPARISON
# ============================================================

def compare_counts(clean, model):

    print("\n" + "=" * 70)
    print("5. SOURCE VS MODEL COUNT")
    print("=" * 70)

    print(
        f"Clean source matches : {len(clean):,}"
    )

    print(
        f"Model rows           : {len(model):,}"
    )

    difference = len(clean) - len(model)

    print(
        f"Difference            : {difference:,}"
    )

    if difference == 0:

        print(
            "PASS: One model row exists for every "
            "source match."
        )

    else:

        print(
            "WARNING: Source and model row counts differ."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ATP FEATURE DEBUGGER")
    print("=" * 70)

    clean, model = load_data()

    check_source_duplicates(clean)

    check_player_pairs(clean)

    check_model_duplicates(model)

    check_target_consistency(model)

    compare_counts(clean, model)

    print("\n" + "=" * 70)
    print("DEBUGGING COMPLETED")
    print("=" * 70)