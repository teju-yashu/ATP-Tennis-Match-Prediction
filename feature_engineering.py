from pathlib import Path
from collections import defaultdict, deque
import math
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "atp_matches_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = OUTPUT_DIR / "model_dataset.csv"

INITIAL_ELO = 1500.0
ELO_K = 32.0

RECENT_MATCHES = 20
RECENT_H2H = 10
RECENCY_DECAY = 0.85


# ============================================================
# PLAYER STATE
# ============================================================

def create_player_state():

    return {
        # Overall results
        "matches": 0,
        "wins": 0,

        # Recent results
        "recent_results": deque(maxlen=RECENT_MATCHES),

        # Surface-specific results
        "surface_results": defaultdict(
            lambda: deque(maxlen=RECENT_MATCHES)
        ),

        # Historical match dates/results
        "history": deque(maxlen=100),

        # Serve statistics
        "serve_points": 0,
        "aces": 0,
        "double_faults": 0,
        "first_in": 0,
        "first_won": 0,
        "second_won": 0,
        "bp_saved": 0,
        "bp_faced": 0,
    }


# ============================================================
# SAFE RATE
# ============================================================

def safe_rate(numerator, denominator, default=0.5):

    if denominator is None:
        return default

    if pd.isna(denominator) or denominator <= 0:
        return default

    if numerator is None or pd.isna(numerator):
        return default

    return numerator / denominator


# ============================================================
# RECENCY-WEIGHTED FORM
# ============================================================

def recency_weighted_form(results):

    if not results:
        return 0.0

    weighted_sum = 0.0
    weight_sum = 0.0

    # results are oldest -> newest
    # reverse gives newest first
    for i, result in enumerate(reversed(results)):

        weight = RECENCY_DECAY ** i

        weighted_sum += result * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    return weighted_sum / weight_sum


# ============================================================
# PLAYER FEATURES
# ============================================================

def get_player_features(
    player_id,
    surface,
    current_date,
    player_states,
    elo_ratings,
    surface_elos
):

    state = player_states[player_id]

    # -------------------------
    # Overall form
    # -------------------------

    recent = list(state["recent_results"])

    last_5 = recent[-5:]
    last_10 = recent[-10:]
    last_20 = recent[-20:]

    win_rate_last_5 = (
        sum(last_5) / len(last_5)
        if last_5 else 0.5
    )

    win_rate_last_10 = (
        sum(last_10) / len(last_10)
        if last_10 else 0.5
    )

    win_rate_last_20 = (
        sum(last_20) / len(last_20)
        if last_20 else 0.5
    )

    weighted_form = recency_weighted_form(
        recent
    )

    # -------------------------
    # Surface form
    # -------------------------

    surface_results = list(
        state["surface_results"][surface]
    )

    surface_win_rate = (
        sum(surface_results) / len(surface_results)
        if surface_results else 0.5
    )

    surface_last_10 = surface_results[-10:]

    surface_recent_win_rate = (
        sum(surface_last_10) / len(surface_last_10)
        if surface_last_10 else 0.5
    )

    # -------------------------
    # Last 30 days form
    # -------------------------

    thirty_days_ago = (
        current_date - pd.Timedelta(days=30)
    )

    recent_30_day_results = [
        result
        for match_date, result in state["history"]
        if match_date >= thirty_days_ago
    ]

    win_rate_last_30_days = (
        sum(recent_30_day_results)
        / len(recent_30_day_results)
        if recent_30_day_results
        else 0.5
    )

    # -------------------------
    # Overall win rate
    # -------------------------

    overall_win_rate = (
        state["wins"] / state["matches"]
        if state["matches"] > 0
        else 0.5
    )

    # -------------------------
    # Historical serve stats
    # -------------------------

    serve_points = state["serve_points"]

    ace_rate = safe_rate(
        state["aces"],
        serve_points
    )

    double_fault_rate = safe_rate(
        state["double_faults"],
        serve_points
    )

    first_serve_percentage = safe_rate(
        state["first_in"],
        serve_points
    )

    first_serve_win_percentage = safe_rate(
        state["first_won"],
        state["first_in"]
    )

    second_serve_points = (
        serve_points - state["first_in"]
    )

    second_serve_win_percentage = safe_rate(
        state["second_won"],
        second_serve_points
    )

    break_point_save_percentage = safe_rate(
        state["bp_saved"],
        state["bp_faced"]
    )

    # -------------------------
    # ELO
    # -------------------------

    overall_elo = elo_ratings[player_id]

    surface_elo = surface_elos[
        player_id
    ][surface]

    return {
        "matches": state["matches"],
        "overall_win_rate": overall_win_rate,

        "win_rate_last_5": win_rate_last_5,
        "win_rate_last_10": win_rate_last_10,
        "win_rate_last_20": win_rate_last_20,
        "recency_weighted_form": weighted_form,

        "surface_win_rate": surface_win_rate,
        "surface_recent_win_rate": (
            surface_recent_win_rate
        ),

        "win_rate_last_30_days": (
            win_rate_last_30_days
        ),

        "ace_rate": ace_rate,
        "double_fault_rate": (
            double_fault_rate
        ),
        "first_serve_percentage": (
            first_serve_percentage
        ),
        "first_serve_win_percentage": (
            first_serve_win_percentage
        ),
        "second_serve_win_percentage": (
            second_serve_win_percentage
        ),
        "break_point_save_percentage": (
            break_point_save_percentage
        ),

        "elo": overall_elo,
        "surface_elo": surface_elo,
    }


# ============================================================
# HEAD-TO-HEAD FEATURES
# ============================================================

def get_h2h_features(
    player1,
    player2,
    surface,
    h2h
):

    key = tuple(
        sorted([player1, player2])
    )

    history = h2h[key]

    if not history:
        return {
            "h2h_matches": 0,
            "h2h_win_rate": 0.5,
            "recent_h2h_win_rate": 0.5,
            "surface_h2h_matches": 0,
            "surface_h2h_win_rate": 0.5,
        }

    # Overall H2H
    p1_wins = sum(
        winner == player1
        for winner, match_surface in history
    )

    h2h_matches = len(history)

    h2h_win_rate = (
        p1_wins / h2h_matches
    )

    # Recent H2H
    recent_history = list(history)[-RECENT_H2H:]

    recent_p1_wins = sum(
        winner == player1
        for winner, match_surface in recent_history
    )

    recent_h2h_win_rate = (
        recent_p1_wins / len(recent_history)
    )

    # Surface H2H
    surface_history = [
        winner
        for winner, match_surface in history
        if match_surface == surface
    ]

    surface_p1_wins = sum(
        winner == player1
        for winner in surface_history
    )

    surface_matches = len(
        surface_history
    )

    surface_h2h_win_rate = (
        surface_p1_wins / surface_matches
        if surface_matches > 0
        else 0.5
    )

    return {
        "h2h_matches": h2h_matches,
        "h2h_win_rate": h2h_win_rate,
        "recent_h2h_win_rate": (
            recent_h2h_win_rate
        ),
        "surface_h2h_matches": surface_matches,
        "surface_h2h_win_rate": (
            surface_h2h_win_rate
        ),
    }


# ============================================================
# TOURNAMENT FORM
# ============================================================

def get_tournament_form(
    player1,
    player2,
    tournament,
    tournament_history
):

    p1_matches, p1_wins = (
        tournament_history.get(
            (player1, tournament),
            (0, 0)
        )
    )

    p2_matches, p2_wins = (
        tournament_history.get(
            (player2, tournament),
            (0, 0)
        )
    )

    p1_rate = (
        p1_wins / p1_matches
        if p1_matches > 0
        else 0.5
    )

    p2_rate = (
        p2_wins / p2_matches
        if p2_matches > 0
        else 0.5
    )

    return (
        p1_rate,
        p2_rate
    )


# ============================================================
# ELO EXPECTED SCORE
# ============================================================

def expected_elo_score(
    player_elo,
    opponent_elo
):

    return 1 / (
        1 + 10 ** (
            (opponent_elo - player_elo) / 400
        )
    )


# ============================================================
# UPDATE ELO
# ============================================================

def update_elo(
    winner,
    loser,
    surface,
    elo_ratings,
    surface_elos
):

    winner_elo = elo_ratings[winner]
    loser_elo = elo_ratings[loser]

    expected_winner = expected_elo_score(
        winner_elo,
        loser_elo
    )

    expected_loser = 1 - expected_winner

    elo_ratings[winner] = (
        winner_elo
        + ELO_K * (1 - expected_winner)
    )

    elo_ratings[loser] = (
        loser_elo
        + ELO_K * (0 - expected_loser)
    )

    # Surface ELO
    winner_surface_elo = (
        surface_elos[winner][surface]
    )

    loser_surface_elo = (
        surface_elos[loser][surface]
    )

    expected_surface_winner = (
        expected_elo_score(
            winner_surface_elo,
            loser_surface_elo
        )
    )

    expected_surface_loser = (
        1 - expected_surface_winner
    )

    surface_elos[winner][surface] = (
        winner_surface_elo
        + ELO_K
        * (1 - expected_surface_winner)
    )

    surface_elos[loser][surface] = (
        loser_surface_elo
        + ELO_K
        * (0 - expected_surface_loser)
    )


# ============================================================
# UPDATE PLAYER STATISTICS
# ============================================================

def update_player_state(
    player_id,
    won,
    surface,
    match_date,
    serve_data,
    player_states
):

    state = player_states[player_id]

    # Overall
    state["matches"] += 1

    if won:
        state["wins"] += 1

    # Recent form
    state["recent_results"].append(
        1 if won else 0
    )

    # Surface form
    state["surface_results"][surface].append(
        1 if won else 0
    )

    # Historical date/result
    state["history"].append(
        (
            match_date,
            1 if won else 0
        )
    )

    # -------------------------
    # Serve statistics
    # -------------------------

    if serve_data is None:
        return

    svpt = serve_data.get("svpt")

    if pd.isna(svpt) or svpt <= 0:
        return

    state["serve_points"] += svpt

    for key in [
        "ace",
        "double_fault",
        "first_in",
        "first_won",
        "second_won",
        "bp_saved",
        "bp_faced"
    ]:

        value = serve_data.get(key)

        if value is not None and not pd.isna(value):

            if key == "ace":
                state["aces"] += value

            elif key == "double_fault":
                state["double_faults"] += value

            elif key == "first_in":
                state["first_in"] += value

            elif key == "first_won":
                state["first_won"] += value

            elif key == "second_won":
                state["second_won"] += value

            elif key == "bp_saved":
                state["bp_saved"] += value

            elif key == "bp_faced":
                state["bp_faced"] += value


# ============================================================
# MAIN FEATURE ENGINE
# ============================================================

def build_features(df):

    print("\nBuilding chronological features...")

    # Ensure chronological ordering
    df = df.sort_values(
        [
            "tourney_date",
            "tourney_id",
            "match_num"
        ]
    ).reset_index(drop=True)

    # -------------------------
    # State containers
    # -------------------------

    player_states = defaultdict(
        create_player_state
    )

    elo_ratings = defaultdict(
        lambda: INITIAL_ELO
    )

    surface_elos = defaultdict(
        lambda: defaultdict(
            lambda: INITIAL_ELO
        )
    )

    h2h = defaultdict(
        lambda: deque(maxlen=100)
    )

    tournament_history = {}

    feature_rows = []

    total_rows = len(df)

    # ========================================================
    # PROCESS ONE DATE AT A TIME
    # ========================================================
    #
    # IMPORTANT:
    # Every match on the same date sees the player state
    # from BEFORE that date.
    #
    # This prevents another match occurring later that same
    # day from leaking information into the prediction.
    # ========================================================

    for date_index, (match_date, day_matches) in enumerate(
        df.groupby("tourney_date", sort=True)
    ):

        # Store updates until ALL matches for the date
        # have generated their features.
        daily_updates = []

        for _, row in day_matches.iterrows():

            winner = int(row["winner_id"])
            loser = int(row["loser_id"])

            surface = row["surface"]
            tournament = row["tourney_name"]

            # ------------------------------------------------
            # Deterministic player ordering
            # ------------------------------------------------
            #
            # We CANNOT simply put winner first.
            # Otherwise target would always be 1.
            #
            # We order by player ID instead.
            # ------------------------------------------------

            if winner < loser:

                player1 = winner
                player2 = loser
                target = 1

                player1_age = row["winner_age"]
                player2_age = row["loser_age"]

                player1_height = row["winner_ht"]
                player2_height = row["loser_ht"]

                player1_hand = row["winner_hand"]
                player2_hand = row["loser_hand"]

                player1_rank = row["winner_rank"]
                player2_rank = row["loser_rank"]

                player1_rank_points = (
                    row["winner_rank_points"]
                )
                player2_rank_points = (
                    row["loser_rank_points"]
                )

            else:

                player1 = loser
                player2 = winner
                target = 0

                player1_age = row["loser_age"]
                player2_age = row["winner_age"]

                player1_height = row["loser_ht"]
                player2_height = row["winner_ht"]

                player1_hand = row["loser_hand"]
                player2_hand = row["winner_hand"]

                player1_rank = row["loser_rank"]
                player2_rank = row["winner_rank"]

                player1_rank_points = (
                    row["loser_rank_points"]
                )
                player2_rank_points = (
                    row["winner_rank_points"]
                )

            # ------------------------------------------------
            # Historical player features
            # ------------------------------------------------

            p1 = get_player_features(
                player1,
                surface,
                match_date,
                player_states,
                elo_ratings,
                surface_elos
            )

            p2 = get_player_features(
                player2,
                surface,
                match_date,
                player_states,
                elo_ratings,
                surface_elos
            )

            # ------------------------------------------------
            # H2H
            # ------------------------------------------------

            h2h_features = get_h2h_features(
                player1,
                player2,
                surface,
                h2h
            )

            # ------------------------------------------------
            # Tournament form
            # ------------------------------------------------

            p1_tourney_rate, p2_tourney_rate = (
                get_tournament_form(
                    player1,
                    player2,
                    tournament,
                    tournament_history
                )
            )

            # ------------------------------------------------
            # Handedness
            # ------------------------------------------------

            if (
                player1_hand == "L"
                and player2_hand == "R"
            ):
                hand_difference = 1

            elif (
                player1_hand == "R"
                and player2_hand == "L"
            ):
                hand_difference = -1

            else:
                hand_difference = 0

            # ------------------------------------------------
            # Feature row
            # ------------------------------------------------

            features = {
                # Identifiers
                "match_date": match_date,
                "tourney_id": row["tourney_id"],
                "match_num": row["match_num"],
                "tourney_name": tournament,
                "surface": surface,

                "player1_id": player1,
                "player2_id": player2,

                # Target
                "target": target,

                # -------------------------
                # Ranking
                # -------------------------

                "rank_difference": (
                    player1_rank - player2_rank
                    if (
                        not pd.isna(player1_rank)
                        and not pd.isna(player2_rank)
                    )
                    else math.nan
                ),

                "rank_points_difference": (
                    player1_rank_points
                    - player2_rank_points
                    if (
                        not pd.isna(player1_rank_points)
                        and not pd.isna(player2_rank_points)
                    )
                    else math.nan
                ),

                # -------------------------
                # ELO
                # -------------------------

                "elo_difference": (
                    p1["elo"] - p2["elo"]
                ),

                "surface_elo_difference": (
                    p1["surface_elo"]
                    - p2["surface_elo"]
                ),

                # -------------------------
                # Overall form
                # -------------------------

                "overall_win_rate_difference": (
                    p1["overall_win_rate"]
                    - p2["overall_win_rate"]
                ),

                "win_rate_last_5_difference": (
                    p1["win_rate_last_5"]
                    - p2["win_rate_last_5"]
                ),

                "win_rate_last_10_difference": (
                    p1["win_rate_last_10"]
                    - p2["win_rate_last_10"]
                ),

                "win_rate_last_20_difference": (
                    p1["win_rate_last_20"]
                    - p2["win_rate_last_20"]
                ),

                "recency_weighted_form_difference": (
                    p1["recency_weighted_form"]
                    - p2["recency_weighted_form"]
                ),

                "win_rate_last_30_days_difference": (
                    p1["win_rate_last_30_days"]
                    - p2["win_rate_last_30_days"]
                ),

                # -------------------------
                # Surface form
                # -------------------------

                "surface_win_rate_difference": (
                    p1["surface_win_rate"]
                    - p2["surface_win_rate"]
                ),

                "surface_recent_win_rate_difference": (
                    p1["surface_recent_win_rate"]
                    - p2["surface_recent_win_rate"]
                ),

                # -------------------------
                # H2H
                # -------------------------

                "h2h_matches": (
                    h2h_features["h2h_matches"]
                ),

                "h2h_win_rate_difference": (
                    h2h_features["h2h_win_rate"]
                    - 0.5
                ),

                "recent_h2h_win_rate_difference": (
                    h2h_features[
                        "recent_h2h_win_rate"
                    ]
                    - 0.5
                ),

                "surface_h2h_matches": (
                    h2h_features[
                        "surface_h2h_matches"
                    ]
                ),

                "surface_h2h_win_rate_difference": (
                    h2h_features[
                        "surface_h2h_win_rate"
                    ]
                    - 0.5
                ),

                # -------------------------
                # Tournament form
                # -------------------------

                "tournament_win_rate_difference": (
                    p1_tourney_rate
                    - p2_tourney_rate
                ),

                # -------------------------
                # Player attributes
                # -------------------------

                "age_difference": (
                    player1_age - player2_age
                    if (
                        not pd.isna(player1_age)
                        and not pd.isna(player2_age)
                    )
                    else math.nan
                ),

                "height_difference": (
                    player1_height
                    - player2_height
                    if (
                        not pd.isna(player1_height)
                        and not pd.isna(player2_height)
                    )
                    else math.nan
                ),

                "hand_difference": hand_difference,

                # -------------------------
                # Historical serve stats
                # -------------------------

                "ace_rate_difference": (
                    p1["ace_rate"]
                    - p2["ace_rate"]
                ),

                "double_fault_rate_difference": (
                    p1["double_fault_rate"]
                    - p2["double_fault_rate"]
                ),

                "first_serve_percentage_difference": (
                    p1["first_serve_percentage"]
                    - p2["first_serve_percentage"]
                ),

                "first_serve_win_percentage_difference": (
                    p1[
                        "first_serve_win_percentage"
                    ]
                    - p2[
                        "first_serve_win_percentage"
                    ]
                ),

                "second_serve_win_percentage_difference": (
                    p1[
                        "second_serve_win_percentage"
                    ]
                    - p2[
                        "second_serve_win_percentage"
                    ]
                ),

                "break_point_save_percentage_difference": (
                    p1[
                        "break_point_save_percentage"
                    ]
                    - p2[
                        "break_point_save_percentage"
                    ]
                ),
            }

            feature_rows.append(features)

            # ------------------------------------------------
            # Save update information.
            #
            # IMPORTANT:
            # We update player states only AFTER all matches
            # for this date have generated features.
            # ------------------------------------------------

            winner_serve_data = {
                "svpt": row["w_svpt"],
                "ace": row["w_ace"],
                "double_fault": row["w_df"],
                "first_in": row["w_1stIn"],
                "first_won": row["w_1stWon"],
                "second_won": row["w_2ndWon"],
                "bp_saved": row["w_bpSaved"],
                "bp_faced": row["w_bpFaced"],
            }

            loser_serve_data = {
                "svpt": row["l_svpt"],
                "ace": row["l_ace"],
                "double_fault": row["l_df"],
                "first_in": row["l_1stIn"],
                "first_won": row["l_1stWon"],
                "second_won": row["l_2ndWon"],
                "bp_saved": row["l_bpSaved"],
                "bp_faced": row["l_bpFaced"],
            }

            daily_updates.append(
                (
                    winner,
                    loser,
                    surface,
                    tournament,
                    winner_serve_data,
                    loser_serve_data
                )
            )

        # ====================================================
        # UPDATE ALL STATES AFTER THE DATE
        # ====================================================

        for (
            winner,
            loser,
            surface,
            tournament,
            winner_serve_data,
            loser_serve_data
        ) in daily_updates:

            # Player history
            update_player_state(
                winner,
                True,
                surface,
                match_date,
                winner_serve_data,
                player_states
            )

            update_player_state(
                loser,
                False,
                surface,
                match_date,
                loser_serve_data,
                player_states
            )

            # ELO
            update_elo(
                winner,
                loser,
                surface,
                elo_ratings,
                surface_elos
            )

            # H2H
            key = tuple(
                sorted([winner, loser])
            )

            h2h[key].append(
                (winner, surface)
            )

            # Tournament history
            winner_key = (
                winner,
                tournament
            )

            loser_key = (
                loser,
                tournament
            )

            old_matches, old_wins = (
                tournament_history.get(
                    winner_key,
                    (0, 0)
                )
            )

            tournament_history[winner_key] = (
                old_matches + 1,
                old_wins + 1
            )

            old_matches, old_wins = (
                tournament_history.get(
                    loser_key,
                    (0, 0)
                )
            )

            tournament_history[loser_key] = (
                old_matches + 1,
                old_wins
            )

        # Progress
        if (
            date_index + 1
        ) % 250 == 0:

            print(
                f"Processed {date_index + 1:,} dates..."
            )

    return pd.DataFrame(feature_rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TENNIS FEATURE ENGINEERING")
    print("=" * 70)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Clean dataset not found:\n{INPUT_PATH}"
        )

    print("\nLoading clean dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["tourney_date"]
    )

    print(
        f"Loaded {len(df):,} matches."
    )

    # Build features
    model_df = build_features(df)

    # Ensure chronological order
    model_df = model_df.sort_values(
        "match_date"
    ).reset_index(drop=True)

    # Create output directory
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    # Save
    model_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print(
        f"Rows generated   : {len(model_df):,}"
    )

    print(
        f"Features generated: "
        f"{len(model_df.columns):,}"
    )

    print(
        f"Date range       : "
        f"{model_df['match_date'].min().date()} → "
        f"{model_df['match_date'].max().date()}"
    )

    print(
        f"Target mean      : "
        f"{model_df['target'].mean():.4f}"
    )

    print("\nFeature columns:")

    for column in model_df.columns:
        print(f"  - {column}")

    print("\nMissing values:")

    missing = model_df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if missing.empty:
        print("  None")

    else:
        print(missing.to_string())

    print("\nSaved to:")

    print(OUTPUT_PATH)

    print("\nDONE.")