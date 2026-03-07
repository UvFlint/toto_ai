"""Feature engineering for CatBoost match result prediction.

Loads historical CSVs from football-data.co.uk, computes rolling averages
per team, odds-implied probabilities, and team strength features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from toto_ai.data_collector.football_data_downloader import ALL_DIVISIONS
from toto_ai.ml.glicko2 import GLICKO2_FEATURE_COLS, GlickoStore, compute_glicko2_ratings
from toto_ai.ml.pi_ratings import PI_FEATURE_COLS, RatingStore, compute_pi_ratings

# --- Column mappings ---

# Main league CSV columns (football-data.co.uk mmz4281 pattern)
MAIN_COLS = {
    "Date": "date",
    "Div": "league",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "home_odds",
    "B365D": "draw_odds",
    "B365A": "away_odds",
}

# Extra league CSV columns (football-data.co.uk /new/ pattern)
EXTRA_COLS = {
    "Date": "date",
    "Home": "home_team",
    "Away": "away_team",
    "HG": "home_goals",
    "AG": "away_goals",
    "Res": "result",
    "AvgCH": "home_odds",
    "AvgCD": "draw_odds",
    "AvgCA": "away_odds",
}

# Rolling window size for team form features
ROLLING_WINDOW = 5

# Inference-only features: NaN during training, populated from MatchStats at prediction time.
# Both CatBoost and XGBoost handle missing values natively.
INFERENCE_ONLY_FEATURES = [
    "home_rest_days",
    "away_rest_days",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_total_matches",
    "home_injury_count",
    "away_injury_count",
    "home_xg_per_game",
    "away_xg_per_game",
    "home_xga_per_game",
    "away_xga_per_game",
    "xg_diff",
    "home_npxgd",
    "away_npxgd",
    "home_position",
    "away_position",
    "position_diff",
]

# Target encoding: FTR/Res values → numeric classes
RESULT_MAP = {"H": 0, "D": 1, "A": 2}


def _parse_season_from_path(path: Path) -> str:
    """Extract season code from file path (e.g. '2425' from 'E0/2425.csv')."""
    return path.stem


def load_main_league_data(data_dir: str | Path) -> pd.DataFrame:
    """Load all main league CSVs into a single DataFrame.

    Reads every {division}/{season}.csv file, renames columns to a standard
    schema, and adds a ``season`` column from the filename.
    """
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []

    for div_code in ALL_DIVISIONS:
        div_path = data_dir / div_code
        if not div_path.is_dir():
            continue
        for csv_path in sorted(div_path.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
            except Exception:
                continue
            if df.empty or "FTR" not in df.columns:
                continue

            # Keep only columns we care about (some older CSVs lack certain cols)
            available = {k: v for k, v in MAIN_COLS.items() if k in df.columns}
            df = df[list(available.keys())].rename(columns=available)
            df["season"] = _parse_season_from_path(csv_path)

            # Drop rows where result is missing
            df = df.dropna(subset=["result"])
            df = df[df["result"].isin(["H", "D", "A"])]
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Ensure numeric columns
    for col in ["home_goals", "away_goals", "home_odds", "draw_odds", "away_odds"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    # Parse dates for pi-rating chronological ordering
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, errors="coerce")
    return combined


def load_extra_league_data(data_dir: str | Path) -> pd.DataFrame:
    """Load all extra league CSVs into a single DataFrame."""
    data_dir = Path(data_dir) / "extra"
    if not data_dir.is_dir():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
        except Exception:
            continue
        if df.empty or "Res" not in df.columns:
            continue

        available = {k: v for k, v in EXTRA_COLS.items() if k in df.columns}
        df = df[list(available.keys())].rename(columns=available)

        # Use Country column as league if available
        if "Country" in pd.read_csv(csv_path, nrows=0).columns:
            raw = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
            df["league"] = raw["Country"].str.strip()
        else:
            df["league"] = csv_path.stem

        # Season column
        if "Season" in pd.read_csv(csv_path, nrows=0).columns:
            raw = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
            df["season"] = raw["Season"].astype(str).str.strip()
        else:
            df["season"] = "all"

        df = df.dropna(subset=["result"])
        df = df[df["result"].isin(["H", "D", "A"])]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    for col in ["home_goals", "away_goals", "home_odds", "draw_odds", "away_odds"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    # Parse dates for pi-rating chronological ordering
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, errors="coerce")
    return combined


def _add_odds_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Add odds-implied probabilities normalized to sum to 1."""
    for col in ["home_odds", "draw_odds", "away_odds"]:
        if col not in df.columns:
            return df

    inv_h = 1.0 / df["home_odds"]
    inv_d = 1.0 / df["draw_odds"]
    inv_a = 1.0 / df["away_odds"]
    total = inv_h + inv_d + inv_a

    df["implied_home"] = inv_h / total
    df["implied_draw"] = inv_d / total
    df["implied_away"] = inv_a / total
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    cols_to_roll: list[str],
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Add rolling average features per team (home and away separately).

    For each match, computes rolling averages from the team's PREVIOUS matches
    (shifted by 1 to avoid data leakage).
    """
    # Data is already in chronological order from CSV loading (sorted by season file + row order)
    df = df.reset_index(drop=True)

    # Points columns for rolling
    df["home_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["away_pts"] = df["result"].map({"H": 0, "D": 1, "A": 3})

    # Build per-team rolling stats
    for prefix, team_col, goals_for, goals_against, pts_col in [
        ("h", "home_team", "home_goals", "away_goals", "home_pts"),
        ("a", "away_team", "away_goals", "home_goals", "away_pts"),
    ]:
        # Group by team and compute shifted rolling means
        for stat_col, feat_name in [
            (goals_for, f"{prefix}_roll_gf"),
            (goals_against, f"{prefix}_roll_ga"),
            (pts_col, f"{prefix}_roll_pts"),
        ]:
            if stat_col not in df.columns:
                continue
            df[feat_name] = df.groupby(team_col)[stat_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Rolling for match stats (shots, corners etc.) — only if available
        for stat_col in cols_to_roll:
            if stat_col not in df.columns:
                continue
            feat_name = f"{prefix}_roll_{stat_col}"
            df[feat_name] = df.groupby(team_col)[stat_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    # Clean up temp columns
    df.drop(columns=["home_pts", "away_pts"], inplace=True, errors="ignore")
    return df


def _add_season_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Add season-level team strength features (expanding mean of points and goals)."""
    df = df.reset_index(drop=True)

    df["_h_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["_a_pts"] = df["result"].map({"H": 0, "D": 1, "A": 3})

    for prefix, team_col, gf, ga, pts in [
        ("h", "home_team", "home_goals", "away_goals", "_h_pts"),
        ("a", "away_team", "away_goals", "home_goals", "_a_pts"),
    ]:
        for stat, name in [(gf, f"{prefix}_season_gpg"), (pts, f"{prefix}_season_ppg")]:
            if stat not in df.columns:
                continue
            df[name] = df.groupby([team_col, "league", "season"])[stat].transform(
                lambda x: x.shift(1).expanding().mean()
            )

    df.drop(columns=["_h_pts", "_a_pts"], inplace=True, errors="ignore")
    return df


def engineer_rich_features(df: pd.DataFrame) -> tuple[pd.DataFrame, RatingStore, GlickoStore]:
    """Full feature engineering for main league data (rich stats + odds)."""
    df = df.copy()

    # Pi-ratings (must run before any reordering)
    df, pi_store = compute_pi_ratings(df)
    df, glicko_store = compute_glicko2_ratings(df)

    # Odds-implied probabilities
    df = _add_odds_implied_probs(df)

    # Half-time goal ratio (what fraction of goals scored in first half)
    if "ht_home_goals" in df.columns and "home_goals" in df.columns:
        df["ht_goal_ratio_home"] = pd.to_numeric(
            df["ht_home_goals"], errors="coerce"
        ) / pd.to_numeric(df["home_goals"], errors="coerce").replace(0, float("nan"))
    if "ht_away_goals" in df.columns and "away_goals" in df.columns:
        df["ht_goal_ratio_away"] = pd.to_numeric(
            df["ht_away_goals"], errors="coerce"
        ) / pd.to_numeric(df["away_goals"], errors="coerce").replace(0, float("nan"))

    # Rolling features for match stats
    rich_stats_cols = [
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
        "home_fouls",
        "away_fouls",
        "home_yellow",
        "away_yellow",
        "home_red",
        "away_red",
        "ht_goal_ratio_home",
        "ht_goal_ratio_away",
    ]
    df = _add_rolling_features(df, cols_to_roll=rich_stats_cols)

    # Season strength
    df = _add_season_strength(df)

    # Inference-only feature placeholders (NaN during training)
    for col in INFERENCE_ONLY_FEATURES:
        df[col] = float("nan")

    # Encode target
    df["target"] = df["result"].map(RESULT_MAP)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df, pi_store, glicko_store


def engineer_simple_features(df: pd.DataFrame) -> tuple[pd.DataFrame, RatingStore, GlickoStore]:
    """Minimal feature engineering for all data (goals + odds only)."""
    df = df.copy()

    # Pi-ratings and Glicko-2 (must run before any reordering)
    df, pi_store = compute_pi_ratings(df)
    df, glicko_store = compute_glicko2_ratings(df)

    df = _add_odds_implied_probs(df)
    df = _add_rolling_features(df, cols_to_roll=[])
    df = _add_season_strength(df)

    df["target"] = df["result"].map(RESULT_MAP)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df, pi_store, glicko_store


# --- Feature lists for model training ---

RICH_FEATURES = [
    # Odds-implied
    "implied_home",
    "implied_draw",
    "implied_away",
    # Raw odds
    "home_odds",
    "draw_odds",
    "away_odds",
    # Rolling goals
    "h_roll_gf",
    "h_roll_ga",
    "h_roll_pts",
    "a_roll_gf",
    "a_roll_ga",
    "a_roll_pts",
    # Rolling match stats
    "h_roll_home_shots",
    "h_roll_away_shots",
    "h_roll_home_shots_on_target",
    "h_roll_away_shots_on_target",
    "h_roll_home_corners",
    "h_roll_away_corners",
    "h_roll_home_fouls",
    "h_roll_away_fouls",
    "h_roll_home_yellow",
    "h_roll_away_yellow",
    "a_roll_home_shots",
    "a_roll_away_shots",
    "a_roll_home_shots_on_target",
    "a_roll_away_shots_on_target",
    "a_roll_home_corners",
    "a_roll_away_corners",
    "a_roll_home_fouls",
    "a_roll_away_fouls",
    "a_roll_home_yellow",
    "a_roll_away_yellow",
    # Season strength
    "h_season_gpg",
    "h_season_ppg",
    "a_season_gpg",
    "a_season_ppg",
    # Categorical
    "league",
    "season",
    # Rolling red cards
    "h_roll_home_red",
    "h_roll_away_red",
    "a_roll_home_red",
    "a_roll_away_red",
    # Rolling half-time goal ratio
    "h_roll_ht_goal_ratio_home",
    "a_roll_ht_goal_ratio_away",
    # Pi-ratings
    *PI_FEATURE_COLS,
    # Glicko-2
    *GLICKO2_FEATURE_COLS,
    # Inference-only (NaN during training, populated at prediction time)
    *INFERENCE_ONLY_FEATURES,
]

SIMPLE_FEATURES = [
    "implied_home",
    "implied_draw",
    "implied_away",
    "home_odds",
    "draw_odds",
    "away_odds",
    "h_roll_gf",
    "h_roll_ga",
    "h_roll_pts",
    "a_roll_gf",
    "a_roll_ga",
    "a_roll_pts",
    "h_season_gpg",
    "h_season_ppg",
    "a_season_gpg",
    "a_season_ppg",
    "league",
    "season",
    # Pi-ratings
    *PI_FEATURE_COLS,
]

CAT_FEATURES = ["league", "season"]


def build_inference_features(
    match_league: str,
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
    home_form_gf: float | None,
    home_form_ga: float | None,
    home_form_pts: float | None,
    away_form_gf: float | None,
    away_form_ga: float | None,
    away_form_pts: float | None,
    home_season_gpg: float | None,
    home_season_ppg: float | None,
    away_season_gpg: float | None,
    away_season_ppg: float | None,
    home_shots: float | None = None,
    away_shots: float | None = None,
    home_sot: float | None = None,
    away_sot: float | None = None,
    home_corners: float | None = None,
    away_corners: float | None = None,
    home_fouls: float | None = None,
    away_fouls: float | None = None,
    home_yellow: float | None = None,
    away_yellow: float | None = None,
    # Pi-rating features
    pi_home_ha: float | None = None,
    pi_home_hd: float | None = None,
    pi_away_aa: float | None = None,
    pi_away_ad: float | None = None,
    pi_home_attack_diff: float | None = None,
    pi_away_attack_diff: float | None = None,
    pi_expected_home_gd: float | None = None,
    pi_expected_away_gd: float | None = None,
    # Glicko-2 features
    glicko_home_rating: float | None = None,
    glicko_away_rating: float | None = None,
    glicko_home_rd: float | None = None,
    glicko_away_rd: float | None = None,
    glicko_home_vol: float | None = None,
    glicko_away_vol: float | None = None,
    glicko_rating_diff: float | None = None,
    glicko_home_expected: float | None = None,
    # Red cards & half-time ratio (Group A)
    home_red: float | None = None,
    away_red: float | None = None,
    home_ht_goal_ratio: float | None = None,
    away_ht_goal_ratio: float | None = None,
    # Inference-only features (Group B)
    home_rest_days: float | None = None,
    away_rest_days: float | None = None,
    h2h_home_win_rate: float | None = None,
    h2h_draw_rate: float | None = None,
    h2h_total_matches: float | None = None,
    home_injury_count: float | None = None,
    away_injury_count: float | None = None,
    home_xg_per_game: float | None = None,
    away_xg_per_game: float | None = None,
    home_xga_per_game: float | None = None,
    away_xga_per_game: float | None = None,
    xg_diff: float | None = None,
    home_npxgd: float | None = None,
    away_npxgd: float | None = None,
    home_position: float | None = None,
    away_position: float | None = None,
    position_diff: float | None = None,
) -> dict:
    """Build a feature dict for a single match at inference time.

    Uses live MatchStats data converted to the same feature schema as training.
    CatBoost handles None/NaN values natively.
    """
    # Odds-implied probabilities
    implied_home = implied_draw = implied_away = None
    if home_odds and draw_odds and away_odds:
        inv_h = 1.0 / home_odds
        inv_d = 1.0 / draw_odds
        inv_a = 1.0 / away_odds
        total = inv_h + inv_d + inv_a
        implied_home = inv_h / total
        implied_draw = inv_d / total
        implied_away = inv_a / total

    return {
        "implied_home": implied_home,
        "implied_draw": implied_draw,
        "implied_away": implied_away,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "h_roll_gf": home_form_gf,
        "h_roll_ga": home_form_ga,
        "h_roll_pts": home_form_pts,
        "a_roll_gf": away_form_gf,
        "a_roll_ga": away_form_ga,
        "a_roll_pts": away_form_pts,
        "h_season_gpg": home_season_gpg,
        "h_season_ppg": home_season_ppg,
        "a_season_gpg": away_season_gpg,
        "a_season_ppg": away_season_ppg,
        # Rich features (may be None for non-main leagues)
        "h_roll_home_shots": home_shots,
        "h_roll_away_shots": away_shots,
        "h_roll_home_shots_on_target": home_sot,
        "h_roll_away_shots_on_target": away_sot,
        "h_roll_home_corners": home_corners,
        "h_roll_away_corners": away_corners,
        "h_roll_home_fouls": home_fouls,
        "h_roll_away_fouls": away_fouls,
        "h_roll_home_yellow": home_yellow,
        "h_roll_away_yellow": away_yellow,
        "a_roll_home_shots": None,
        "a_roll_away_shots": None,
        "a_roll_home_shots_on_target": None,
        "a_roll_away_shots_on_target": None,
        "a_roll_home_corners": None,
        "a_roll_away_corners": None,
        "a_roll_home_fouls": None,
        "a_roll_away_fouls": None,
        "a_roll_home_yellow": None,
        "a_roll_away_yellow": None,
        "league": match_league,
        "season": "current",
        # Pi-ratings
        "pi_home_ha": pi_home_ha,
        "pi_home_hd": pi_home_hd,
        "pi_away_aa": pi_away_aa,
        "pi_away_ad": pi_away_ad,
        "pi_home_attack_diff": pi_home_attack_diff,
        "pi_away_attack_diff": pi_away_attack_diff,
        "pi_expected_home_gd": pi_expected_home_gd,
        "pi_expected_away_gd": pi_expected_away_gd,
        # Glicko-2
        "glicko_home_rating": glicko_home_rating,
        "glicko_away_rating": glicko_away_rating,
        "glicko_home_rd": glicko_home_rd,
        "glicko_away_rd": glicko_away_rd,
        "glicko_home_vol": glicko_home_vol,
        "glicko_away_vol": glicko_away_vol,
        "glicko_rating_diff": glicko_rating_diff,
        "glicko_home_expected": glicko_home_expected,
        # Red cards & half-time ratio
        "h_roll_home_red": home_red,
        "h_roll_away_red": away_red,
        "a_roll_home_red": None,
        "a_roll_away_red": None,
        "h_roll_ht_goal_ratio_home": home_ht_goal_ratio,
        "a_roll_ht_goal_ratio_away": away_ht_goal_ratio,
        # Inference-only
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "h2h_home_win_rate": h2h_home_win_rate,
        "h2h_draw_rate": h2h_draw_rate,
        "h2h_total_matches": h2h_total_matches,
        "home_injury_count": home_injury_count,
        "away_injury_count": away_injury_count,
        "home_xg_per_game": home_xg_per_game,
        "away_xg_per_game": away_xg_per_game,
        "home_xga_per_game": home_xga_per_game,
        "away_xga_per_game": away_xga_per_game,
        "xg_diff": xg_diff,
        "home_npxgd": home_npxgd,
        "away_npxgd": away_npxgd,
        "home_position": home_position,
        "away_position": away_position,
        "position_diff": position_diff,
    }
