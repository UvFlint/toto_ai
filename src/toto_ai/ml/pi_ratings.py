"""Pi-ratings: team strength ratings for football match prediction.

Each team maintains 4 ratings:
  - ha: home attack strength
  - hd: home defense strength
  - aa: away attack strength
  - ad: away defense strength

Ratings update after each match based on the error between expected and actual
goal differences. Reference: Constantinou & Fenton (2013), "Determining the
number of goals in association football".
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import pandas as pd

# --- Constants ---

B = 10.0  # base for expected goal difference
C = 3.0  # scaling constant
DEFAULT_LR = 0.035  # learning rate (lambda)
DEFAULT_RATING = 0.0  # initial rating for unseen teams

# Feature column names produced by compute_pi_ratings
PI_FEATURE_COLS = [
    "pi_home_ha",
    "pi_home_hd",
    "pi_away_aa",
    "pi_away_ad",
    "pi_home_attack_diff",
    "pi_away_attack_diff",
    "pi_expected_home_gd",
    "pi_expected_away_gd",
]

# Type alias for the ratings store: team_name -> {ha, hd, aa, ad}
RatingStore = dict[str, dict[str, float]]


def _default_ratings() -> dict[str, float]:
    return {"ha": DEFAULT_RATING, "hd": DEFAULT_RATING, "aa": DEFAULT_RATING, "ad": DEFAULT_RATING}


def _expected_goal_diff(rating_diff: float) -> float:
    """Compute expected goal difference from rating difference."""
    if rating_diff == 0:
        return 0.0
    sign = 1.0 if rating_diff > 0 else -1.0
    return sign * (B ** (abs(rating_diff) / C) - 1.0)


def _psi(error: float) -> float:
    """Update magnitude function."""
    if error == 0:
        return 0.0
    sign = 1.0 if error > 0 else -1.0
    return sign * C * math.log10(1.0 + abs(error))


def _update_ratings(
    store: RatingStore,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    lr: float,
) -> None:
    """Update pi-ratings for both teams after a match (in-place)."""
    h = store.setdefault(home_team, _default_ratings())
    a = store.setdefault(away_team, _default_ratings())

    # Rating differences for this matchup
    r_home = h["ha"] - a["ad"]
    r_away = a["aa"] - h["hd"]

    # Expected goal differences
    exp_home_gd = _expected_goal_diff(r_home)
    exp_away_gd = _expected_goal_diff(r_away)

    # Actual goal differences
    actual_home_gd = home_goals - away_goals
    actual_away_gd = away_goals - home_goals

    # Errors
    e_home = actual_home_gd - exp_home_gd
    e_away = actual_away_gd - exp_away_gd

    # Update magnitudes
    psi_home = _psi(e_home)
    psi_away = _psi(e_away)

    # Apply updates
    h["ha"] += lr * psi_home
    a["ad"] -= lr * psi_home
    a["aa"] += lr * psi_away
    h["hd"] -= lr * psi_away


def compute_pi_ratings(
    df: pd.DataFrame,
    lr: float = DEFAULT_LR,
) -> tuple[pd.DataFrame, RatingStore]:
    """Compute pi-ratings from historical match data.

    Args:
        df: DataFrame with columns: home_team, away_team, home_goals, away_goals, date.
            Must contain valid date values for chronological ordering.
        lr: Learning rate for rating updates.

    Returns:
        Tuple of (DataFrame with 8 pi-rating feature columns added, final RatingStore).
    """
    df = df.copy()

    # Sort chronologically (essential for pi-ratings)
    df = df.sort_values("date", na_position="first").reset_index(drop=True)

    store: RatingStore = {}

    # Pre-allocate feature arrays
    n = len(df)
    features = {col: [None] * n for col in PI_FEATURE_COLS}

    for i in range(n):
        row = df.iloc[i]
        home = row["home_team"]
        away = row["away_team"]
        hg = row["home_goals"]
        ag = row["away_goals"]

        # Skip rows with missing data
        if pd.isna(home) or pd.isna(away) or pd.isna(hg) or pd.isna(ag) or pd.isna(row["date"]):
            continue

        hg = int(hg)
        ag = int(ag)

        # Get current ratings BEFORE this match
        h = store.get(home, _default_ratings())
        a = store.get(away, _default_ratings())

        # Record raw ratings
        features["pi_home_ha"][i] = h["ha"]
        features["pi_home_hd"][i] = h["hd"]
        features["pi_away_aa"][i] = a["aa"]
        features["pi_away_ad"][i] = a["ad"]

        # Derived features
        home_attack_diff = h["ha"] - a["ad"]
        away_attack_diff = a["aa"] - h["hd"]
        features["pi_home_attack_diff"][i] = home_attack_diff
        features["pi_away_attack_diff"][i] = away_attack_diff
        features["pi_expected_home_gd"][i] = _expected_goal_diff(home_attack_diff)
        features["pi_expected_away_gd"][i] = _expected_goal_diff(away_attack_diff)

        # Update ratings AFTER recording features
        _update_ratings(store, home, away, hg, ag, lr)

    # Add feature columns to DataFrame
    for col in PI_FEATURE_COLS:
        df[col] = features[col]

    return df, store


def compute_pi_features(store: RatingStore, home_team: str, away_team: str) -> dict[str, float]:
    """Extract pi-rating features for a single match at inference time.

    Unknown teams default to 0.0 (neutral rating).
    """
    h = store.get(home_team, _default_ratings())
    a = store.get(away_team, _default_ratings())

    home_attack_diff = h["ha"] - a["ad"]
    away_attack_diff = a["aa"] - h["hd"]

    return {
        "pi_home_ha": h["ha"],
        "pi_home_hd": h["hd"],
        "pi_away_aa": a["aa"],
        "pi_away_ad": a["ad"],
        "pi_home_attack_diff": home_attack_diff,
        "pi_away_attack_diff": away_attack_diff,
        "pi_expected_home_gd": _expected_goal_diff(home_attack_diff),
        "pi_expected_away_gd": _expected_goal_diff(away_attack_diff),
    }


def save_ratings(store: RatingStore, path: str | Path) -> None:
    """Save pi-rating store to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(store, f)


def load_ratings(path: str | Path) -> RatingStore:
    """Load pi-rating store from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
