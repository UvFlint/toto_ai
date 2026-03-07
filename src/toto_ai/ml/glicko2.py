"""Glicko-2 team ratings for football match prediction.

Each team maintains 3 values:
  - mu: rating (default 1500)
  - rd: rating deviation — confidence/uncertainty
  - vol: volatility — how erratic performance is

Ratings update after each match based on the outcome (W=1.0, D=0.5, L=0.0).
The key signals beyond pi-ratings are RD (uncertainty) and volatility (consistency).

Reference: Glickman, M.E. (2012), "Example of the Glicko-2 system".
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import pandas as pd

# --- Constants ---

MU_DEFAULT = 1500.0
RD_DEFAULT = 350.0
VOL_DEFAULT = 0.06
TAU = 0.5  # system constant (constrains volatility change)
GLICKO2_SCALE = 173.7178  # scaling factor between Glicko and Glicko-2 scales
EPSILON = 1e-6  # convergence tolerance for volatility solver
MAX_ITERATIONS = 100  # safety cap for Illinois algorithm

# Feature column names produced by compute_glicko2_ratings
GLICKO2_FEATURE_COLS = [
    "glicko_home_rating",
    "glicko_away_rating",
    "glicko_home_rd",
    "glicko_away_rd",
    "glicko_home_vol",
    "glicko_away_vol",
    "glicko_rating_diff",
    "glicko_home_expected",
]

# Type alias for the ratings store: team_name -> {mu, rd, vol}
GlickoStore = dict[str, dict[str, float]]


def _default_glicko() -> dict[str, float]:
    return {"mu": MU_DEFAULT, "rd": RD_DEFAULT, "vol": VOL_DEFAULT}


def _to_glicko2(mu: float, rd: float) -> tuple[float, float]:
    """Convert from Glicko scale to Glicko-2 internal scale."""
    return (mu - MU_DEFAULT) / GLICKO2_SCALE, rd / GLICKO2_SCALE


def _from_glicko2(mu2: float, phi: float) -> tuple[float, float]:
    """Convert from Glicko-2 internal scale back to Glicko scale."""
    return mu2 * GLICKO2_SCALE + MU_DEFAULT, phi * GLICKO2_SCALE


def _g(phi: float) -> float:
    """Reduction factor for opponent's RD."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu2: float, mu2_j: float, phi_j: float) -> float:
    """Expected score against opponent."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu2 - mu2_j)))


def _update_volatility(sigma: float, phi: float, v: float, delta: float) -> float:
    """Compute new volatility using the Illinois algorithm (Glickman Step 5)."""
    a = math.log(sigma * sigma)
    phi_sq = phi * phi
    delta_sq = delta * delta
    tau_sq = TAU * TAU

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta_sq - phi_sq - v - ex)
        denom = 2.0 * (phi_sq + v + ex) ** 2
        return num / denom - (x - a) / tau_sq

    # Set initial bounds
    big_a = a
    if delta_sq > phi_sq + v:
        big_b = math.log(delta_sq - phi_sq - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
            if k > MAX_ITERATIONS:
                return sigma  # safety: no convergence
        big_b = a - k * TAU

    f_a = f(big_a)
    f_b = f(big_b)

    for _ in range(MAX_ITERATIONS):
        if abs(big_b - big_a) < EPSILON:
            break
        big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(big_c)

        if f_c * f_b <= 0:
            big_a = big_b
            f_a = f_b
        else:
            f_a /= 2.0

        big_b = big_c
        f_b = f_c

    return math.exp(big_a / 2.0)


def _update_glicko2(
    store: GlickoStore,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    """Update Glicko-2 ratings for both teams after a match (in-place)."""
    h = store.setdefault(home_team, _default_glicko())
    a = store.setdefault(away_team, _default_glicko())

    # Match outcome
    if home_goals > away_goals:
        s_home, s_away = 1.0, 0.0
    elif home_goals < away_goals:
        s_home, s_away = 0.0, 1.0
    else:
        s_home, s_away = 0.5, 0.5

    # Convert to Glicko-2 scale
    h_mu2, h_phi = _to_glicko2(h["mu"], h["rd"])
    a_mu2, a_phi = _to_glicko2(a["mu"], a["rd"])

    # Update home team
    g_a = _g(a_phi)
    e_h = _E(h_mu2, a_mu2, a_phi)
    v_h = 1.0 / (g_a * g_a * e_h * (1.0 - e_h))
    delta_h = v_h * g_a * (s_home - e_h)

    new_sigma_h = _update_volatility(h["vol"], h_phi, v_h, delta_h)
    phi_star_h = math.sqrt(h_phi * h_phi + new_sigma_h * new_sigma_h)
    new_phi_h = 1.0 / math.sqrt(1.0 / (phi_star_h * phi_star_h) + 1.0 / v_h)
    new_mu_h = h_mu2 + new_phi_h * new_phi_h * g_a * (s_home - e_h)

    mu_h, rd_h = _from_glicko2(new_mu_h, new_phi_h)
    h["mu"] = mu_h
    h["rd"] = rd_h
    h["vol"] = new_sigma_h

    # Update away team
    g_h = _g(h_phi)  # use pre-update phi for away team's perspective
    e_a = _E(a_mu2, h_mu2, h_phi)
    v_a = 1.0 / (g_h * g_h * e_a * (1.0 - e_a))
    delta_a = v_a * g_h * (s_away - e_a)

    new_sigma_a = _update_volatility(a["vol"], a_phi, v_a, delta_a)
    phi_star_a = math.sqrt(a_phi * a_phi + new_sigma_a * new_sigma_a)
    new_phi_a = 1.0 / math.sqrt(1.0 / (phi_star_a * phi_star_a) + 1.0 / v_a)
    new_mu_a = a_mu2 + new_phi_a * new_phi_a * g_h * (s_away - e_a)

    mu_a, rd_a = _from_glicko2(new_mu_a, new_phi_a)
    a["mu"] = mu_a
    a["rd"] = rd_a
    a["vol"] = new_sigma_a


def compute_glicko2_ratings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, GlickoStore]:
    """Compute Glicko-2 ratings from historical match data.

    Args:
        df: DataFrame with columns: home_team, away_team, home_goals, away_goals, date.

    Returns:
        Tuple of (DataFrame with 8 Glicko-2 feature columns added, final GlickoStore).
    """
    df = df.copy()
    df = df.sort_values("date", na_position="first").reset_index(drop=True)

    store: GlickoStore = {}
    n = len(df)
    features = {col: [None] * n for col in GLICKO2_FEATURE_COLS}

    for i in range(n):
        row = df.iloc[i]
        home = row["home_team"]
        away = row["away_team"]
        hg = row["home_goals"]
        ag = row["away_goals"]

        if pd.isna(home) or pd.isna(away) or pd.isna(hg) or pd.isna(ag) or pd.isna(row["date"]):
            continue

        hg = int(hg)
        ag = int(ag)

        # Record current ratings BEFORE this match
        h = store.get(home, _default_glicko())
        a = store.get(away, _default_glicko())

        features["glicko_home_rating"][i] = h["mu"]
        features["glicko_away_rating"][i] = a["mu"]
        features["glicko_home_rd"][i] = h["rd"]
        features["glicko_away_rd"][i] = a["rd"]
        features["glicko_home_vol"][i] = h["vol"]
        features["glicko_away_vol"][i] = a["vol"]
        features["glicko_rating_diff"][i] = h["mu"] - a["mu"]

        # Expected score on Glicko-2 scale
        h_mu2, _ = _to_glicko2(h["mu"], h["rd"])
        a_mu2, a_phi = _to_glicko2(a["mu"], a["rd"])
        features["glicko_home_expected"][i] = _E(h_mu2, a_mu2, a_phi)

        # Update ratings AFTER recording features
        _update_glicko2(store, home, away, hg, ag)

    for col in GLICKO2_FEATURE_COLS:
        df[col] = features[col]

    return df, store


def compute_glicko2_features(
    store: GlickoStore, home_team: str, away_team: str
) -> dict[str, float]:
    """Extract Glicko-2 features for a single match at inference time.

    Unknown teams default to initial values (1500/350/0.06).
    """
    h = store.get(home_team, _default_glicko())
    a = store.get(away_team, _default_glicko())

    h_mu2, _ = _to_glicko2(h["mu"], h["rd"])
    a_mu2, a_phi = _to_glicko2(a["mu"], a["rd"])

    return {
        "glicko_home_rating": h["mu"],
        "glicko_away_rating": a["mu"],
        "glicko_home_rd": h["rd"],
        "glicko_away_rd": a["rd"],
        "glicko_home_vol": h["vol"],
        "glicko_away_vol": a["vol"],
        "glicko_rating_diff": h["mu"] - a["mu"],
        "glicko_home_expected": _E(h_mu2, a_mu2, a_phi),
    }


def save_glicko_store(store: GlickoStore, path: str | Path) -> None:
    """Save Glicko-2 store to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(store, f)


def load_glicko_store(path: str | Path) -> GlickoStore:
    """Load Glicko-2 store from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301
