"""Poisson/Dixon-Coles statistical model for match outcome prediction.

Uses team attack/defense strength from standings, recent form, and xG data
to compute expected goals and match outcome probabilities. No external
dependencies — uses only ``math`` stdlib.
"""

from __future__ import annotations

import math

from toto_ai.analyzer.models import FullColumn, MatchPrediction
from toto_ai.console import console
from toto_ai.scraper.models import Match
from toto_ai.stats.models import (
    MatchStats,
    PoissonProbabilities,
    Standing,
    TeamForm,
    TeamXG,
)

# Default league average goals per team per game (UEFA average ~1.3)
_DEFAULT_LEAGUE_AVG = 1.30
# Home advantage multiplier (empirical: home teams score ~25% more)
_HOME_ADVANTAGE = 1.25
# Dixon-Coles correlation parameter for low-scoring outcomes
_RHO = -0.13
# Maximum goals to consider in the probability matrix
_MAX_GOALS = 7


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function: P(X = k) given rate lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _estimate_goals_per_game(
    standing: Standing | None,
    form: TeamForm | None,
    xg: TeamXG | None,
) -> tuple[float, float]:
    """Estimate (goals_scored_per_game, goals_conceded_per_game) for a team.

    Uses tiered data: xG > standings > form. Blends season + recent form.
    Returns (_DEFAULT_LEAGUE_AVG, _DEFAULT_LEAGUE_AVG) if no data available.
    """
    season_gpg: float | None = None
    season_cpg: float | None = None

    # Tier 1: xG per game (most predictive)
    if xg and xg.matches_played >= 3:
        season_gpg = xg.xg_per_game
        season_cpg = xg.xga_per_game

    # Tier 2: Standings (full season goals)
    if season_gpg is None and standing and standing.played >= 3:
        season_gpg = standing.goals_for / standing.played
        season_cpg = standing.goals_against / standing.played

    # Recent form (last 5 matches)
    recent_gpg: float | None = None
    recent_cpg: float | None = None
    if form and form.recent_matches:
        n = len(form.recent_matches)
        recent_gpg = form.goals_for / n
        recent_cpg = form.goals_against / n

    # Blend season + recent (70/30)
    if season_gpg is not None and recent_gpg is not None:
        gpg = 0.7 * season_gpg + 0.3 * recent_gpg
        cpg = 0.7 * season_cpg + 0.3 * recent_cpg  # type: ignore[operator]
    elif season_gpg is not None:
        gpg = season_gpg
        cpg = season_cpg  # type: ignore[assignment]
    elif recent_gpg is not None:
        gpg = recent_gpg
        cpg = recent_cpg  # type: ignore[assignment]
    else:
        return _DEFAULT_LEAGUE_AVG, _DEFAULT_LEAGUE_AVG

    return max(gpg, 0.1), max(cpg, 0.1)


def _estimate_league_avg(
    home_standing: Standing | None,
    away_standing: Standing | None,
) -> float:
    """Estimate league average goals per team per game from available standings."""
    totals: list[float] = []
    for s in (home_standing, away_standing):
        if s and s.played >= 3:
            totals.append((s.goals_for + s.goals_against) / (2 * s.played))
    if totals:
        return sum(totals) / len(totals)
    return _DEFAULT_LEAGUE_AVG


def _dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float = _RHO,
) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts the independent Poisson assumption for (0,0), (1,0), (0,1), (1,1).
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def compute_match_probabilities(
    stats: MatchStats,
    home_advantage: float = _HOME_ADVANTAGE,
) -> PoissonProbabilities | None:
    """Compute 1/X/2 probabilities for a single match using Poisson/Dixon-Coles.

    Returns None if insufficient data to make a meaningful estimate.
    """
    # Need at least some data for one team
    has_data = (
        stats.home_standing
        or stats.away_standing
        or stats.home_form
        or stats.away_form
        or stats.home_xg
        or stats.away_xg
    )
    if not has_data:
        return None

    league_avg = _estimate_league_avg(stats.home_standing, stats.away_standing)

    home_gpg, home_cpg = _estimate_goals_per_game(
        stats.home_standing, stats.home_form, stats.home_xg
    )
    away_gpg, away_cpg = _estimate_goals_per_game(
        stats.away_standing, stats.away_form, stats.away_xg
    )

    # Attack/defense strength relative to league average
    home_attack = home_gpg / league_avg
    home_defense = home_cpg / league_avg
    away_attack = away_gpg / league_avg
    away_defense = away_cpg / league_avg

    # Expected goals
    lambda_home = home_attack * away_defense * league_avg * home_advantage
    lambda_away = away_attack * home_defense * league_avg

    # Clamp to reasonable range
    lambda_home = max(0.2, min(lambda_home, 5.0))
    lambda_away = max(0.2, min(lambda_away, 5.0))

    # Build probability matrix
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for i in range(_MAX_GOALS):
        for j in range(_MAX_GOALS):
            p_ij = (
                poisson_pmf(i, lambda_home)
                * poisson_pmf(j, lambda_away)
                * _dixon_coles_tau(i, j, lambda_home, lambda_away)
            )
            if i > j:
                p_home_win += p_ij
            elif i == j:
                p_draw += p_ij
            else:
                p_away_win += p_ij

    # Normalize (should be very close to 1.0 already)
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw /= total
        p_away_win /= total

    return PoissonProbabilities(
        home_win=round(p_home_win, 4),
        draw=round(p_draw, 4),
        away_win=round(p_away_win, 4),
        expected_home_goals=round(lambda_home, 2),
        expected_away_goals=round(lambda_away, 2),
    )


def compute_all_probabilities(
    matches: list[Match],
    stats: list[MatchStats],
) -> list[PoissonProbabilities | None]:
    """Compute Poisson probabilities for all matches."""
    results: list[PoissonProbabilities | None] = []
    for i, _match in enumerate(matches):
        if i < len(stats):
            results.append(compute_match_probabilities(stats[i]))
        else:
            results.append(None)
    return results


def build_poisson_column(
    matches: list[Match],
    stats: list[MatchStats],
) -> FullColumn:
    """Build a FullColumn from Poisson model predictions for consensus voting."""
    predictions: list[MatchPrediction] = []

    for match in matches:
        s = next((st for st in stats if st.home_team == match.home_team), None)
        probs = s.poisson_probs if s else None

        if not probs:
            continue

        # Pick the most likely outcome
        outcomes = {"1": probs.home_win, "X": probs.draw, "2": probs.away_win}
        best = max(outcomes, key=outcomes.get)  # type: ignore[arg-type]
        confidence = outcomes[best]

        # Determine data sources used
        factors: list[str] = []
        if s:
            if s.home_xg or s.away_xg:
                factors.append("xG data")
            if s.home_standing or s.away_standing:
                factors.append("Season standings")
            if s.home_form or s.away_form:
                factors.append("Recent form")

        home = s.home_team_english or match.home_team if s else match.home_team
        away = s.away_team_english or match.away_team if s else match.away_team

        predictions.append(
            MatchPrediction(
                match_number=match.match_number,
                home_team=home,
                away_team=away,
                prediction=best,
                confidence=round(confidence, 3),
                reasoning=(
                    f"Expected goals: {probs.expected_home_goals:.2f} - "
                    f"{probs.expected_away_goals:.2f}. "
                    f"P(1)={probs.home_win:.1%} P(X)={probs.draw:.1%} P(2)={probs.away_win:.1%}"
                ),
                key_factors=factors or ["Poisson model"],
            )
        )

    console.print(f"[green]Poisson/DC model: {len(predictions)} predictions generated[/green]")

    return FullColumn(model_name="Poisson/DC", predictions=predictions)
