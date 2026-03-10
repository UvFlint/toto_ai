from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class NewsCategory(str, Enum):
    """Football news categories ranked by match-outcome impact."""

    STAR_PLAYER_ABSENCE = "star_player_absence"
    GOALKEEPER_CHANGE = "goalkeeper_change"
    MANAGER_SACKING = "manager_sacking"
    MASS_ABSENCE = "mass_absence"
    KEY_PLAYER_RETURN = "key_player_return"
    TACTICAL_SHIFT = "tactical_shift"
    MOTIVATION_SHIFT = "motivation_shift"
    TRANSFER_DISRUPTION = "transfer_disruption"
    MINOR_INJURY = "minor_injury"
    TRAVEL_WEATHER = "travel_weather"
    FAN_ATMOSPHERE = "fan_atmosphere"
    GENERAL_CONTEXT = "general_context"


CATEGORY_WEIGHTS: dict[NewsCategory, float] = {
    NewsCategory.STAR_PLAYER_ABSENCE: 0.9,
    NewsCategory.GOALKEEPER_CHANGE: 0.85,
    NewsCategory.MANAGER_SACKING: 0.8,
    NewsCategory.MASS_ABSENCE: 0.85,
    NewsCategory.KEY_PLAYER_RETURN: 0.5,
    NewsCategory.TACTICAL_SHIFT: 0.45,
    NewsCategory.MOTIVATION_SHIFT: 0.55,
    NewsCategory.TRANSFER_DISRUPTION: 0.5,
    NewsCategory.MINOR_INJURY: 0.2,
    NewsCategory.TRAVEL_WEATHER: 0.15,
    NewsCategory.FAN_ATMOSPHERE: 0.1,
    NewsCategory.GENERAL_CONTEXT: 0.05,
}

_TIER1_CATEGORIES = {
    NewsCategory.STAR_PLAYER_ABSENCE,
    NewsCategory.GOALKEEPER_CHANGE,
    NewsCategory.MANAGER_SACKING,
    NewsCategory.MASS_ABSENCE,
}

_DEFAULT_POST_ODDS_MULTIPLIER = 1.5
_cached_multiplier: float | None = None


def _get_post_odds_multiplier() -> float:
    """Load from calibration.json if available, otherwise default 1.5."""
    global _cached_multiplier
    if _cached_multiplier is None:
        try:
            from toto_ai.calibration import CalibrationManager

            _cached_multiplier = CalibrationManager().post_odds_multiplier
        except Exception:
            _cached_multiplier = _DEFAULT_POST_ODDS_MULTIPLIER
    return _cached_multiplier


class NewsItem(BaseModel):
    """A single categorized news item for a match."""

    headline: str = Field(description="One-sentence summary of the news")
    category: NewsCategory
    affected_team: Literal["home", "away", "both"] = Field(description="Which team is affected")
    direction: Literal["positive", "negative", "neutral"] = Field(
        description="'positive' helps the team, 'negative' hurts the team"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the news is accurate and confirmed (0.9+ official, 0.6-0.8 journalist, 0.3-0.5 rumor)",
    )
    is_post_odds: bool = Field(
        default=False,
        description="True if this news emerged AFTER the betting form was published (~7 days before deadline, when odds were set)",
    )
    player_name: str | None = Field(default=None, description="Affected player name, if applicable")


class MatchNewsAnalysis(BaseModel):
    """Structured news analysis for a single match."""

    items: list[NewsItem] = Field(default_factory=list)

    def _team_impact(self, team: str) -> float:
        score = 0.0
        for item in self.items:
            if item.affected_team not in (team, "both"):
                continue
            weight = CATEGORY_WEIGHTS[item.category]
            multiplier = _get_post_odds_multiplier() if item.is_post_odds else 1.0
            if item.direction == "positive":
                sign = 1.0
            elif item.direction == "negative":
                sign = -1.0
            else:
                continue
            score += weight * multiplier * sign * item.confidence
        return round(score, 3)

    @property
    def home_impact_score(self) -> float:
        """Net impact on home team. Positive = helps home."""
        return self._team_impact("home")

    @property
    def away_impact_score(self) -> float:
        """Net impact on away team. Positive = helps away."""
        return self._team_impact("away")

    @property
    def net_impact(self) -> float:
        """Positive favors home, negative favors away."""
        return round(self.home_impact_score - self.away_impact_score, 3)

    @property
    def max_item_weight(self) -> float:
        """Weight of the highest-impact news item."""
        if not self.items:
            return 0.0
        return max(CATEGORY_WEIGHTS[item.category] for item in self.items)

    @property
    def has_x_factor(self) -> bool:
        """True if there is at least one high-impact, post-odds news item."""
        return any(
            item.is_post_odds and CATEGORY_WEIGHTS[item.category] >= 0.7 for item in self.items
        )

    def team_absence_count(self, team: str) -> int:
        """Count of tier-1 negative items for the given team."""
        return sum(
            1
            for item in self.items
            if item.affected_team in (team, "both")
            and item.direction == "negative"
            and item.category in _TIER1_CATEGORIES
        )
