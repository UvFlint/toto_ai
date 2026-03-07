from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FixtureResult(BaseModel):
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    date: datetime | None = None
    league: str = ""


class H2HData(BaseModel):
    home_team: str
    away_team: str
    matches: list[FixtureResult] = []
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0


class TeamForm(BaseModel):
    team_name: str
    recent_matches: list[FixtureResult] = []
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    form_string: str = ""  # e.g. "WWDLW"


class Standing(BaseModel):
    position: int
    team_name: str
    team_id: int = 0
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_diff: int = 0
    points: int = 0


class InjuryInfo(BaseModel):
    player_name: str
    team_name: str
    type: str = ""  # "Injury" or "Suspension"
    reason: str = ""


class OddsData(BaseModel):
    bookmaker: str = ""
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0


class MatchStats(BaseModel):
    """All gathered statistics for a single match."""

    home_team: str
    away_team: str
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_team_english: str | None = None
    away_team_english: str | None = None
    h2h: H2HData | None = None
    home_form: TeamForm | None = None
    away_form: TeamForm | None = None
    home_standing: Standing | None = None
    away_standing: Standing | None = None
    news: str = ""
    match_date: datetime | None = None
    home_injuries: list[InjuryInfo] = []
    away_injuries: list[InjuryInfo] = []
    odds: OddsData | None = None
