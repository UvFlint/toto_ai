from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Match(BaseModel):
    match_number: int
    home_team: str
    away_team: str
    league: str = ""
    country: str = ""
    match_date: datetime | None = None
    result: Literal["1", "X", "2"] | None = None


class WinnerForm(BaseModel):
    form_number: str | None = None
    deadline: datetime | None = None
    matches: list[Match] = []
