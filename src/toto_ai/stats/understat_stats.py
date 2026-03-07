from __future__ import annotations

import difflib
import re
from datetime import datetime

import httpx

from toto_ai.console import console
from toto_ai.stats.models import MatchStats, TeamXG

# Hebrew league name -> understat league identifier
HEBREW_LEAGUE_MAP: dict[str, str] = {
    "פרמייר ליג": "EPL",
    "פרמיירליג": "EPL",
    "Premier League": "EPL",
    "לה ליגה": "La_liga",
    "La Liga": "La_liga",
    "בונדסליגה": "Bundesliga",
    "Bundesliga": "Bundesliga",
    "סרייה א": "Serie_A",
    "סריה א": "Serie_A",
    "Serie A": "Serie_A",
    "ליג 1": "Ligue_1",
    "Ligue 1": "Ligue_1",
    "ספרדית ראשונה": "La_liga",
    "איטלקית ראשונה": "Serie_A",
    "גרמנית ראשונה": "Bundesliga",
    "צרפתית ראשונה": "Ligue_1",
}

_UNDERSTAT_BASE = "https://understat.com/getLeagueData"

# Known name mismatches between API-Football and understat
_NAME_ALIASES: dict[str, str] = {
    "wolverhampton wanderers": "wolverhampton",
    "west ham united": "west ham",
    "tottenham hotspur": "tottenham",
    "atletico madrid": "atletico madrid",
    "paris saint germain": "paris saint-germain",
    "borussia monchengladbach": "borussia m.gladbach",
    "internazionale": "inter",
    "ac milan": "milan",
}

_STRIP_SUFFIXES = re.compile(r"\b(fc|cf|sc|afc|ssc|as|us|ac|rc|og|1\.\s*fc)\b", re.IGNORECASE)
_STRIP_PUNCT = re.compile(r"[^\w\s-]")


def _normalize_name(name: str) -> str:
    """Normalize a team name for matching."""
    n = name.lower().strip()
    n = _STRIP_SUFFIXES.sub("", n)
    n = _STRIP_PUNCT.sub("", n)
    return " ".join(n.split())


def _current_season() -> int:
    """Return the current football season year (e.g. 2025 for 2025/26 season)."""
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def _aggregate_team(title: str, history: list[dict]) -> TeamXG:
    """Aggregate per-match understat history into a season-level TeamXG."""
    n = len(history)
    if n == 0:
        return TeamXG(team_name=title)

    # Sort by date descending to get most recent matches first
    sorted_hist = sorted(history, key=lambda m: m.get("date", ""), reverse=True)

    # Extract last 5 match-level xG/xGA
    recent_xg = [round(m["xG"], 2) for m in sorted_hist[:5]]
    recent_xga = [round(m["xGA"], 2) for m in sorted_hist[:5]]

    xg = sum(m["xG"] for m in history)
    xga = sum(m["xGA"] for m in history)
    npxg = sum(m["npxG"] for m in history)
    npxga = sum(m["npxGA"] for m in history)
    npxgd = sum(m["npxGD"] for m in history)
    scored = sum(m["scored"] for m in history)
    missed = sum(m["missed"] for m in history)
    pts = sum(m["pts"] for m in history)
    xpts = sum(m["xpts"] for m in history)
    deep = sum(m["deep"] for m in history)
    deep_allowed = sum(m["deep_allowed"] for m in history)

    # PPDA: total passes / total defensive actions across all matches
    ppda_att = sum(m["ppda"]["att"] for m in history)
    ppda_def = sum(m["ppda"]["def"] for m in history)
    oppda_att = sum(m["ppda_allowed"]["att"] for m in history)
    oppda_def = sum(m["ppda_allowed"]["def"] for m in history)
    ppda = round(ppda_att / ppda_def, 2) if ppda_def else 0.0
    oppda = round(oppda_att / oppda_def, 2) if oppda_def else 0.0

    return TeamXG(
        team_name=title,
        matches_played=n,
        xg=round(xg, 2),
        xga=round(xga, 2),
        npxg=round(npxg, 2),
        npxga=round(npxga, 2),
        npxgd=round(npxgd, 2),
        actual_goals=scored,
        actual_goals_against=missed,
        xg_diff=round(scored - xg, 2),
        xga_diff=round(missed - xga, 2),
        xpts=round(xpts, 2),
        actual_pts=pts,
        xpts_diff=round(pts - xpts, 2),
        ppda=ppda,
        oppda=oppda,
        dc=deep,
        odc=deep_allowed,
        xg_per_game=round(xg / n, 2),
        xga_per_game=round(xga / n, 2),
        recent_match_xg=recent_xg,
        recent_match_xga=recent_xga,
    )


class UnderstatCollector:
    """Collect xG data from understat.com for supported European leagues."""

    def __init__(self) -> None:
        self._league_cache: dict[str, dict[str, TeamXG]] = {}

    async def enrich_match_stats(
        self,
        stats: list[MatchStats],
        leagues: list[str],
    ) -> None:
        """Enrich MatchStats with xG data where the league is supported.

        Args:
            stats: list of MatchStats (one per match, same order as leagues)
            leagues: Hebrew or English league names from Match.league
        """
        needed: dict[str, str] = {}
        match_league_map: list[str | None] = []
        for league in leagues:
            ul = HEBREW_LEAGUE_MAP.get(league)
            match_league_map.append(ul)
            if ul and ul not in needed:
                needed[ul] = league

        if not needed:
            console.print("[dim]Understat: no supported leagues found in form[/dim]")
            return

        season = _current_season()

        async with httpx.AsyncClient(timeout=30) as client:
            for ul in needed:
                if ul in self._league_cache:
                    continue
                try:
                    data = await self._fetch_league_data(client, ul, season)
                    self._league_cache[ul] = data
                    console.print(
                        f"[dim]Understat: fetched {len(data)} teams for {ul} {season}[/dim]"
                    )
                except Exception as e:
                    console.print(f"[yellow]Understat: failed to fetch {ul}: {e}[/yellow]")
                    self._league_cache[ul] = {}

        matched = 0
        for i, s in enumerate(stats):
            ul = match_league_map[i]
            if not ul or ul not in self._league_cache:
                continue

            teams_data = self._league_cache[ul]
            if not teams_data:
                continue

            home_name = s.home_team_english or s.home_team
            away_name = s.away_team_english or s.away_team

            home_xg = self._match_team(home_name, teams_data)
            away_xg = self._match_team(away_name, teams_data)

            if home_xg:
                s.home_xg = home_xg
            if away_xg:
                s.away_xg = away_xg
            if home_xg or away_xg:
                matched += 1

        console.print(f"[green]xG data matched for {matched}/{len(stats)} matches[/green]")

    async def _fetch_league_data(
        self,
        client: httpx.AsyncClient,
        league: str,
        season: int,
    ) -> dict[str, TeamXG]:
        """Fetch league data from understat AJAX endpoint and aggregate into TeamXG."""
        url = f"{_UNDERSTAT_BASE}/{league}/{season}"
        resp = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        teams_raw = data.get("teams", {})
        result: dict[str, TeamXG] = {}
        for _team_id, team_data in teams_raw.items():
            title = team_data["title"]
            history = team_data.get("history", [])
            team_xg = _aggregate_team(title, history)
            normalized = _normalize_name(title)
            result[normalized] = team_xg

        return result

    def _match_team(self, english_name: str, understat_teams: dict[str, TeamXG]) -> TeamXG | None:
        """Match an API-Football team name to an understat team."""
        normalized = _normalize_name(english_name)

        # Check alias map first
        if normalized in _NAME_ALIASES:
            alias_norm = _normalize_name(_NAME_ALIASES[normalized])
            if alias_norm in understat_teams:
                return understat_teams[alias_norm]

        # Exact match
        if normalized in understat_teams:
            return understat_teams[normalized]

        # Fuzzy match
        matches = difflib.get_close_matches(normalized, understat_teams.keys(), n=1, cutoff=0.6)
        if matches:
            return understat_teams[matches[0]]

        console.print(f"[dim]Understat: no match for '{english_name}'[/dim]")
        return None
