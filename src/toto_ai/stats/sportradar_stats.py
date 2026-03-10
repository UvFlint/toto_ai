"""Sportradar stats collector — replaces API-Football for H2H, form, standings, and odds.

Data is fetched from stats.fn.sportradar.com using IDs parsed from the sportradar_url
already captured on each Match object by winner_scraper._enrich_sportradar_urls().

URL format:
  https://s5.sir.sportradar.com/israelsportsbettingboard/en/1/season/{seasonId}/
    headtohead/{homeId}/{awayId}/match/{matchId}

API base: https://stats.fn.sportradar.com/israelsportsbettingboard/en/Asia:Jerusalem/gismo/
  - match_get/{matchId}               → teams, odds, probabilities, referee
  - stats_team_versus/{homeId}/{awayId} → H2H matches
  - stats_team_streaks/{teamId}        → recent form (W/D/L)
  - stats_season_tables/{tableId}      → standings (tableId from match_get.tournament.livetable)
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import NamedTuple
import httpx

from toto_ai.console import console
from toto_ai.scraper.models import Match
from toto_ai.stats.api_football_stats import ApiFootballStatsCollector
from toto_ai.stats.models import (
    FixtureResult,
    H2HData,
    InjuryInfo,
    MatchStats,
    OddsData,
    Standing,
    TeamForm,
)

_SR_API_BASE = "https://stats.fn.sportradar.com/israelsportsbettingboard/en/Asia:Jerusalem/gismo"
_SR_HEADERS = {
    "Referer": "https://s5.sir.sportradar.com/",
    "Origin": "https://s5.sir.sportradar.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Regex to parse IDs from the Sportradar widget URL
_URL_PATTERN = re.compile(
    r"/season/(?P<season_id>\d+)/headtohead/(?P<home_id>\d+)/(?P<away_id>\d+)/match/(?P<match_id>\d+)"
)


class _SrIds(NamedTuple):
    match_id: int
    home_id: int
    away_id: int
    season_id: int


def _parse_sr_url(url: str) -> _SrIds | None:
    """Extract Sportradar IDs from widget URL."""
    m = _URL_PATTERN.search(url)
    if not m:
        return None
    return _SrIds(
        match_id=int(m.group("match_id")),
        home_id=int(m.group("home_id")),
        away_id=int(m.group("away_id")),
        season_id=int(m.group("season_id")),
    )


def _uts_to_datetime(uts: int | None) -> datetime | None:
    """Convert Unix timestamp to UTC datetime."""
    if not uts:
        return None
    try:
        return datetime.fromtimestamp(uts, tz=timezone.utc)
    except (OSError, ValueError):
        return None


async def _sr_get(client: httpx.AsyncClient, endpoint: str) -> dict | None:
    """GET a Sportradar gismo endpoint, return parsed doc[0].data or None on error."""
    url = f"{_SR_API_BASE}/{endpoint}"
    try:
        resp = await client.get(url, headers=_SR_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("doc", [])
        if not docs:
            return None
        doc = docs[0]
        if doc.get("event") == "exception":
            return None
        return doc.get("data")
    except Exception as e:
        console.print(f"[dim]Sportradar {endpoint}: {e}[/dim]")
        return None


def _parse_h2h(data: dict, home_name: str, away_name: str) -> H2HData:
    """Parse stats_team_versus response into H2HData."""
    matches_raw = data.get("matches", [])
    results: list[FixtureResult] = []
    home_wins = draws = away_wins = 0

    for m in matches_raw:
        result = m.get("result", {})
        home_goals = result.get("home")
        away_goals = result.get("away")
        if home_goals is None or away_goals is None:
            continue
        home_goals = int(home_goals)
        away_goals = int(away_goals)

        teams = m.get("teams", {})
        h_name = teams.get("home", {}).get("name", "")
        a_name = teams.get("away", {}).get("name", "")
        league = m.get("tournament", {}).get("name", "") if isinstance(m.get("tournament"), dict) else ""

        dt = _uts_to_datetime(m.get("time", {}).get("uts"))

        results.append(
            FixtureResult(
                home_team=h_name,
                away_team=a_name,
                home_goals=home_goals,
                away_goals=away_goals,
                date=dt,
                league=league,
            )
        )

        # Determine winner from the perspective of original home/away
        winner = result.get("winner")
        if winner == "home":
            home_wins += 1
        elif winner == "away":
            away_wins += 1
        else:
            draws += 1

    return H2HData(
        home_team=home_name,
        away_team=away_name,
        matches=results,
        home_wins=home_wins,
        draws=draws,
        away_wins=away_wins,
    )


def _parse_form(data: dict, team_name: str) -> TeamForm:
    """Parse stats_team_streaks response into TeamForm."""
    form_entries = data.get("lastmatchesform", {}).get("total", [])

    wins = draws = losses = 0
    form_chars: list[str] = []
    recent: list[FixtureResult] = []

    for entry in form_entries:
        typeid = entry.get("typeid", "")
        if typeid == "W":
            wins += 1
            form_chars.append("W")
        elif typeid == "D":
            draws += 1
            form_chars.append("D")
        elif typeid == "L":
            losses += 1
            form_chars.append("L")

    form_string = "".join(form_chars[:5])

    return TeamForm(
        team_name=team_name,
        recent_matches=recent,
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=0,
        goals_against=0,
        form_string=form_string,
    )


def _parse_standings(data: dict, team_id: int, team_name: str) -> Standing | None:
    """Parse stats_season_tables response, find the row for team_id."""
    tables = data.get("tables", [])
    if not tables:
        return None

    for table in tables:
        rows = table.get("tablerows", [])
        for row in rows:
            t = row.get("team", {})
            row_uid = t.get("uid") or t.get("_id")
            if row_uid == team_id:
                return Standing(
                    position=row.get("pos", 0),
                    team_name=t.get("name", team_name),
                    team_id=team_id,
                    played=row.get("pld", 0),
                    wins=row.get("win", 0),
                    draws=row.get("draw", 0),
                    losses=row.get("loss", 0),
                    goals_for=row.get("gf", 0),
                    goals_against=row.get("ga", 0),
                    goal_diff=row.get("gd", 0),
                    points=row.get("pts", 0),
                )
    return None


def _parse_odds(data: dict) -> OddsData | None:
    """Parse odds from match_get response."""
    odds_section = data.get("odds", {})
    # Try bookmakers in priority order
    for key in ("iseodds", "pinnacleodds", "bet365odds"):
        entries = odds_section.get(key, [])
        if entries:
            entry = entries[0]
            try:
                return OddsData(
                    bookmaker=entry.get("bookmaker", {}).get("name", key),
                    home_odds=float(entry["home"]["odds"]),
                    draw_odds=float(entry["draw"]["odds"]),
                    away_odds=float(entry["away"]["odds"]),
                )
            except (KeyError, ValueError, TypeError):
                continue
    return None


async def _fetch_standings_and_injuries_from_api_football(
    home_team: str,
    away_team: str,
    home_team_english: str,
    away_team_english: str,
    league: str,
    match_date: datetime | None,
    af_collector: ApiFootballStatsCollector,
    af_client: httpx.AsyncClient,
) -> tuple[Standing | None, Standing | None, list[InjuryInfo], list[InjuryInfo]]:
    """Fetch standings + injuries from API-Football for a single match.

    Uses the TeamMapper to resolve team IDs, then fetches only standings and
    injuries — skipping H2H/form/odds which come from Sportradar.
    """
    from toto_ai.stats.api_football_stats import _current_season, _find_standing, _parse_injuries

    api_headers = af_collector._client._headers()
    base_url = af_collector._client._base_url

    # Resolve team IDs via TeamMapper (Hebrew names + league hint)
    home_info, away_info = await asyncio.gather(
        af_collector._mapper.resolve(home_team, af_client, api_headers=api_headers, base_url=base_url),
        af_collector._mapper.resolve(away_team, af_client, api_headers=api_headers, base_url=base_url),
    )

    if not home_info or not away_info:
        return None, None, [], []

    home_id = home_info.v3_id
    away_id = away_info.v3_id
    league_id = home_info.league_id or away_info.league_id
    season = _current_season()

    # Fetch fixture ID (for injuries) and standings in parallel
    tasks: list = [af_collector._find_fixture_id(home_id, away_id, match_date, af_client)]
    standings_idx = None
    if league_id:
        standings_idx = len(tasks)
        tasks.append(af_collector._fetch_standings(league_id, season, af_client))

    results = await asyncio.gather(*tasks)
    fixture_id, _referee = results[0]
    standings_raw = results[standings_idx] if standings_idx is not None else []

    # Parse standings
    home_standing = away_standing = None
    if standings_raw:
        home_standing = _find_standing(standings_raw, home_id, home_team_english or home_team)
        away_standing = _find_standing(standings_raw, away_id, away_team_english or away_team)

    # Fetch injuries if we have a fixture ID
    home_injuries: list[InjuryInfo] = []
    away_injuries: list[InjuryInfo] = []
    if fixture_id:
        try:
            inj_data = await af_collector._client.get(
                "injuries", {"fixture": fixture_id}, client=af_client
            )
            all_inj = _parse_injuries(inj_data.get("response", []))
            h_name = (home_team_english or home_team).lower()
            for inj in all_inj:
                t = inj.team_name.lower()
                if t in h_name or h_name in t:
                    home_injuries.append(inj)
                else:
                    away_injuries.append(inj)
        except Exception as e:
            console.print(f"[dim]Injuries fetch failed: {e}[/dim]")

    return home_standing, away_standing, home_injuries, away_injuries


async def research_match_from_sportradar(
    match: Match,
    sr_client: httpx.AsyncClient,
    af_collector: ApiFootballStatsCollector | None = None,
    af_client: httpx.AsyncClient | None = None,
) -> MatchStats:
    """Fetch full match stats from Sportradar API using IDs from match.sportradar_url.

    Standings and injuries are fetched from API-Football when af_collector/af_client
    are provided.
    """
    home_team = match.home_team
    away_team = match.away_team

    if not match.sportradar_url:
        console.print(f"[dim]No Sportradar URL for {home_team} vs {away_team}[/dim]")
        return MatchStats(home_team=home_team, away_team=away_team, match_date=match.match_date)

    ids = _parse_sr_url(match.sportradar_url)
    if not ids:
        console.print(f"[dim]Could not parse Sportradar URL: {match.sportradar_url}[/dim]")
        return MatchStats(home_team=home_team, away_team=away_team, match_date=match.match_date)

    # Fetch match_get, H2H, and both team streaks in parallel from Sportradar
    match_data, h2h_data, home_streaks, away_streaks = await asyncio.gather(
        _sr_get(sr_client, f"match_get/{ids.match_id}"),
        _sr_get(sr_client, f"stats_team_versus/{ids.home_id}/{ids.away_id}"),
        _sr_get(sr_client, f"stats_team_streaks/{ids.home_id}"),
        _sr_get(sr_client, f"stats_team_streaks/{ids.away_id}"),
    )

    # Parse team names from match_get (English)
    home_english = away_english = ""
    referee = ""
    odds: OddsData | None = None

    if match_data:
        teams = match_data.get("teams", {})
        home_english = teams.get("home", {}).get("name", "") or home_team
        away_english = teams.get("away", {}).get("name", "") or away_team

        # Referee from match info
        referee = match_data.get("referee", {}).get("name", "") if isinstance(match_data.get("referee"), dict) else ""

        odds = _parse_odds(match_data)

    # Parse H2H
    h2h: H2HData | None = None
    if h2h_data:
        h2h = _parse_h2h(h2h_data, home_english or home_team, away_english or away_team)

    # Parse form
    home_form: TeamForm | None = None
    away_form: TeamForm | None = None
    if home_streaks:
        home_form = _parse_form(home_streaks, home_english or home_team)
    if away_streaks:
        away_form = _parse_form(away_streaks, away_english or away_team)

    # Fetch standings + injuries from API-Football
    home_standing = away_standing = None
    home_injuries: list[InjuryInfo] = []
    away_injuries: list[InjuryInfo] = []
    if af_collector and af_client:
        home_standing, away_standing, home_injuries, away_injuries = (
            await _fetch_standings_and_injuries_from_api_football(
                home_team=home_team,
                away_team=away_team,
                home_team_english=home_english,
                away_team_english=away_english,
                league=match.league,
                match_date=match.match_date,
                af_collector=af_collector,
                af_client=af_client,
            )
        )

    # Compute rest days from form
    from toto_ai.stats.api_football_stats import _compute_rest_days

    home_rest, home_avg = _compute_rest_days(home_form, match.match_date)
    away_rest, away_avg = _compute_rest_days(away_form, match.match_date)

    return MatchStats(
        home_team=home_team,
        away_team=away_team,
        home_team_id=ids.home_id,
        away_team_id=ids.away_id,
        home_team_english=home_english or None,
        away_team_english=away_english or None,
        h2h=h2h,
        home_form=home_form,
        away_form=away_form,
        home_standing=home_standing,
        away_standing=away_standing,
        home_injuries=home_injuries,
        away_injuries=away_injuries,
        odds=odds,
        match_date=match.match_date,
        referee=referee,
        home_rest_days=home_rest,
        away_rest_days=away_rest,
        home_avg_days_between=home_avg,
        away_avg_days_between=away_avg,
    )


class SportradarStatsCollector:
    """Collects match stats from Sportradar (free) + API-Football for standings/injuries.

    Sportradar: H2H, form, odds (free, no rate limit).
    API-Football free tier: standings (cached per league) + injuries (~2 calls/match).

    Requires each Match to have a sportradar_url field populated by the scraper.
    Matches without a URL fall back to empty MatchStats.
    """

    def __init__(self, use_api_football: bool = True, max_concurrent: int = 4):
        self._use_af = use_api_football
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def research_all_matches(self, matches: list[Match]) -> list[MatchStats]:
        """Research all matches concurrently."""
        from toto_ai.config import settings

        af_collector: ApiFootballStatsCollector | None = None
        if self._use_af and settings.API_FOOTBALL_API_KEY:
            af_collector = ApiFootballStatsCollector()

        async with httpx.AsyncClient() as sr_client:
            af_http_client: httpx.AsyncClient | None = None
            if af_collector:
                af_http_client = httpx.AsyncClient(timeout=30)

            try:
                async def _with_limit(idx: int, match: Match) -> tuple[int, MatchStats]:
                    async with self._semaphore:
                        label = f"{match.home_team} vs {match.away_team}"
                        console.print(f"[dim]Sportradar: {label}...[/dim]")
                        try:
                            stats = await research_match_from_sportradar(
                                match,
                                sr_client=sr_client,
                                af_collector=af_collector,
                                af_client=af_http_client,
                            )
                        except Exception as e:
                            console.print(f"[yellow]Sportradar research failed for {label}: {e}[/yellow]")
                            stats = MatchStats(
                                home_team=match.home_team,
                                away_team=match.away_team,
                                match_date=match.match_date,
                            )
                        return idx, stats

                tasks = [_with_limit(i, m) for i, m in enumerate(matches)]
                results = await asyncio.gather(*tasks)
            finally:
                if af_http_client:
                    await af_http_client.aclose()

        ordered = [MatchStats(home_team="", away_team="")] * len(matches)
        for idx, stats in results:
            ordered[idx] = stats

        if af_collector:
            console.print(f"[dim]API-Football: {af_collector._client.call_count} calls used[/dim]")
        sr_count = sum(1 for s in ordered if s.home_team_id is not None)
        console.print(f"[green]Sportradar: {sr_count}/{len(matches)} matches enriched[/green]")
        return ordered
