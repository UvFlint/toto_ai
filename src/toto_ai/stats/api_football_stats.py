from __future__ import annotations

import asyncio
from datetime import datetime

import httpx

from toto_ai.console import console
from toto_ai.stats.api_football_client import ApiFootballClient
from toto_ai.stats.models import (
    FixtureResult,
    H2HData,
    InjuryInfo,
    MatchStats,
    OddsData,
    Standing,
    TeamForm,
    TeamFormStats,
)
from toto_ai.stats.team_mapper import TeamMapper


def _current_season() -> int:
    """Return the current football season year (e.g. 2025 for 2025/26 season)."""
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def _parse_fixture_result(fixture: dict) -> FixtureResult:
    """Parse an API-Football fixture object into a FixtureResult."""
    teams = fixture.get("teams", {})
    goals = fixture.get("goals", {})
    info = fixture.get("fixture", {})

    date = None
    date_str = info.get("date", "")
    if date_str:
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    league = fixture.get("league", {})
    return FixtureResult(
        home_team=teams.get("home", {}).get("name", ""),
        away_team=teams.get("away", {}).get("name", ""),
        home_goals=goals.get("home") or 0,
        away_goals=goals.get("away") or 0,
        date=date,
        league=league.get("name", ""),
        fixture_id=info.get("id"),
        referee=(info.get("referee") or ""),
    )


def _build_h2h(fixtures: list[dict], home_name: str, away_name: str) -> H2HData:
    """Build H2HData from API-Football fixtures/headtohead response."""
    results = [_parse_fixture_result(f) for f in fixtures]
    home_wins = 0
    draws = 0
    away_wins = 0
    for r in results:
        if r.home_goals > r.away_goals:
            # Check which side corresponds to our home/away
            if r.home_team == home_name:
                home_wins += 1
            else:
                away_wins += 1
        elif r.home_goals < r.away_goals:
            if r.away_team == home_name:
                home_wins += 1
            else:
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


def _build_team_form(fixtures: list[dict], team_name: str, team_id: int) -> TeamForm:
    """Build TeamForm from API-Football fixtures response for a team."""
    results = [_parse_fixture_result(f) for f in fixtures]
    wins = draws = losses = goals_for = goals_against = 0
    form_chars: list[str] = []

    for f in fixtures:
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        home_id = teams.get("home", {}).get("id")
        is_home = home_id == team_id

        gf = (goals.get("home") or 0) if is_home else (goals.get("away") or 0)
        ga = (goals.get("away") or 0) if is_home else (goals.get("home") or 0)
        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
            form_chars.append("W")
        elif gf < ga:
            losses += 1
            form_chars.append("L")
        else:
            draws += 1
            form_chars.append("D")

    return TeamForm(
        team_name=team_name,
        recent_matches=results,
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=goals_for,
        goals_against=goals_against,
        form_string="".join(form_chars),
    )


def _find_standing(standings_data: list, team_id: int, team_name: str) -> Standing | None:
    """Find a team's standing in the API-Football standings response."""
    for group in standings_data:
        if not isinstance(group, list):
            continue
        for entry in group:
            if entry.get("team", {}).get("id") == team_id:
                return Standing(
                    position=entry.get("rank", 0),
                    team_name=team_name,
                    team_id=team_id,
                    played=entry.get("all", {}).get("played", 0),
                    wins=entry.get("all", {}).get("win", 0),
                    draws=entry.get("all", {}).get("draw", 0),
                    losses=entry.get("all", {}).get("lose", 0),
                    goals_for=entry.get("all", {}).get("goals", {}).get("for", 0),
                    goals_against=entry.get("all", {}).get("goals", {}).get("against", 0),
                    goal_diff=entry.get("goalsDiff", 0),
                    points=entry.get("points", 0),
                )
    return None


def _parse_injuries(injuries_data: list[dict]) -> list[InjuryInfo]:
    """Parse API-Football injuries response into InjuryInfo list."""
    result = []
    for entry in injuries_data:
        player = entry.get("player", {})
        team = entry.get("team", {})
        result.append(
            InjuryInfo(
                player_name=player.get("name", ""),
                team_name=team.get("name", ""),
                type=player.get("type", ""),
                reason=player.get("reason", ""),
            )
        )
    return result


def _parse_odds(odds_data: list[dict]) -> OddsData | None:
    """Parse API-Football odds response into OddsData."""
    for bookmaker_entry in odds_data:
        bookmaker_name = bookmaker_entry.get("bookmaker", {}).get("name", "")
        for bet in bookmaker_entry.get("bets", []):
            # bet ID 1 = "Match Winner"
            if bet.get("id") == 1 or bet.get("name") == "Match Winner":
                values = {v.get("value"): v.get("odd") for v in bet.get("values", [])}
                try:
                    return OddsData(
                        bookmaker=bookmaker_name,
                        home_odds=float(values.get("Home", 0)),
                        draw_odds=float(values.get("Draw", 0)),
                        away_odds=float(values.get("Away", 0)),
                    )
                except (ValueError, TypeError):
                    continue
    return None


def _compute_rest_days(
    form: TeamForm | None,
    match_date: datetime | None,
) -> tuple[int | None, float | None]:
    """Return (rest_days_before_match, avg_days_between_recent_matches).

    rest_days: days between the team's most recent match and the upcoming match.
    avg_days_between: average gap between consecutive recent matches.
    """
    if not form or not form.recent_matches:
        return None, None

    dated = sorted(
        (m for m in form.recent_matches if m.date),
        key=lambda m: m.date,  # type: ignore[arg-type]
        reverse=True,
    )
    if not dated:
        return None, None

    rest_days = None
    if match_date:
        most_recent = dated[0].date
        if most_recent:
            # Strip tzinfo for safe subtraction
            md = match_date.replace(tzinfo=None) if match_date.tzinfo else match_date
            mr = most_recent.replace(tzinfo=None) if most_recent.tzinfo else most_recent
            rest_days = (md - mr).days

    avg_gap = None
    if len(dated) >= 2:
        gaps: list[int] = []
        for a, b in zip(dated, dated[1:]):
            if a.date and b.date:
                da = a.date.replace(tzinfo=None) if a.date.tzinfo else a.date
                db = b.date.replace(tzinfo=None) if b.date.tzinfo else b.date
                gaps.append((da - db).days)
        if gaps:
            avg_gap = round(sum(gaps) / len(gaps), 1)

    return rest_days, avg_gap


def _parse_fixture_statistics(stats_response: list[dict], team_id: int) -> dict[str, float] | None:
    """Extract key statistics for a specific team from /fixtures/statistics response."""
    for team_stats in stats_response:
        if team_stats.get("team", {}).get("id") != team_id:
            continue
        result: dict[str, float] = {}
        for stat in team_stats.get("statistics", []):
            stat_type = stat.get("type", "")
            value = stat.get("value")
            if value is None:
                continue
            if stat_type == "Ball Possession":
                # e.g. "55%" -> 55.0
                try:
                    result["possession"] = float(str(value).replace("%", ""))
                except ValueError:
                    pass
            elif stat_type == "Total Shots":
                result["shots_total"] = float(value)
            elif stat_type == "Shots on Goal":
                result["shots_on_target"] = float(value)
            elif stat_type == "Corner Kicks":
                result["corners"] = float(value)
            elif stat_type == "Fouls":
                result["fouls"] = float(value)
            elif stat_type == "Blocked Shots":
                result["blocked_shots"] = float(value)
            elif stat_type == "Goalkeeper Saves":
                result["gk_saves"] = float(value)
            elif stat_type == "Passes %":
                try:
                    result["pass_accuracy"] = float(str(value).replace("%", ""))
                except ValueError:
                    pass
            elif stat_type == "Yellow Cards":
                result["yellow_cards"] = float(value)
            elif stat_type == "Red Cards":
                result["red_cards"] = float(value)
            elif stat_type == "Offsides":
                result["offsides"] = float(value)
            elif stat_type == "Shots insidebox":
                result["shots_insidebox"] = float(value)
            elif stat_type == "Shots outsidebox":
                result["shots_outsidebox"] = float(value)
            elif stat_type == "Total passes":
                result["total_passes"] = float(value)
        return result if result else None
    return None


class ApiFootballStatsCollector:
    """Collect match statistics from API-Football v3."""

    def __init__(self) -> None:
        from toto_ai.config import settings

        self._fetch_match_stats = settings.API_FOOTBALL_FETCH_MATCH_STATS
        self._fixture_stats_cache: dict[int, list[dict]] = {}
        self._client = ApiFootballClient(
            api_key=settings.API_FOOTBALL_API_KEY,
            host=settings.API_FOOTBALL_HOST,
            use_rapidapi=settings.API_FOOTBALL_USE_RAPIDAPI,
        )
        self._mapper = TeamMapper(settings.TEAM_CACHE_FILE)
        self._max_concurrent = settings.API_FOOTBALL_MAX_CONCURRENT
        self._standings_cache: dict[tuple[int, int], list] = {}  # (league_id, season) -> data

    @property
    def api_call_count(self) -> int:
        return self._client.call_count

    async def _find_fixture_id(
        self,
        home_id: int,
        away_id: int,
        match_date: datetime | None,
        client: httpx.AsyncClient,
    ) -> tuple[int | None, str]:
        """Find the fixture ID and referee for a specific match."""
        params: dict = {"team": home_id, "season": _current_season()}
        if match_date:
            params["date"] = match_date.strftime("%Y-%m-%d")
        else:
            params["next"] = 5

        try:
            data = await self._client.get("fixtures", params, client=client)
            for fixture in data.get("response", []):
                teams = fixture.get("teams", {})
                info = fixture.get("fixture", {})
                matched = (
                    teams.get("home", {}).get("id") == home_id
                    and teams.get("away", {}).get("id") == away_id
                ) or (
                    teams.get("home", {}).get("id") == away_id
                    and teams.get("away", {}).get("id") == home_id
                )
                if matched:
                    return info.get("id"), (info.get("referee") or "")
        except Exception as e:
            console.print(f"[dim]Fixture lookup failed: {e}[/dim]")

        return None, ""

    async def _fetch_standings(
        self, league_id: int, season: int, client: httpx.AsyncClient
    ) -> list:
        """Fetch standings for a league, using cache."""
        cache_key = (league_id, season)
        if cache_key in self._standings_cache:
            return self._standings_cache[cache_key]

        try:
            data = await self._client.get(
                "standings",
                {"league": league_id, "season": season},
                client=client,
            )
            response = data.get("response", [])
            if response:
                standings = response[0].get("league", {}).get("standings", [])
                self._standings_cache[cache_key] = standings
                return standings
        except Exception as e:
            console.print(f"[dim]Standings fetch failed for league {league_id}: {e}[/dim]")

        return []

    async def _fetch_form_stats(
        self,
        form: TeamForm,
        team_id: int,
        client: httpx.AsyncClient,
    ) -> TeamFormStats | None:
        """Fetch and aggregate match statistics for a team's recent matches."""
        fixture_ids = [m.fixture_id for m in form.recent_matches if m.fixture_id is not None]
        if not fixture_ids:
            return None

        # Fetch statistics for each fixture (using cache to avoid duplicates)
        all_stats: list[dict[str, float]] = []
        for fid in fixture_ids:
            if fid not in self._fixture_stats_cache:
                try:
                    data = await self._client.get(
                        "fixtures/statistics", {"fixture": fid}, client=client
                    )
                    self._fixture_stats_cache[fid] = data.get("response", [])
                except Exception as e:
                    console.print(f"[dim]Stats fetch failed for fixture {fid}: {e}[/dim]")
                    self._fixture_stats_cache[fid] = []

            parsed = _parse_fixture_statistics(self._fixture_stats_cache[fid], team_id)
            if parsed:
                all_stats.append(parsed)

        if not all_stats:
            return None

        n = len(all_stats)
        return TeamFormStats(
            avg_possession=round(sum(s.get("possession", 0) for s in all_stats) / n, 1),
            avg_shots_total=round(sum(s.get("shots_total", 0) for s in all_stats) / n, 1),
            avg_shots_on_target=round(sum(s.get("shots_on_target", 0) for s in all_stats) / n, 1),
            avg_corners=round(sum(s.get("corners", 0) for s in all_stats) / n, 1),
            avg_fouls=round(sum(s.get("fouls", 0) for s in all_stats) / n, 1),
            avg_blocked_shots=round(sum(s.get("blocked_shots", 0) for s in all_stats) / n, 1),
            avg_gk_saves=round(sum(s.get("gk_saves", 0) for s in all_stats) / n, 1),
            avg_pass_accuracy=round(sum(s.get("pass_accuracy", 0) for s in all_stats) / n, 1),
            avg_yellow_cards=round(sum(s.get("yellow_cards", 0) for s in all_stats) / n, 1),
            avg_red_cards=round(sum(s.get("red_cards", 0) for s in all_stats) / n, 1),
            avg_offsides=round(sum(s.get("offsides", 0) for s in all_stats) / n, 1),
            avg_shots_insidebox=round(sum(s.get("shots_insidebox", 0) for s in all_stats) / n, 1),
            avg_shots_outsidebox=round(sum(s.get("shots_outsidebox", 0) for s in all_stats) / n, 1),
            avg_total_passes=round(sum(s.get("total_passes", 0) for s in all_stats) / n, 1),
            matches_with_stats=n,
        )

    async def research_match(
        self,
        home_team: str,
        away_team: str,
        league: str = "",
        match_date: datetime | None = None,
    ) -> MatchStats:
        """Research a single match via API-Football v3."""
        label = f"{home_team} vs {away_team}"
        season = _current_season()

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            api_headers = self._client._headers()
            base_url = self._client._base_url

            # Resolve team IDs
            home_info = await self._mapper.resolve(
                home_team, client, api_headers=api_headers, base_url=base_url
            )
            away_info = await self._mapper.resolve(
                away_team, client, api_headers=api_headers, base_url=base_url
            )

            if not home_info or not away_info:
                console.print(f"[yellow]Cannot resolve team IDs for {label}[/yellow]")
                return MatchStats(
                    home_team=home_team,
                    away_team=away_team,
                    home_team_english=home_info.english_name if home_info else None,
                    away_team_english=away_info.english_name if away_info else None,
                    match_date=match_date,
                )

            home_id = home_info.v3_id
            away_id = away_info.v3_id
            home_english = home_info.english_name
            away_english = away_info.english_name

            # Find fixture ID and referee (for injuries, odds, predictions)
            fixture_id, referee = await self._find_fixture_id(home_id, away_id, match_date, client)

            # Fetch data in parallel
            h2h_task = self._client.get(
                "fixtures/headtohead",
                {"h2h": f"{home_id}-{away_id}", "last": 10},
                client=client,
            )
            home_form_task = self._client.get(
                "fixtures",
                {"team": home_id, "last": 5},
                client=client,
            )
            away_form_task = self._client.get(
                "fixtures",
                {"team": away_id, "last": 5},
                client=client,
            )

            # Standings — use league_id from either team
            league_id = home_info.league_id or away_info.league_id

            tasks: list = [h2h_task, home_form_task, away_form_task]
            standings_idx = None
            if league_id:
                standings_idx = len(tasks)
                tasks.append(self._fetch_standings(league_id, season, client))

            injuries_idx = odds_idx = None
            if fixture_id:
                injuries_idx = len(tasks)
                tasks.append(self._client.get("injuries", {"fixture": fixture_id}, client=client))
                odds_idx = len(tasks)
                tasks.append(self._client.get("odds", {"fixture": fixture_id}, client=client))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions from the parallel tasks
            task_names = ["H2H", "Home form", "Away form"]
            if standings_idx is not None:
                task_names.append("Standings")
            if fixture_id:
                task_names.extend(["Injuries", "Odds"])
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    name = task_names[idx] if idx < len(task_names) else f"Task {idx}"
                    console.print(f"[yellow]{name} fetch failed for {label}: {result}[/yellow]")

            # Parse H2H
            h2h = None
            h2h_result = results[0]
            if isinstance(h2h_result, dict):
                h2h_fixtures = h2h_result.get("response", [])
                if h2h_fixtures:
                    h2h = _build_h2h(h2h_fixtures, home_english, away_english)

            # Parse home form
            home_form = None
            home_form_result = results[1]
            if isinstance(home_form_result, dict):
                home_fixtures = home_form_result.get("response", [])
                if home_fixtures:
                    home_form = _build_team_form(home_fixtures, home_english, home_id)

            # Parse away form
            away_form = None
            away_form_result = results[2]
            if isinstance(away_form_result, dict):
                away_fixtures = away_form_result.get("response", [])
                if away_fixtures:
                    away_form = _build_team_form(away_fixtures, away_english, away_id)

            # Parse standings
            home_standing = away_standing = None
            if standings_idx is not None:
                standings_result = results[standings_idx]
                if isinstance(standings_result, list) and standings_result:
                    home_standing = _find_standing(standings_result, home_id, home_english)
                    away_standing = _find_standing(standings_result, away_id, away_english)

            # Retry standings if missing — refresh league_id from API
            if home_standing is None and away_standing is None:
                api_headers = self._client._headers()
                base_url = self._client._base_url
                new_league_id = await self._mapper.refresh_league_id(
                    home_id, client, api_headers=api_headers, base_url=base_url
                )
                if new_league_id and new_league_id != league_id:
                    console.print(
                        f"[dim]League ID updated for {home_english}: "
                        f"{league_id} → {new_league_id}[/dim]"
                    )
                    self._mapper.update_league_id(home_team, new_league_id)
                    retry_standings = await self._fetch_standings(new_league_id, season, client)
                    if retry_standings:
                        home_standing = _find_standing(retry_standings, home_id, home_english)
                        away_standing = _find_standing(retry_standings, away_id, away_english)

            # Parse injuries
            home_injuries: list[InjuryInfo] = []
            away_injuries: list[InjuryInfo] = []
            if injuries_idx is not None:
                inj_result = results[injuries_idx]
                if isinstance(inj_result, dict):
                    all_injuries = _parse_injuries(inj_result.get("response", []))
                    for inj in all_injuries:
                        if (
                            inj.team_name.lower() in home_english.lower()
                            or home_english.lower() in inj.team_name.lower()
                        ):
                            home_injuries.append(inj)
                        else:
                            away_injuries.append(inj)

            # Parse odds
            odds = None
            if odds_idx is not None:
                odds_result = results[odds_idx]
                if isinstance(odds_result, dict):
                    odds_response = odds_result.get("response", [])
                    if odds_response:
                        bookmakers = odds_response[0].get("bookmakers", [])
                        odds = _parse_odds(bookmakers)

            # Fetch match statistics (opt-in, costs extra API calls)
            if self._fetch_match_stats:
                stats_tasks = []
                if home_form:
                    stats_tasks.append(self._fetch_form_stats(home_form, home_id, client))
                if away_form:
                    stats_tasks.append(self._fetch_form_stats(away_form, away_id, client))
                if stats_tasks:
                    stats_results = await asyncio.gather(*stats_tasks, return_exceptions=True)
                    idx = 0
                    if home_form:
                        r = stats_results[idx]
                        if isinstance(r, TeamFormStats):
                            home_form.form_stats = r
                        idx += 1
                    if away_form:
                        r = stats_results[idx]
                        if isinstance(r, TeamFormStats):
                            away_form.form_stats = r

            # Compute fixture congestion
            home_rest, home_avg_gap = _compute_rest_days(home_form, match_date)
            away_rest, away_avg_gap = _compute_rest_days(away_form, match_date)

            return MatchStats(
                home_team=home_team,
                away_team=away_team,
                home_team_id=home_id,
                away_team_id=away_id,
                home_team_english=home_english,
                away_team_english=away_english,
                h2h=h2h,
                home_form=home_form,
                away_form=away_form,
                home_standing=home_standing,
                away_standing=away_standing,
                home_injuries=home_injuries,
                away_injuries=away_injuries,
                odds=odds,
                match_date=match_date,
                referee=referee,
                home_rest_days=home_rest,
                away_rest_days=away_rest,
                home_avg_days_between=home_avg_gap,
                away_avg_days_between=away_avg_gap,
            )

    async def research_all_matches(
        self,
        matches: list[tuple[str, str, str, datetime | None]],
    ) -> list[MatchStats]:
        """Research all matches with controlled concurrency.

        Each tuple is (home_team, away_team, league, match_date).
        """
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _research_with_limit(
            idx: int, home: str, away: str, league: str, match_date: datetime | None
        ) -> tuple[int, MatchStats]:
            async with semaphore:
                console.print(f"[dim]Researching: {home} vs {away}...[/dim]")
                try:
                    stats = await self.research_match(home, away, league, match_date)
                except Exception as e:
                    console.print(f"[yellow]Research failed for {home} vs {away}: {e}[/yellow]")
                    stats = MatchStats(home_team=home, away_team=away, match_date=match_date)
                return idx, stats

        tasks = [
            _research_with_limit(i, home, away, league, md)
            for i, (home, away, league, md) in enumerate(matches)
        ]
        results = await asyncio.gather(*tasks)

        ordered = [MatchStats(home_team="", away_team="")] * len(matches)
        for idx, stats in results:
            ordered[idx] = stats

        console.print(f"[dim]API-Football: {self._client.call_count} API calls used[/dim]")
        return ordered
