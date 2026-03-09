"""Download Israeli league historical match data from API-Football.

Fetches fixtures, match statistics, and odds for Israeli leagues,
saving them as CSVs in the same format as football-data.co.uk main leagues.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from toto_ai.console import console
from toto_ai.stats.api_football_client import ApiFootballClient

# Israeli league configurations: division code → expected league name substring
ISRAELI_LEAGUES: dict[str, str] = {
    "ISR1": "Ligat Ha'al",
    "ISR_CUP": "State Cup",
}

# API-Football stat type → football-data.co.uk column names (home, away)
STAT_MAP: dict[str, tuple[str, str]] = {
    "Total Shots": ("HS", "AS"),
    "Shots on Goal": ("HST", "AST"),
    "Corner Kicks": ("HC", "AC"),
    "Fouls": ("HF", "AF"),
    "Yellow Cards": ("HY", "AY"),
    "Red Cards": ("HR", "AR"),
    "Ball Possession": ("HPoss", "APoss"),
    "Blocked Shots": ("HBS", "ABS"),
    "Goalkeeper Saves": ("HGS", "AGS"),
    "Passes %": ("HPP", "APP"),
}

# Stat types whose API values contain a '%' suffix (e.g. "55%")
_PCT_STATS = {"Ball Possession", "Passes %"}

# CSV columns in football-data.co.uk order
CSV_COLUMNS = [
    "Div",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    "HPoss",
    "APoss",
    "HBS",
    "ABS",
    "HGS",
    "AGS",
    "HPP",
    "APP",
    "B365H",
    "B365D",
    "B365A",
]


def _season_code(year: int) -> str:
    """Convert API-Football season year to football-data season code (e.g. 2024 → '2425')."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _format_date(iso_date: str) -> str:
    """Convert ISO date string to DD/MM/YYYY format."""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return ""


def _result_from_goals(home: int, away: int) -> str:
    if home > away:
        return "H"
    elif away > home:
        return "A"
    return "D"


class ApiFootballDownloader:
    """Downloads Israeli league historical data from API-Football v3."""

    def __init__(
        self,
        api_key: str,
        data_dir: str = "data/football_data",
        host: str = "v3.football.api-sports.io",
        use_rapidapi: bool = False,
        max_concurrent: int = 4,
    ) -> None:
        self._client = ApiFootballClient(
            api_key=api_key,
            host=host,
            use_rapidapi=use_rapidapi,
            max_concurrent=max_concurrent,
        )
        self._data_dir = Path(data_dir)
        self._league_ids: dict[str, int] = {}  # div_code → API-Football league ID

    def _progress_path(self, div_code: str) -> Path:
        return self._data_dir / div_code / ".progress.json"

    def _load_progress(self, div_code: str) -> dict:
        path = self._progress_path(div_code)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _save_progress(self, div_code: str, progress: dict) -> None:
        path = self._progress_path(div_code)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    async def discover_leagues(self, http: httpx.AsyncClient) -> dict[str, int]:
        """Discover Israeli league IDs from API-Football."""
        data = await self._client.get("leagues", params={"country": "Israel"}, client=http)
        leagues = data.get("response", [])

        for item in leagues:
            league = item.get("league", {})
            name = league.get("name", "")
            league_id = league.get("id")

            for div_code, expected_name in ISRAELI_LEAGUES.items():
                if expected_name.lower() in name.lower() and div_code not in self._league_ids:
                    self._league_ids[div_code] = league_id
                    console.print(f"  [green]Found {div_code}:[/green] {name} (ID: {league_id})")

        missing = set(ISRAELI_LEAGUES.keys()) - set(self._league_ids.keys())
        if missing:
            console.print(f"  [yellow]Could not find leagues: {', '.join(missing)}[/yellow]")
            console.print("  Available Israeli leagues:")
            for item in leagues:
                lg = item.get("league", {})
                console.print(f"    - {lg.get('name')} (ID: {lg.get('id')})")

        return self._league_ids

    async def _fetch_season_fixtures(
        self, http: httpx.AsyncClient, league_id: int, season: int
    ) -> list[dict]:
        """Fetch all finished fixtures for a league/season."""
        data = await self._client.get(
            "fixtures",
            params={"league": league_id, "season": season, "status": "FT"},
            client=http,
        )
        return data.get("response", [])

    async def _fetch_fixture_stats(
        self, http: httpx.AsyncClient, fixture_id: int
    ) -> dict[str, dict[str, int | None]]:
        """Fetch match statistics for a single fixture.

        Returns {"home": {"HS": 12, ...}, "away": {"AS": 8, ...}}.
        """
        data = await self._client.get(
            "fixtures/statistics",
            params={"fixture": fixture_id},
            client=http,
        )
        response = data.get("response", [])

        result: dict[str, dict[str, int | None]] = {"home": {}, "away": {}}

        for i, team_stats in enumerate(response[:2]):
            side = "home" if i == 0 else "away"
            stats_list = team_stats.get("statistics", [])
            for stat in stats_list:
                stat_type = stat.get("type", "")
                if stat_type in STAT_MAP:
                    col_home, col_away = STAT_MAP[stat_type]
                    col = col_home if side == "home" else col_away
                    value = stat.get("value")
                    if isinstance(value, str):
                        cleaned = value.replace("%", "").strip()
                        if stat_type in _PCT_STATS:
                            try:
                                value = float(cleaned)
                            except ValueError:
                                value = None
                        else:
                            value = int(cleaned) if cleaned.isdigit() else None
                    result[side][col] = value

        return result

    async def _fetch_fixture_odds(
        self, http: httpx.AsyncClient, fixture_id: int
    ) -> tuple[float | None, float | None, float | None]:
        """Fetch match winner odds for a fixture. Returns (home, draw, away)."""
        data = await self._client.get(
            "odds",
            params={"fixture": fixture_id},
            client=http,
        )
        response = data.get("response", [])
        if not response:
            return None, None, None

        # Look through bookmakers for Bet365 first, then fall back to first available
        bookmakers = response[0].get("bookmakers", [])
        target_bm = None
        for bm in bookmakers:
            if "bet365" in bm.get("name", "").lower():
                target_bm = bm
                break
        if not target_bm and bookmakers:
            target_bm = bookmakers[0]
        if not target_bm:
            return None, None, None

        # Find "Match Winner" bet
        for bet in target_bm.get("bets", []):
            if "match winner" in bet.get("name", "").lower():
                values = bet.get("values", [])
                home_odd = draw_odd = away_odd = None
                for v in values:
                    val_name = v.get("value", "").lower()
                    odd = v.get("odd")
                    if odd:
                        odd = float(odd)
                    if val_name == "home":
                        home_odd = odd
                    elif val_name == "draw":
                        draw_odd = odd
                    elif val_name == "away":
                        away_odd = odd
                return home_odd, draw_odd, away_odd

        return None, None, None

    def _fixture_to_row(self, fixture: dict, div_code: str) -> dict:
        """Convert an API-Football fixture to a CSV row dict (basic fields only)."""
        info = fixture.get("fixture", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        score = fixture.get("score", {})
        halftime = score.get("halftime", {})

        home_goals = goals.get("home") or 0
        away_goals = goals.get("away") or 0

        return {
            "Div": div_code,
            "Date": _format_date(info.get("date", "")),
            "HomeTeam": teams.get("home", {}).get("name", ""),
            "AwayTeam": teams.get("away", {}).get("name", ""),
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": _result_from_goals(home_goals, away_goals),
            "HTHG": halftime.get("home"),
            "HTAG": halftime.get("away"),
            # Stats and odds filled later
            "HS": None,
            "AS": None,
            "HST": None,
            "AST": None,
            "HC": None,
            "AC": None,
            "HF": None,
            "AF": None,
            "HY": None,
            "AY": None,
            "HR": None,
            "AR": None,
            "HPoss": None,
            "APoss": None,
            "HBS": None,
            "ABS": None,
            "HGS": None,
            "AGS": None,
            "HPP": None,
            "APP": None,
            "B365H": None,
            "B365D": None,
            "B365A": None,
            # Internal: fixture ID for enrichment
            "_fixture_id": info.get("id"),
        }

    async def download_season(
        self,
        http: httpx.AsyncClient,
        div_code: str,
        league_id: int,
        season: int,
        fetch_stats: bool = True,
        force: bool = False,
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> int:
        """Download a single season for a league. Returns number of matches saved."""
        sc = _season_code(season)
        csv_path = self._data_dir / div_code / f"{sc}.csv"

        # Load existing progress for this division
        prog_data = self._load_progress(div_code)
        season_key = str(season)
        enriched_ids: set[int] = set(prog_data.get(f"{season_key}_enriched", []))

        # Phase 1: Fetch all fixtures
        fixtures_fetched = False
        if csv_path.exists() and not force:
            # Load existing data
            df = pd.read_csv(csv_path, encoding="utf-8")
        else:
            fixtures = await self._fetch_season_fixtures(http, league_id, season)
            if not fixtures:
                console.print(f"  [dim]{div_code} {sc}: no fixtures found[/dim]")
                if progress and task_id is not None:
                    progress.advance(task_id)
                return 0

            rows = [self._fixture_to_row(f, div_code) for f in fixtures]
            df = pd.DataFrame(rows)
            fixtures_fetched = True

        if df.empty:
            if progress and task_id is not None:
                progress.advance(task_id)
            return 0

        # Phase 2 & 3: Enrich with stats and odds
        if fetch_stats and "_fixture_id" in df.columns:
            fixture_ids = df["_fixture_id"].dropna().astype(int).tolist()
            to_enrich = [fid for fid in fixture_ids if fid not in enriched_ids]

            if to_enrich:
                desc = f"{div_code} {sc} stats+odds"
                if progress and task_id is not None:
                    # Create a sub-task for enrichment
                    enrich_task = progress.add_task(f"  {desc}", total=len(to_enrich))
                else:
                    enrich_task = None

                for fid in to_enrich:
                    idx = df.index[df["_fixture_id"] == fid]
                    if idx.empty:
                        if enrich_task is not None and progress:
                            progress.advance(enrich_task)
                        continue

                    # Fetch stats and odds concurrently
                    stats_result, odds_result = await asyncio.gather(
                        self._fetch_fixture_stats(http, fid),
                        self._fetch_fixture_odds(http, fid),
                    )

                    # Apply stats
                    for col, val in stats_result.get("home", {}).items():
                        df.loc[idx, col] = val
                    for col, val in stats_result.get("away", {}).items():
                        df.loc[idx, col] = val

                    # Apply odds
                    home_odd, draw_odd, away_odd = odds_result
                    df.loc[idx, "B365H"] = home_odd
                    df.loc[idx, "B365D"] = draw_odd
                    df.loc[idx, "B365A"] = away_odd

                    enriched_ids.add(fid)
                    if enrich_task is not None and progress:
                        progress.advance(enrich_task)

                # Save enrichment progress
                prog_data[f"{season_key}_enriched"] = list(enriched_ids)
                self._save_progress(div_code, prog_data)

        # Save CSV (drop internal columns)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_cols = [c for c in CSV_COLUMNS if c in df.columns]
        df[out_cols].to_csv(csv_path, index=False, encoding="utf-8")

        match_count = len(df)
        status = "downloaded" if fixtures_fetched else "enriched"
        console.print(f"  [green]{div_code} {sc}:[/green] {match_count} matches ({status})")

        if progress and task_id is not None:
            progress.advance(task_id)

        return match_count

    async def download_all(
        self,
        seasons: list[int] | None = None,
        fetch_stats: bool = True,
        force: bool = False,
    ) -> None:
        """Download all Israeli league data."""
        if seasons is None:
            seasons = list(range(2019, 2026))

        console.print("[bold blue]Discovering Israeli leagues on API-Football...[/bold blue]")

        async with httpx.AsyncClient(timeout=30) as http:
            league_ids = await self.discover_leagues(http)

            if not league_ids:
                console.print("[red]No Israeli leagues found. Check API key.[/red]")
                return

            total_tasks = len(league_ids) * len(seasons)
            total_matches = 0

            console.print(
                f"\n[bold]Downloading {len(league_ids)} leagues × "
                f"{len(seasons)} seasons = {total_tasks} season files[/bold]"
            )
            if fetch_stats:
                console.print("[dim]Fetching per-match stats + odds (this may take a while)[/dim]")
            console.print()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                main_task = progress.add_task("Seasons", total=total_tasks)

                for div_code, league_id in league_ids.items():
                    for season in seasons:
                        count = await self.download_season(
                            http=http,
                            div_code=div_code,
                            league_id=league_id,
                            season=season,
                            fetch_stats=fetch_stats,
                            force=force,
                            progress=progress,
                            task_id=main_task,
                        )
                        total_matches += count

            console.print(
                f"\n[bold green]Done![/bold green] "
                f"{total_matches} total matches across {len(league_ids)} leagues"
            )
            console.print(f"API calls used: {self._client.call_count}")
