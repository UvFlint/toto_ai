"""Scrape historical per-match statistics from SofaScore for Israeli leagues.

Uses Selenium to establish a browser session on SofaScore, then makes fetch()
calls from the browser context to SofaScore's internal API. This avoids the
403 blocks on direct HTTP requests while being much faster than page-by-page
navigation (~4100 matches in ~1-2 hours instead of ~4.5 hours).
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from toto_ai.console import console

SOFASCORE_BASE_URL = "https://www.sofascore.com"
TOURNAMENT_ID = 266  # Israeli Premier League

# SofaScore API stat name → (home CSV column, away CSV column)
# Keys verified from SofaScore's API response
STAT_MAP: dict[str, tuple[str, str]] = {
    "Total shots": ("HS", "AS"),
    "Shots on target": ("HST", "AST"),
    "Corner kicks": ("HC", "AC"),
    "Fouls": ("HF", "AF"),
    "Yellow cards": ("HY", "AY"),
    "Red cards": ("HR", "AR"),
}

# SofaScore team name → canonical CSV name (API-Football convention)
TEAM_ALIASES: dict[str, str] = {
    "Maccabi Tel-Aviv": "Maccabi Tel Aviv",
    "Maccabi Tel Aviv": "Maccabi Tel Aviv",
    "Hapoel Tel-Aviv": "Hapoel Tel Aviv",
    "Hapoel Tel Aviv": "Hapoel Tel Aviv",
    "Hapoel Be'er Sheva": "Hapoel Beer Sheva",
    "Hapoel Beer Sheva": "Hapoel Beer Sheva",
    "Hapoel Be'er Sheva FC": "Hapoel Beer Sheva",
    "Maccabi Haifa": "Maccabi Haifa",
    "Hapoel Haifa": "Hapoel Haifa",
    "Beitar Jerusalem": "Beitar Jerusalem",
    "Bnei Yehuda Tel-Aviv": "Bnei Yehuda",
    "Bnei Yehuda Tel Aviv": "Bnei Yehuda",
    "Bnei Yehuda": "Bnei Yehuda",
    "Ironi Kiryat Shmona": "Ironi Kiryat Shmona",
    "Hapoel Kiryat Shmona": "Ironi Kiryat Shmona",
    "Maccabi Netanya": "Maccabi Netanya",
    "Maccabi Petah Tikva": "Maccabi Petah Tikva",
    "Maccabi Petach Tikva": "Maccabi Petah Tikva",
    "Hapoel Petah Tikva": "Hapoel Petah Tikva",
    "Hapoel Petach Tikva": "Hapoel Petah Tikva",
    "Bnei Sakhnin": "Bnei Sakhnin",
    "FC Ashdod": "Ashdod",
    "MS Ashdod": "Ashdod",
    "Maccabi Bnei Reineh": "Maccabi Bnei Raina",
    "Maccabi Bnei Raina": "Maccabi Bnei Raina",
    "Hapoel Kfar Saba": "Hapoel Kfar Saba",
    "Hapoel Kfar Shalem": "Hapoel Kfar Saba",
    "Sektzia Ness Ziona": "Sektzia Nes Tziona",
    "Sektzia Nes Tziona": "Sektzia Nes Tziona",
    "Hapoel Ra'anana": "Hapoel Ra'anana",
    "Hapoel Raanana": "Hapoel Ra'anana",
    "Hapoel Katamon Jerusalem": "Hapoel Katamon",
    "Hapoel Katamon": "Hapoel Katamon",
    "Hapoel Jerusalem": "Hapoel Katamon",
    "Hapoel Nof HaGalil": "Hapoel Nof HaGalil",
    "Hapoel Nazareth Illit": "Hapoel Nof HaGalil",
    "Hapoel Nir Ramat HaSharon": "Hapoel Ramat HaSharon",
    "Hapoel Ramat Gan": "Hap. Ramat Gan",
    "Hapoel Akko": "H. Akko",
    "Hapoel Acre": "H. Akko",
    "Ironi Tiberias": "Ironi Tiberias",
    "Hapoel Hadera": "Hapoel Hadera",
    "Hapoel Ashkelon": "Ashkelon",
}


def _season_code(year: int) -> str:
    """Convert start year to season code (e.g. 2024 → '2425')."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _normalize_team(name: str) -> str:
    """Normalize team name via alias lookup."""
    return TEAM_ALIASES.get(name, name)


def _fuzzy_team_match(name: str, csv_teams: set[str]) -> str | None:
    """Try to fuzzy-match a SofaScore team name against CSV team names."""
    if name in csv_teams:
        return name

    name_lower = name.lower()

    for csv_name in csv_teams:
        csv_lower = csv_name.lower()
        if name_lower in csv_lower or csv_lower in name_lower:
            return csv_name

    CITY_ALIASES = {
        "nof hagalil": "nazareth illit",
        "nazareth illit": "nof hagalil",
    }
    for old, new in CITY_ALIASES.items():
        if old in name_lower:
            alt = name_lower.replace(old, new)
            for csv_name in csv_teams:
                if alt in csv_name.lower() or csv_name.lower() in alt:
                    return csv_name

    return None


class DriverCrashError(Exception):
    """Raised when the Selenium driver crashes (connection lost)."""


class SofaScoreScraper:
    """Scrape per-match statistics from SofaScore using Selenium + browser fetch().

    Instead of navigating to each match page, we establish one browser session
    and make API calls via JavaScript fetch() from the browser context.
    This is fast because SofaScore's API recognizes the browser session's cookies.
    """

    def __init__(self, headless: bool = True, delay: float = 2.0) -> None:
        self._headless = headless
        self._delay = delay

    def _create_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.page_load_strategy = "eager"
        if self._headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-extensions")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        )

        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        driver.set_page_load_timeout(30)
        driver.set_script_timeout(30)
        return driver

    def _init_session(self, driver) -> None:
        """Navigate to SofaScore to establish session cookies."""
        driver.get(
            f"{SOFASCORE_BASE_URL}/tournament/football/israel/"
            f"israeli-premier-league/{TOURNAMENT_ID}"
        )
        time.sleep(6)

    def _api_fetch(self, driver, path: str, retries: int = 3) -> dict | None:
        """Make a fetch() call to SofaScore's API from the browser context.

        Args:
            driver: Selenium WebDriver with an active SofaScore session.
            path: API path (e.g., '/api/v1/unique-tournament/266/seasons').
            retries: Number of retry attempts on failure.

        Returns:
            Parsed JSON response, or None on error.
        """
        for attempt in range(retries + 1):
            try:
                result = driver.execute_async_script(
                    """
                    var callback = arguments[arguments.length - 1];
                    fetch(arguments[0])
                        .then(function(r) { return r.text(); })
                        .then(function(text) { callback(text); })
                        .catch(function(e) { callback('FETCH_ERROR: ' + e.message); });
                    """,
                    path,
                )
                if not result or (isinstance(result, str) and result.startswith("FETCH_ERROR:")):
                    if attempt < retries:
                        time.sleep(2 + 2 ** attempt)
                        continue
                    return None
                data = json.loads(result)
                # SofaScore may return error responses
                if isinstance(data, dict) and data.get("error"):
                    if attempt < retries:
                        time.sleep(2 + 2 ** attempt)
                        continue
                    return None
                return data
            except json.JSONDecodeError:
                if attempt < retries:
                    time.sleep(2 + 2 ** attempt)
                    continue
                return None
            except Exception as e:
                if attempt < retries:
                    time.sleep(2 + 2 ** attempt)
                    continue
                raise DriverCrashError(str(e)) from e
        return None

    def _discover_season_ids(self, driver) -> dict[int, int]:
        """Fetch all available season IDs from SofaScore API.

        Returns dict of {start_year: sofascore_season_id}.
        """
        data = self._api_fetch(
            driver, f"/api/v1/unique-tournament/{TOURNAMENT_ID}/seasons"
        )
        if not data:
            return {}

        season_map: dict[int, int] = {}
        for s in data.get("seasons", []):
            season_id = s.get("id")
            name = s.get("name", "")
            year = s.get("year", "")
            start_year = _parse_season_year(name, year)
            if start_year and season_id:
                season_map[start_year] = season_id

        if season_map:
            console.print(
                f"  [green]Discovered {len(season_map)} seasons[/green] "
                f"({min(season_map)}/{min(season_map)+1} to "
                f"{max(season_map)}/{max(season_map)+1})"
            )
        else:
            console.print("  [red]Could not discover season IDs[/red]")

        return season_map

    def _get_season_matches(
        self, driver, season_id: int, start_year: int
    ) -> list[dict]:
        """Fetch all finished match events for a season via round-based API calls.

        Uses the rounds API to get complete season data. Falls back to paginated
        events/last endpoint if rounds API is unavailable.

        Returns list of match dicts with: id, date, home_team, away_team,
        home_goals, away_goals, ht_home_goals, ht_away_goals.
        """
        all_matches: list[dict] = []
        seen_ids: set[int] = set()

        # First, get the list of rounds for this season
        rounds_data = self._api_fetch(
            driver,
            f"/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/rounds",
        )

        rounds: list[int] = []
        if rounds_data and "rounds" in rounds_data:
            rounds = [r.get("round", 0) for r in rounds_data["rounds"] if r.get("round")]

        if rounds:
            # Use round-based event listing (more complete)
            for round_num in rounds:
                data = self._api_fetch(
                    driver,
                    f"/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{season_id}"
                    f"/events/round/{round_num}",
                )
                if data:
                    self._extract_events(data.get("events", []), all_matches, seen_ids)
                time.sleep(1.0 + random.uniform(0, 0.5))
        else:
            # Fallback: paginated events/last endpoint
            for page in range(50):
                data = self._api_fetch(
                    driver,
                    f"/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{season_id}"
                    f"/events/last/{page}",
                )
                if not data:
                    break
                events = data.get("events", [])
                if not events:
                    break
                self._extract_events(events, all_matches, seen_ids)
                time.sleep(1.0 + random.uniform(0, 0.5))

        return all_matches

    def _extract_events(
        self, events: list[dict], all_matches: list[dict], seen_ids: set[int]
    ) -> None:
        """Extract finished match data from SofaScore event list."""
        for event in events:
            match_id = event.get("id")
            if not match_id or match_id in seen_ids:
                continue

            status = event.get("status", {})
            if status.get("type") != "finished":
                continue

            home_team = event.get("homeTeam", {}).get("name", "")
            away_team = event.get("awayTeam", {}).get("name", "")
            home_score = event.get("homeScore", {})
            away_score = event.get("awayScore", {})

            timestamp = event.get("startTimestamp")
            date_str = ""
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime("%d/%m/%Y")

            if not home_team or not away_team:
                continue

            seen_ids.add(match_id)
            all_matches.append({
                "id": match_id,
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_score.get("current"),
                "away_goals": away_score.get("current"),
                "ht_home_goals": home_score.get("period1"),
                "ht_away_goals": away_score.get("period1"),
            })

    def _fetch_match_stats(self, driver, match_id: int) -> dict:
        """Fetch statistics for a single match via API.

        Returns dict with CSV column names as keys (HS, AS, HST, AST, etc.),
        or empty dict if stats unavailable.
        """
        data = self._api_fetch(
            driver, f"/api/v1/event/{match_id}/statistics"
        )
        if not data:
            return {}

        stats: dict[str, float] = {}
        for period_data in data.get("statistics", []):
            period = period_data.get("period", "")
            if period != "ALL":
                continue

            for group in period_data.get("groups", []):
                for item in group.get("statisticsItems", []):
                    stat_name = item.get("name", "")
                    if stat_name in STAT_MAP:
                        home_col, away_col = STAT_MAP[stat_name]
                        try:
                            stats[home_col] = float(str(item.get("home", "")))
                            stats[away_col] = float(str(item.get("away", "")))
                        except (ValueError, TypeError):
                            pass

        return stats

    def scrape_season(
        self,
        driver,
        start_year: int,
        season_id: int,
        progress: dict,
        progress_path: Path | None = None,
    ) -> list[dict]:
        """Scrape all match stats for one season.

        Returns list of dicts with match info + stats, ready for CSV merge.
        Raises DriverCrashError if the browser dies (caller should recover).
        """
        sc = _season_code(start_year)
        season_key = str(start_year)

        # Phase 1: Get match list (use cached if available)
        if season_key in progress.get("season_matches", {}):
            matches = progress["season_matches"][season_key]
            console.print(
                f"  [dim]Using cached match list for {sc}: {len(matches)} matches[/dim]"
            )
        else:
            matches = self._get_season_matches(driver, season_id, start_year)
            progress.setdefault("season_matches", {})[season_key] = matches
            if progress_path:
                _save_progress(progress, progress_path)
            console.print(
                f"  [dim]Season {sc}: {len(matches)} matches found[/dim]"
            )

        if not matches:
            console.print(f"  [yellow]No matches found for {sc}[/yellow]")
            return []

        # Phase 2: Fetch stats for each match
        scraped_stats = progress.setdefault("scraped_stats", {})
        results: list[dict] = []
        stats_count = 0
        new_scraped = 0

        for i, match in enumerate(matches):
            match_id = match["id"]
            mid_str = str(match_id)

            if mid_str in scraped_stats:
                match_with_stats = {**match, **scraped_stats[mid_str]}
                results.append(match_with_stats)
                if scraped_stats[mid_str]:
                    stats_count += 1
                continue

            # Fetch stats — may raise DriverCrashError
            stats = self._fetch_match_stats(driver, match_id)
            scraped_stats[mid_str] = stats

            if stats:
                match_with_stats = {**match, **stats}
                stats_count += 1
            else:
                match_with_stats = {**match}

            results.append(match_with_stats)
            new_scraped += 1

            # Rate limiting
            time.sleep(self._delay + random.uniform(0, self._delay * 0.5))

            # Save progress every 20 newly fetched matches
            if new_scraped > 0 and new_scraped % 20 == 0:
                if progress_path:
                    _save_progress(progress, progress_path)
                console.print(
                    f"    [dim]{sc}: {new_scraped} fetched, "
                    f"{stats_count}/{len(results)} with stats[/dim]"
                )

        # Final save for this season
        if progress_path and new_scraped > 0:
            _save_progress(progress, progress_path)

        console.print(
            f"  [green]{sc}:[/green] {len(results)} matches, "
            f"{stats_count} with stats"
            + (f" ({new_scraped} newly fetched)" if new_scraped else " (all cached)")
        )
        return results

    def _create_session(self) -> object:
        """Create a new driver and establish SofaScore session."""
        driver = self._create_driver()
        self._init_session(driver)
        return driver

    def scrape_all(
        self,
        data_dir: str | Path,
        seasons: list[int] | None = None,
    ) -> None:
        """Scrape SofaScore stats for all requested seasons and merge into CSVs."""
        data_dir = Path(data_dir)
        progress_path = data_dir / "ISR1" / ".sofascore_progress.json"
        progress = _load_progress(progress_path)

        console.print("[dim]Establishing SofaScore session...[/dim]")
        driver = self._create_session()

        try:
            # Discover season IDs
            if progress.get("season_ids"):
                season_ids = {int(k): v for k, v in progress["season_ids"].items()}
                console.print(
                    f"[dim]Using cached season IDs ({len(season_ids)} seasons)[/dim]"
                )
            else:
                season_ids = self._discover_season_ids(driver)
                progress["season_ids"] = {str(k): v for k, v in season_ids.items()}
                _save_progress(progress, progress_path)

            if not season_ids:
                console.print("[red]No season IDs available. Aborting.[/red]")
                return

            # Default: scrape seasons that might have missing stats (2008-2022)
            if seasons is None:
                seasons = [y for y in range(2008, 2023) if y in season_ids]

            valid_seasons = [y for y in seasons if y in season_ids]
            skipped_seasons = [y for y in seasons if y not in season_ids]
            if skipped_seasons:
                console.print(
                    f"[yellow]Skipping seasons not on SofaScore: "
                    f"{[_season_code(y) for y in skipped_seasons]}[/yellow]"
                )

            console.print(
                f"\n[bold]Scraping {len(valid_seasons)} seasons from SofaScore...[/bold]\n"
            )

            all_results: list[dict] = []
            max_retries = 3

            for start_year in valid_seasons:
                sid = season_ids[start_year]

                for attempt in range(max_retries + 1):
                    try:
                        season_results = self.scrape_season(
                            driver, start_year, sid, progress, progress_path
                        )
                        all_results.extend(season_results)
                        break  # Success
                    except DriverCrashError:
                        _save_progress(progress, progress_path)
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        if attempt < max_retries:
                            console.print(
                                f"  [yellow]Browser crashed during "
                                f"{_season_code(start_year)}. "
                                f"Recovering... (attempt {attempt + 1}/{max_retries})[/yellow]"
                            )
                            time.sleep(15 * (attempt + 1))
                            driver = self._create_session()
                        else:
                            console.print(
                                f"  [red]Giving up on {_season_code(start_year)} "
                                f"after {max_retries} retries[/red]"
                            )
                            # Still need a driver for next seasons
                            driver = self._create_session()
                    except Exception as e:
                        console.print(
                            f"  [red]Error scraping {_season_code(start_year)}: {e}[/red]"
                        )
                        _save_progress(progress, progress_path)
                        break

                # Small delay between seasons
                if start_year != valid_seasons[-1]:
                    time.sleep(2 + random.uniform(0, 2))

        except Exception as e:
            console.print(f"[red]Fatal error: {e}[/red]")
        finally:
            _save_progress(progress, progress_path)
            try:
                driver.quit()
            except Exception:
                pass

        # Merge all results into CSVs
        if all_results:
            console.print(f"\n[bold]Merging {len(all_results)} matches into CSVs...[/bold]")
            merge_stats_to_csvs(all_results, data_dir)


def _parse_season_year(name: str, year: str | int) -> int | None:
    """Parse a start year from season name/year fields."""
    m = re.search(r"(\d{2,4})/(\d{2,4})", str(name))
    if m:
        y1 = m.group(1)
        if len(y1) == 2:
            y1 = "20" + y1 if int(y1) < 50 else "19" + y1
        return int(y1)

    m = re.search(r"(\d{2,4})/(\d{2,4})", str(year))
    if m:
        y1 = m.group(1)
        if len(y1) == 2:
            y1 = "20" + y1 if int(y1) < 50 else "19" + y1
        return int(y1)

    return None


def _load_progress(path: Path) -> dict:
    """Load progress tracking file."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_progress(progress: dict, path: Path) -> None:
    """Save progress tracking file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_stats_to_csvs(
    match_results: list[dict],
    data_dir: str | Path,
) -> dict[str, int]:
    """Merge scraped match stats into existing ISR1 CSV files.

    Matches by Date + HomeTeam + AwayTeam (with alias resolution and ±1 day tolerance).
    Only fills columns that are currently NaN — never overwrites existing data.

    Returns dict of {season_code: matches_updated}.
    """
    data_dir = Path(data_dir)
    div_dir = data_dir / "ISR1"
    results: dict[str, int] = {}

    if not match_results:
        return results

    # Group matches by season
    season_groups: dict[int, list[dict]] = {}
    for match in match_results:
        date_str = match.get("date", "")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%d/%m/%Y")
            start_year = dt.year if dt.month >= 8 else dt.year - 1
            season_groups.setdefault(start_year, []).append(match)
        except ValueError:
            continue

    stat_cols = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]

    for start_year, matches in sorted(season_groups.items()):
        sc = _season_code(start_year)
        csv_path = div_dir / f"{sc}.csv"

        if not csv_path.exists():
            console.print(f"  [yellow]{sc}: CSV not found, skipping[/yellow]")
            continue

        existing = pd.read_csv(csv_path, encoding="utf-8")
        csv_teams = set(existing["HomeTeam"].dropna().unique()) | set(
            existing["AwayTeam"].dropna().unique()
        )

        fuzzy_cache: dict[str, str] = {}

        def resolve_team(name: str) -> str:
            norm = _normalize_team(name)
            if norm in csv_teams:
                return norm
            if norm in fuzzy_cache:
                return fuzzy_cache[norm]
            fuzzy = _fuzzy_team_match(norm, csv_teams)
            if fuzzy:
                fuzzy_cache[norm] = fuzzy
                return fuzzy
            return norm

        matched = 0
        unmatched: list[str] = []

        for match in matches:
            home_resolved = resolve_team(match.get("home_team", ""))
            away_resolved = resolve_team(match.get("away_team", ""))
            date_str = match.get("date", "")

            has_stats = any(match.get(col) is not None for col in stat_cols)
            has_ht = match.get("ht_home_goals") is not None

            if not has_stats and not has_ht:
                continue

            # Try exact date match
            mask = (
                (existing["Date"] == date_str)
                & (existing["HomeTeam"] == home_resolved)
                & (existing["AwayTeam"] == away_resolved)
            )

            # Try ±1 day tolerance if no exact match
            if not mask.any() and date_str:
                try:
                    dt = datetime.strptime(date_str, "%d/%m/%Y")
                    for delta in [-1, 1]:
                        alt_date = (dt + timedelta(days=delta)).strftime("%d/%m/%Y")
                        alt_mask = (
                            (existing["Date"] == alt_date)
                            & (existing["HomeTeam"] == home_resolved)
                            & (existing["AwayTeam"] == away_resolved)
                        )
                        if alt_mask.any():
                            mask = alt_mask
                            break
                except ValueError:
                    pass

            if mask.any():
                idx = existing.index[mask]
                for col in stat_cols:
                    val = match.get(col)
                    if val is not None:
                        current = existing.loc[idx, col]
                        if current.isna().all() or (current == "").all():
                            existing.loc[idx, col] = val

                if has_ht:
                    for csv_col, match_key in [
                        ("HTHG", "ht_home_goals"),
                        ("HTAG", "ht_away_goals"),
                    ]:
                        val = match.get(match_key)
                        if val is not None:
                            current = existing.loc[idx, csv_col]
                            if current.isna().all() or (current == "").all():
                                existing.loc[idx, csv_col] = val

                matched += 1
            else:
                unmatched.append(
                    f"{date_str} {home_resolved} vs {away_resolved}"
                )

        if unmatched:
            console.print(
                f"    [yellow]Unmatched in {sc}: {len(unmatched)} matches[/yellow]"
            )
            for m in unmatched[:5]:
                console.print(f"      [dim]{m}[/dim]")

        existing.to_csv(csv_path, index=False, encoding="utf-8")
        results[sc] = matched
        console.print(f"  [green]{sc}:[/green] {matched} matches updated")

    return results


def run_sofascore_scrape(
    data_dir: str = "data/football_data",
    headless: bool = True,
    seasons: list[int] | None = None,
    delay: float = 2.0,
) -> None:
    """Scrape SofaScore match stats for Israeli leagues and merge into CSVs.

    Args:
        data_dir: Path to football data directory.
        headless: Run Chrome in headless mode.
        seasons: Optional list of start years to scrape. Defaults to 2008-2022.
        delay: Base delay between match stats API calls (seconds).
    """
    console.print("[bold blue]Toto AI - SofaScore Stats Scraper[/bold blue]")
    console.print()
    console.print(
        f"Target: ISR1 (Israeli Premier League) | "
        f"Delay: {delay}s | Headless: {headless}"
    )
    console.print()

    scraper = SofaScoreScraper(headless=headless, delay=delay)
    scraper.scrape_all(data_dir=data_dir, seasons=seasons)

    console.print()
    console.print("[bold green]SofaScore scraping complete![/bold green]")
