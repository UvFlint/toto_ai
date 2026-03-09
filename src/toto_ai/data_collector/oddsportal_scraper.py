"""Scrape historical 1X2 odds from OddsPortal for Israeli leagues.

Fetches match results and closing odds from OddsPortal using Selenium,
then merges them into existing CSV files or creates new ones.
"""

from __future__ import annotations

import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from toto_ai.console import console

BASE_URL = "https://www.oddsportal.com/football/israel"

LEAGUE_SLUGS: dict[str, str] = {
    "ISR1": "ligat-ha-al",
}

# OddsPortal team name → API-Football team name (built incrementally)
TEAM_ALIASES: dict[str, str] = {
    "H. Beer Sheva": "Hapoel Beer Sheva",
    "Kiryat Shmona": "Ironi Kiryat Shmona",
    "SC Ashdod": "Ashdod",
    "Sakhnin": "Bnei Sakhnin",
    "Netanya": "Maccabi Netanya",
    "Hapoel Jerusalem": "Hapoel Katamon",
    "Ness Ziona": "Sektzia Nes Tziona",
    "Sektzia Ness Ziona": "Sektzia Nes Tziona",
    "H. Tel Aviv": "Hapoel Tel Aviv",
    "M. Petach Tikva": "Maccabi Petah Tikva",
    "H. Kiryat Shmona": "Ironi Kiryat Shmona",
    "M. Bnei Reineh": "Maccabi Bnei Raina",
    "H. Kfar Saba": "Hapoel Kfar Saba",
    "H. Petach Tikva": "Hapoel Petah Tikva",
    "Hapoel Petah Tikva": "Hapoel Petah Tikva",
    "Hapoel Kfar Saba": "Hapoel Kfar Saba",
    "Hapoel Beer Sheva": "Hapoel Beer Sheva",
    "H. Raanana": "Hapoel Ra'anana",
    "Bnei Yehuda": "Bnei Yehuda",
    "H. Nof HaGalil": "Hapoel Nof HaGalil",
    "H. Akko": "H. Akko",
    "Hap. Ramat Gan": "Hap. Ramat Gan",
    "Hapoel Akko": "H. Akko",
    "Hapoel Ramat Gan": "Hap. Ramat Gan",
    "Hapoel Ashkelon": "Ashkelon",
}

# football-data.co.uk CSV columns
CSV_COLUMNS = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG",
    "HS", "AS", "HST", "AST", "HC", "AC",
    "HF", "AF", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
]


def _season_code(year: int) -> str:
    """Convert start year to season code (e.g. 2024 → '2425')."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _result_from_goals(home: int, away: int) -> str:
    if home > away:
        return "H"
    elif away > home:
        return "A"
    return "D"


def _normalize_team(name: str) -> str:
    """Normalize team name for matching."""
    return TEAM_ALIASES.get(name, name)


def _fuzzy_team_match(name: str, csv_teams: set[str]) -> str | None:
    """Try to fuzzy-match an OddsPortal team name against CSV team names.

    Handles cases where API-Football uses different name variants across seasons
    (e.g. "Bnei Yehuda" vs "Bnei Yehuda Tel Aviv", "Hapoel Nof HaGalil" vs
    "Hapoel Nazareth Illit").
    """
    # Exact match already handled by caller
    if name in csv_teams:
        return name

    name_lower = name.lower()

    # Try substring matching in both directions
    for csv_name in csv_teams:
        csv_lower = csv_name.lower()
        if name_lower in csv_lower or csv_lower in name_lower:
            return csv_name

    # Known city renames / equivalences
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


def _parse_date(date_str: str) -> str:
    """Parse OddsPortal date like '24 May 2025 - Championship Group' to DD/MM/YYYY."""
    # Strip everything after the year (group/round info)
    clean = re.sub(r"\s*-\s*.*$", "", date_str).strip()
    try:
        dt = datetime.strptime(clean, "%d %b %Y")
        return dt.strftime("%d/%m/%Y")
    except ValueError:
        return ""


class OddsPortalScraper:
    """Scrape historical 1X2 odds from OddsPortal for Israeli leagues."""

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless

    def _create_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        if self._headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )

        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        return driver

    def _dismiss_cookie_banner(self, driver) -> None:
        """Click 'I Accept' on the cookie consent banner if present."""
        from selenium.webdriver.common.by import By

        try:
            buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'I Accept')]")
            if buttons:
                buttons[0].click()
                time.sleep(1)
        except Exception:
            pass

    def _scroll_to_load(self, driver) -> None:
        """Scroll down repeatedly to trigger lazy loading of all match rows."""
        for _ in range(15):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.8)

    def _parse_page(self, driver) -> tuple[list[dict], str]:
        """Parse all match rows on the current page.

        Returns (matches, current_date) where current_date is carried across pages.
        """
        from selenium.webdriver.common.by import By

        rows = driver.find_elements(By.CSS_SELECTOR, "div.eventRow")
        matches: list[dict] = []
        current_date = ""

        for row in rows:
            # Check for date header
            date_headers = row.find_elements(By.CSS_SELECTOR, '[data-testid="date-header"]')
            if date_headers:
                raw_date = date_headers[0].text.strip()
                parsed = _parse_date(raw_date)
                if parsed:
                    current_date = parsed

            # Get team names
            team_elems = row.find_elements(By.CSS_SELECTOR, "p.participant-name")
            if len(team_elems) < 2:
                continue
            home_team = team_elems[0].text.strip()
            away_team = team_elems[1].text.strip()

            if not home_team or not away_team:
                continue

            # Get odds (1X2)
            odds_elems = row.find_elements(By.CSS_SELECTOR, 'p[class*="height-content"]')
            odds_values = [o.text.strip() for o in odds_elems]
            if len(odds_values) < 3:
                continue

            try:
                home_odds = float(odds_values[0])
                draw_odds = float(odds_values[1])
                away_odds = float(odds_values[2])
            except (ValueError, IndexError):
                continue

            # Get score from the match link text
            # Format: "HH:MM\nHomeTeam\nhome_goals\n–\naway_goals\nAwayTeam"
            match_links = row.find_elements(
                By.CSS_SELECTOR, 'a[href*="/football/israel/"][href$="/"]'
            )
            home_goals = away_goals = None
            for ml in match_links:
                href = ml.get_attribute("href") or ""
                # Skip league/tournament links
                if href.endswith("results/") or href.endswith("standings/"):
                    continue
                if "ligat-ha-al-" not in href and "liga-leumit-" not in href:
                    continue
                # This should be a match link
                text_parts = ml.text.strip().split("\n")
                # Find two consecutive digits separated by a dash-like char
                for j, part in enumerate(text_parts):
                    if part.strip().isdigit() and j + 2 < len(text_parts):
                        next_part = text_parts[j + 2].strip()
                        if next_part.isdigit():
                            home_goals = int(part.strip())
                            away_goals = int(next_part.strip())
                            break
                if home_goals is not None:
                    break

            matches.append({
                "date": current_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
            })

        return matches, current_date

    def _scrape_season_with_driver(self, driver, div_code: str, start_year: int) -> list[dict]:
        """Scrape all match odds for one season using an existing driver.

        Args:
            driver: Selenium WebDriver instance.
            div_code: Division code (e.g. ISR1).
            start_year: Start year of the season (e.g. 2024 for 2024/25).

        Returns:
            List of match dicts with date, teams, scores, and odds.
        """
        from selenium.webdriver.common.by import By

        slug = LEAGUE_SLUGS[div_code]
        end_year = start_year + 1
        url = f"{BASE_URL}/{slug}-{start_year}-{end_year}/results/"

        all_matches: list[dict] = []

        try:
            console.print(f"  [dim]Scraping {div_code} {start_year}/{end_year}...[/dim]")
            driver.get(url)
            time.sleep(5 + random.uniform(1, 3))

            # Dismiss cookie banner on first page load
            self._dismiss_cookie_banner(driver)

            page = 1
            while True:
                self._scroll_to_load(driver)
                time.sleep(1)

                matches, _ = self._parse_page(driver)
                if not matches:
                    break

                all_matches.extend(matches)
                console.print(
                    f"    Page {page}: {len(matches)} matches "
                    f"(total: {len(all_matches)})"
                )

                # Find the next page number to click
                next_page = page + 1
                pag_links = driver.find_elements(By.CSS_SELECTOR, "a.pagination-link")
                target_link = None
                for pl in pag_links:
                    if pl.text.strip() == str(next_page):
                        target_link = pl
                        break

                if not target_link:
                    # Also try the "Next" link as fallback
                    for pl in pag_links:
                        if pl.text.strip() == "Next":
                            target_link = pl
                            break

                if not target_link:
                    break

                # Scroll up and click next page
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(0.5)
                try:
                    driver.execute_script("arguments[0].click();", target_link)
                except Exception:
                    break

                page += 1
                time.sleep(3 + random.uniform(1, 2))

        except Exception as e:
            console.print(f"  [red]Error scraping {div_code} {start_year}: {e}[/red]")

        console.print(
            f"  [green]{div_code} {_season_code(start_year)}:[/green] "
            f"{len(all_matches)} matches scraped"
        )
        return all_matches

    def scrape_season(self, div_code: str, start_year: int) -> list[dict]:
        """Scrape all match odds for one season (creates and destroys a driver)."""
        driver = self._create_driver()
        try:
            return self._scrape_season_with_driver(driver, div_code, start_year)
        finally:
            driver.quit()

    def scrape_all(
        self,
        div_code: str,
        seasons: list[int],
    ) -> pd.DataFrame:
        """Scrape multiple seasons for a league using a single browser session.

        Returns a DataFrame with all scraped matches.
        """
        all_rows: list[dict] = []
        driver = self._create_driver()

        try:
            # Dismiss cookie banner once at the start
            driver.get(f"{BASE_URL}/{LEAGUE_SLUGS[div_code]}/results/")
            time.sleep(4)
            self._dismiss_cookie_banner(driver)

            for start_year in seasons:
                matches = self._scrape_season_with_driver(driver, div_code, start_year)
                all_rows.extend(matches)
                # Delay between seasons
                if start_year != seasons[-1]:
                    time.sleep(2 + random.uniform(1, 3))
        finally:
            driver.quit()

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        return df


def merge_odds_to_csvs(
    odds_df: pd.DataFrame,
    data_dir: str | Path,
    div_code: str,
) -> dict[str, int]:
    """Merge scraped odds into existing CSV files or create new ones.

    For seasons where a CSV already exists (from API-Football), matches are
    matched by date + normalized team names and odds columns are updated.

    For seasons without an existing CSV, a new file is created with scores
    and odds (no match stats).

    Returns dict of {season_code: matches_updated_or_created}.
    """
    data_dir = Path(data_dir)
    div_dir = data_dir / div_code
    div_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}

    if odds_df.empty:
        return results

    # Group odds by season (inferred from date)
    odds_df = odds_df.copy()
    odds_df["_parsed_date"] = pd.to_datetime(odds_df["date"], format="%d/%m/%Y", errors="coerce")

    # Determine season from date: Aug-Dec = year/year+1, Jan-Jul = year-1/year
    def _date_to_season_year(dt):
        if pd.isna(dt):
            return None
        if dt.month >= 8:
            return dt.year
        return dt.year - 1

    odds_df["_season_year"] = odds_df["_parsed_date"].apply(_date_to_season_year)

    for season_year, season_odds in odds_df.groupby("_season_year"):
        if season_year is None:
            continue

        sc = _season_code(int(season_year))
        csv_path = div_dir / f"{sc}.csv"

        if csv_path.exists():
            # Merge into existing CSV
            existing = pd.read_csv(csv_path, encoding="utf-8")
            csv_teams = set(existing["HomeTeam"].unique()) | set(
                existing["AwayTeam"].unique()
            )
            matched = 0
            unmatched_teams: set[str] = set()

            # Build fuzzy name cache for this season
            fuzzy_cache: dict[str, str] = {}

            def _resolve_team(name: str) -> str:
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

            for _, odds_row in season_odds.iterrows():
                home_resolved = _resolve_team(odds_row["home_team"])
                away_resolved = _resolve_team(odds_row["away_team"])
                date_str = odds_row["date"]

                # Try exact match on date + resolved team names
                mask = (
                    (existing["Date"] == date_str)
                    & (existing["HomeTeam"] == home_resolved)
                    & (existing["AwayTeam"] == away_resolved)
                )

                if mask.any():
                    idx = existing.index[mask]
                    existing.loc[idx, "B365H"] = odds_row["home_odds"]
                    existing.loc[idx, "B365D"] = odds_row["draw_odds"]
                    existing.loc[idx, "B365A"] = odds_row["away_odds"]
                    matched += 1
                else:
                    unmatched_teams.add(home_resolved)
                    unmatched_teams.add(away_resolved)

            if unmatched_teams:
                missing = unmatched_teams - csv_teams
                if missing:
                    console.print(
                        f"    [yellow]Unmatched teams in {sc}: {sorted(missing)}[/yellow]"
                    )

            existing.to_csv(csv_path, index=False, encoding="utf-8")
            results[sc] = matched
            status = "merged"
        else:
            # Create new CSV with scores + odds
            rows: list[dict] = []
            for _, odds_row in season_odds.iterrows():
                hg = odds_row.get("home_goals")
                ag = odds_row.get("away_goals")
                if hg is None or ag is None or pd.isna(hg) or pd.isna(ag):
                    continue
                hg, ag = int(hg), int(ag)
                rows.append({
                    "Div": div_code,
                    "Date": odds_row["date"],
                    "HomeTeam": _normalize_team(odds_row["home_team"]),
                    "AwayTeam": _normalize_team(odds_row["away_team"]),
                    "FTHG": hg,
                    "FTAG": ag,
                    "FTR": _result_from_goals(hg, ag),
                    "B365H": odds_row["home_odds"],
                    "B365D": odds_row["draw_odds"],
                    "B365A": odds_row["away_odds"],
                })

            if rows:
                new_df = pd.DataFrame(rows)
                out_cols = [c for c in CSV_COLUMNS if c in new_df.columns]
                new_df[out_cols].to_csv(csv_path, index=False, encoding="utf-8")
                results[sc] = len(rows)
                status = "created"
            else:
                results[sc] = 0
                status = "empty"

        console.print(f"  [green]{div_code} {sc}:[/green] {results[sc]} matches ({status})")

    return results


def run_odds_scrape(
    data_dir: str = "data/football_data",
    headless: bool = True,
    seasons: list[int] | None = None,
) -> None:
    """Scrape OddsPortal odds for Israeli leagues and merge into CSVs.

    ISR1 (Ligat Ha'al): Available from 2005/06 through 2024/25 on OddsPortal.

    Args:
        data_dir: Path to football data directory.
        headless: Run Chrome in headless mode.
        seasons: Optional list of start years to scrape. Defaults to all (2005-2024).
    """
    scraper = OddsPortalScraper(headless=headless)

    # Only ISR1 has odds on OddsPortal
    div_code = "ISR1"
    if seasons is None:
        seasons = list(range(2005, 2025))  # 2005/06 through 2024/25
    console.print(f"[bold]Scraping {div_code} odds from OddsPortal ({len(seasons)} seasons)...[/bold]")
    console.print()

    odds_df = scraper.scrape_all(div_code, seasons)

    if odds_df.empty:
        console.print(f"[yellow]No odds data scraped for {div_code}[/yellow]")
        return

    # Save raw scraped data for future re-merges
    raw_path = Path(data_dir) / div_code / "_odds_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        existing_raw = pd.read_csv(raw_path, encoding="utf-8")
        odds_df = pd.concat([existing_raw, odds_df]).drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="last"
        )
    odds_df.to_csv(raw_path, index=False, encoding="utf-8")

    console.print(f"\n[bold]Merging {len(odds_df)} matches into CSVs...[/bold]")
    results = merge_odds_to_csvs(odds_df, data_dir, div_code)

    total = sum(results.values())
    console.print(f"\n[bold green]Done![/bold green] {total} matches across {len(results)} seasons")
