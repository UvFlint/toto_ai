from __future__ import annotations

import json
import re
import time
from datetime import datetime

from toto_ai.config import settings
from toto_ai.scraper.models import Match, WinnerForm
from toto_ai.console import console

WINNER_BASE_URL = "https://www.winner.co.il"
WINNER16_PATH = "/%D7%9E%D7%A9%D7%97%D7%A7%D7%99%D7%9D/%D7%95%D7%95%D7%99%D7%A0%D7%A8-16"
WINNER16_SUBMIT_PATH = "/%D7%9E%D7%A9%D7%97%D7%A7%D7%99%D7%9D/%D7%95%D7%95%D7%99%D7%A0%D7%A8-16/%D7%A8%D7%92%D7%99%D7%9C"


class WinnerScraper:
    """Scrape the current Winner 16 form from winner.co.il using Selenium."""

    async def fetch_current_form(self) -> WinnerForm:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By

        options = Options()
        if settings.HEADLESS:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--lang=he-IL")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-popup-blocking")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
        # Enable performance logging to capture network requests
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        console.print("[dim]Launching Chrome...[/dim]")
        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )

        try:
            console.print("[dim]Navigating to winner.co.il...[/dim]")
            driver.get(WINNER_BASE_URL + WINNER16_PATH)

            # Wait for SPA content to load
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            "[class*='game'], [class*='match'], [class*='row'], table",
                        )
                    )
                )
                time.sleep(3)
            except Exception:
                console.print(
                    "[yellow]Timed out waiting for content selectors, using sleep fallback...[/yellow]"
                )
                time.sleep(12)

            # Try to extract data from captured network responses first
            form = self._extract_from_network_logs(driver)
            if form and form.matches:
                console.print(
                    f"[green]Found {len(form.matches)} matches from API responses[/green]"
                )
                self._enrich_sportradar_urls(driver, form)
                return form

            # Fallback: extract from rendered DOM
            console.print("[dim]Trying DOM extraction...[/dim]")
            form = self._extract_from_dom(driver)

            if form.matches:
                console.print(f"[green]Found {len(form.matches)} matches from DOM[/green]")
                self._enrich_sportradar_urls(driver, form)
            else:
                console.print(
                    "[yellow]No matches found. The site structure may have changed.[/yellow]"
                )
                self._dump_debug_info(driver)

            return form
        finally:
            driver.quit()

    def _enrich_sportradar_urls(self, driver: object, form: WinnerForm) -> None:
        """Extract Sportradar widget URLs from bet-radar-N buttons and attach to matches.

        Detects new browser tabs opened by each button click and captures the URL,
        which works regardless of how the React app triggers the navigation.
        """
        import time

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        try:
            main_handle = driver.current_window_handle  # type: ignore[attr-defined]
        except Exception:
            return

        sportradar_urls: dict[int, str] = {}

        for i in range(16):
            btn_id = f"bet-radar-{i}"
            try:
                btn = driver.find_element(By.ID, btn_id)  # type: ignore[attr-defined]
                driver.execute_script("arguments[0].scrollIntoView(true);", btn)  # type: ignore[attr-defined]
                time.sleep(0.3)

                handles_before = set(driver.window_handles)  # type: ignore[attr-defined]
                driver.execute_script("arguments[0].click();", btn)  # type: ignore[attr-defined]

                # Poll for a new tab (up to 4 seconds)
                new_handle = None
                deadline = time.time() + 4
                while time.time() < deadline:
                    new_handles = set(driver.window_handles) - handles_before  # type: ignore[attr-defined]
                    if new_handles:
                        new_handle = new_handles.pop()
                        break
                    time.sleep(0.1)

                if new_handle:
                    driver.switch_to.window(new_handle)  # type: ignore[attr-defined]
                    try:
                        WebDriverWait(driver, 5).until(EC.url_contains("sportradar"))
                    except Exception:
                        pass
                    url = driver.current_url  # type: ignore[attr-defined]
                    driver.close()  # type: ignore[attr-defined]
                    driver.switch_to.window(main_handle)  # type: ignore[attr-defined]

                    if url and "sportradar" in url:
                        sportradar_urls[i] = url
                        console.print(f"[dim]bet-radar-{i}: {url[:80]}[/dim]")
            except Exception:
                # Ensure we stay on the main tab
                try:
                    if main_handle in driver.window_handles:  # type: ignore[attr-defined]
                        driver.switch_to.window(main_handle)  # type: ignore[attr-defined]
                except Exception:
                    pass
                continue

        if not sportradar_urls:
            console.print("[dim]No sportradar URLs found via button clicks.[/dim]")
            return

        # Attach URLs to matches by index
        for match in form.matches:
            idx = match.match_number - 1
            if idx in sportradar_urls:
                match.sportradar_url = sportradar_urls[idx]

        console.print(f"[green]Enriched {len(sportradar_urls)} matches with Sportradar URLs[/green]")

    def _dump_debug_info(self, driver: object) -> None:
        """Save screenshot and page source for debugging when scraping fails."""
        try:
            driver.save_screenshot("/tmp/winner_debug.png")  # type: ignore[attr-defined]
            console.print("[yellow]Debug screenshot saved to /tmp/winner_debug.png[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Could not save screenshot: {e}[/yellow]")
        try:
            console.print(f"[yellow]Page title: {driver.title}[/yellow]")  # type: ignore[attr-defined]
            console.print(f"[yellow]Current URL: {driver.current_url}[/yellow]")  # type: ignore[attr-defined]
            source = driver.page_source[:2000]  # type: ignore[attr-defined]
            console.print(f"[yellow]Page source (first 2000 chars):[/yellow]\n{source}")
        except Exception as e:
            console.print(f"[yellow]Could not dump page source: {e}[/yellow]")

    def _extract_from_network_logs(self, driver: object) -> WinnerForm | None:
        """Extract match data from captured network performance logs."""
        try:
            logs = driver.get_log("performance")  # type: ignore[attr-defined]
        except Exception:
            return None

        interesting_patterns = [
            "/api/",
            "/games",
            "/events",
            "/toto",
            "/winner",
            "/matches",
            "/fixtures",
            "/program",
        ]

        captured_data: list[tuple[str, object]] = []

        for entry in logs:
            try:
                msg = json.loads(entry["message"])["message"]
                if msg["method"] != "Network.responseReceived":
                    continue

                url = msg["params"]["response"]["url"]
                if not any(p in url.lower() for p in interesting_patterns):
                    continue

                # Get the response body via CDP
                request_id = msg["params"]["requestId"]
                try:
                    body = driver.execute_cdp_cmd(  # type: ignore[attr-defined]
                        "Network.getResponseBody", {"requestId": request_id}
                    )
                except Exception:
                    continue

                data = json.loads(body["body"])
                console.print(f"[dim]Captured API response: {url[:80]}...[/dim]")
                captured_data.append((url, data))

            except Exception:
                continue

        # Try each captured response
        for url, data in captured_data:
            # Debug: dump structure for Toto-related endpoints
            if "toto" in url.lower() or "draw" in url.lower():
                self._debug_dump_structure(data, url)

            # Try Winner-specific API format (GetTotoDraws, GetNWData)
            form = self._try_parse_winner_api(data, url)
            if form and form.matches:
                return form

            # Try generic format
            matches = self._try_extract_matches_from_json(data)
            if matches and len(matches) >= 16:
                return WinnerForm(matches=matches[:16])

        # If nothing matched, dump ALL captured API structures for debugging
        if captured_data:
            console.print("[yellow]No matches found in API responses. Dumping structures:[/yellow]")
            for url, data in captured_data:
                self._debug_dump_structure(data, url)

        return None

    def _debug_dump_structure(self, data: object, url: str) -> None:
        """Print the structure of a JSON response for debugging."""
        console.print(f"[dim]--- Debug: {url[:80]} ---[/dim]")
        if isinstance(data, dict):
            for key, value in data.items():
                val_type = type(value).__name__
                if isinstance(value, list):
                    console.print(
                        f"[dim]  {key}: list[{len(value)}] "
                        f"(first: {type(value[0]).__name__ if value else 'empty'})[/dim]"
                    )
                    if value and isinstance(value[0], dict):
                        console.print(
                            f"[dim]    first item keys: {list(value[0].keys())[:15]}[/dim]"
                        )
                        # Show first item values (truncated)
                        for k, v in list(value[0].items())[:10]:
                            v_str = str(v)[:80] if v is not None else "None"
                            console.print(f"[dim]      {k}: {v_str}[/dim]")
                elif isinstance(value, dict):
                    console.print(f"[dim]  {key}: dict with keys: {list(value.keys())[:10]}[/dim]")
                else:
                    v_str = str(value)[:80]
                    console.print(f"[dim]  {key}: {val_type} = {v_str}[/dim]")
        elif isinstance(data, list):
            console.print(f"[dim]  Root: list[{len(data)}][/dim]")
            if data and isinstance(data[0], dict):
                console.print(f"[dim]  first item keys: {list(data[0].keys())[:15]}[/dim]")
        console.print("[dim]--- End debug ---[/dim]")

    def _try_parse_winner_api(self, data: object, url: str) -> WinnerForm | None:
        """Parse Winner.co.il specific API response formats."""
        if not isinstance(data, dict):
            return None

        # Handle GetTotoDraws response: { games: [{ drawNumber, rows: [...] }] }
        if "games" in data and isinstance(data["games"], list):
            for game in data["games"]:
                if not isinstance(game, dict):
                    continue
                # Look for Winner 16 (gameType 88) or any game with 16 rows
                rows = game.get("rows", [])
                if not isinstance(rows, list):
                    continue

                draw_number = game.get("drawNumber")
                draw_id = game.get("drawId")
                close_dt = game.get("closeDateTime")
                form_number = str(draw_number) if draw_number else None

                matches: list[Match] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    team_a = row.get("teamA", "")
                    team_b = row.get("teamB", "")
                    if not team_a or not team_b:
                        continue

                    match_date = None
                    est = row.get("eventStartTime")
                    if est:
                        try:
                            match_date = datetime.fromisoformat(str(est).replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            pass

                    matches.append(
                        Match(
                            match_number=row.get("rowNumber", len(matches) + 1),
                            home_team=team_a,
                            away_team=team_b,
                            league=row.get("league", ""),
                            match_date=match_date,
                        )
                    )

                if len(matches) >= 16:
                    deadline = None
                    if close_dt:
                        try:
                            deadline = datetime.fromisoformat(str(close_dt).replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            pass
                    console.print(
                        f"[green]Found Winner 16 draw #{form_number} "
                        f"with {len(matches)} matches[/green]"
                    )
                    return WinnerForm(
                        form_number=form_number,
                        deadline=deadline,
                        matches=matches[:16],
                    )

        return None

    def _deep_search_matches(self, data: object, depth: int = 0) -> WinnerForm | None:
        """Recursively search JSON for arrays of match-like objects."""
        if depth > 5:
            return None

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and len(value) >= 8:
                    # Check if items look like matches (have team-related fields)
                    matches = self._try_extract_matches_from_json(value)
                    if matches and len(matches) >= 8:
                        console.print(f"[dim]Found {len(matches)} matches under key '{key}'[/dim]")
                        return WinnerForm(matches=matches[:16])
                elif isinstance(value, (dict, list)):
                    result = self._deep_search_matches(value, depth + 1)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    result = self._deep_search_matches(item, depth + 1)
                    if result:
                        return result

        return None

    def _try_extract_matches_from_json(self, data: object) -> list[Match]:
        """Attempt to extract match data from various JSON structures."""
        matches: list[Match] = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                match = self._try_parse_match_item(item, i + 1)
                if match:
                    matches.append(match)
        elif isinstance(data, dict):
            # Try all list values in the dict
            for key, value in data.items():
                if isinstance(value, list) and len(value) >= 16:
                    for i, item in enumerate(value):
                        match = self._try_parse_match_item(item, i + 1)
                        if match:
                            matches.append(match)
                    if len(matches) >= 16:
                        break
                    matches.clear()

        return matches

    def _try_parse_match_item(self, item: object, number: int) -> Match | None:
        """Try to parse a single match from a JSON item."""
        if not isinstance(item, dict):
            return None

        home_keys = [
            "homeTeam",
            "home_team",
            "home",
            "team1",
            "HomeTeam",
            "homeName",
            "HomeTeamName",
            "HomeName",
            "Home",
            "Team1",
            "team_home",
            "homeTeamName",
            "homeClub",
            "HomeClub",
        ]
        away_keys = [
            "awayTeam",
            "away_team",
            "away",
            "team2",
            "AwayTeam",
            "awayName",
            "AwayTeamName",
            "AwayName",
            "Away",
            "Team2",
            "team_away",
            "awayTeamName",
            "awayClub",
            "AwayClub",
        ]
        league_keys = [
            "league",
            "competition",
            "leagueName",
            "Liga",
            "League",
            "LeagueName",
            "CompetitionName",
            "TournamentName",
        ]
        date_keys = [
            "date",
            "matchDate",
            "startDate",
            "kickoff",
            "DateTime",
            "Date",
            "MatchDate",
            "StartDate",
            "GameDate",
            "EventDate",
        ]

        home = self._extract_team_name(item, home_keys)
        away = self._extract_team_name(item, away_keys)

        if not home or not away:
            return None

        league = ""
        for k in league_keys:
            if k in item:
                val = item[k]
                if isinstance(val, str):
                    league = val
                elif isinstance(val, dict):
                    league = str(val.get("name", ""))
                if league:
                    break

        match_date = None
        for k in date_keys:
            if k in item and item[k]:
                try:
                    match_date = datetime.fromisoformat(str(item[k]).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

        return Match(
            match_number=number,
            home_team=home,
            away_team=away,
            league=league,
            match_date=match_date,
        )

    def _extract_team_name(self, item: dict, keys: list[str]) -> str:
        """Extract team name from a dict using multiple possible key names."""
        for k in keys:
            if k in item:
                val = item[k]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    for name_key in ["name", "Name", "displayName", "shortName"]:
                        if name_key in val:
                            return str(val[name_key])
        return ""

    def _extract_from_dom(self, driver: object) -> WinnerForm:
        """Extract match data directly from the rendered DOM."""
        from selenium.webdriver.common.by import By

        matches: list[Match] = []

        selectors_to_try = [
            "table tr",
            "[class*='game']",
            "[class*='match']",
            "[class*='event']",
            "[class*='fixture']",
            "[class*='row']",
            "[data-game]",
            "[data-match]",
            "[data-event]",
        ]

        for selector in selectors_to_try:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)  # type: ignore[attr-defined]
            except Exception:
                continue

            if len(elements) >= 16:
                console.print(
                    f"[dim]Found {len(elements)} elements with selector: {selector}[/dim]"
                )
                for i, el in enumerate(elements[:16]):
                    text = el.text
                    match = self._parse_match_from_text(text, i + 1)
                    if match:
                        matches.append(match)
                if len(matches) >= 16:
                    break
                matches.clear()

        if not matches:
            console.print("[dim]Trying full page text extraction...[/dim]")
            try:
                page_text = driver.find_element(By.TAG_NAME, "body").text  # type: ignore[attr-defined]
                matches = self._parse_matches_from_full_text(page_text)
            except Exception:
                pass

        return WinnerForm(matches=matches)

    def _parse_match_from_text(self, text: str, number: int) -> Match | None:
        """Parse a match from a text block (e.g., a table row)."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) < 2:
            return None

        hebrew_pattern = re.compile(r"[\u0590-\u05FF\s]+")
        team_names = [line for line in lines if hebrew_pattern.search(line) and len(line) > 2]

        if len(team_names) >= 2:
            return Match(
                match_number=number,
                home_team=team_names[0].strip(),
                away_team=team_names[1].strip(),
            )
        return None

    def _parse_matches_from_full_text(self, text: str) -> list[Match]:
        """Last resort: try to extract matches from the full page text."""
        matches: list[Match] = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        hebrew_pattern = re.compile(r"^[\u0590-\u05FF\s\.]+$")
        team_candidates: list[str] = []

        for line in lines:
            if hebrew_pattern.match(line) and 3 < len(line) < 50:
                team_candidates.append(line)

        for i in range(0, len(team_candidates) - 1, 2):
            if len(matches) >= 16:
                break
            matches.append(
                Match(
                    match_number=len(matches) + 1,
                    home_team=team_candidates[i],
                    away_team=team_candidates[i + 1],
                )
            )

        return matches

    async def fetch_results_form(self, url: str) -> WinnerForm:
        """Scrape a past results form from winner.co.il given a results URL.

        Navigates to the URL, captures API responses, and extracts match results.
        The results page is expected to contain scores for completed matches.
        """
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By

        options = Options()
        if settings.HEADLESS:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--lang=he-IL")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        console.print("[dim]Launching Chrome for results page...[/dim]")
        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )

        try:
            console.print(f"[dim]Navigating to: {url}[/dim]")
            driver.get(url)

            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # First load — SPA shell loads but matches often don't appear until refresh
            console.print("[dim]Waiting for initial page load...[/dim]")
            time.sleep(8)

            # Refresh to trigger the actual data load
            console.print("[dim]Refreshing page to load match data...[/dim]")
            driver.refresh()

            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "li.toto-box"))
                )
                time.sleep(3)
            except Exception:
                console.print(
                    "[yellow]Timed out waiting for toto-box elements, using sleep fallback...[/yellow]"
                )
                time.sleep(15)

            # Try network log extraction first
            form = self._extract_results_from_network_logs(driver)
            if form and form.matches:
                results_count = sum(1 for m in form.matches if m.result)
                console.print(
                    f"[green]Found {len(form.matches)} matches "
                    f"({results_count} with results) from API responses[/green]"
                )
                return form

            # Fallback: extract from rendered DOM
            console.print("[dim]Trying DOM extraction for results...[/dim]")
            form = self._extract_results_from_dom(driver)
            if form and form.matches:
                results_count = sum(1 for m in form.matches if m.result)
                console.print(
                    f"[green]Found {len(form.matches)} matches "
                    f"({results_count} with results) from DOM[/green]"
                )
                return form

            console.print(
                "[yellow]Could not extract results from DOM either. Dumping debug info...[/yellow]"
            )
            self._dump_debug_info(driver)
            return WinnerForm()
        finally:
            driver.quit()

    def _extract_results_from_dom(self, driver: object) -> WinnerForm | None:
        """Extract match results from the rendered DOM of the results page.

        Expected structure: li.toto-box elements each containing a .toto-line div
        with an aria-label like "באירוע 1 ... התוצאה היא 2" and a .yellow button
        indicating the result.
        """
        from selenium.webdriver.common.by import By

        try:
            boxes = driver.find_elements(By.CSS_SELECTOR, "li.toto-box")  # type: ignore[attr-defined]
        except Exception:
            return None

        if not boxes:
            return None

        # Extract form number from draw-info
        form_number = None
        try:
            spans = driver.find_elements(By.CSS_SELECTOR, ".draw-info span")  # type: ignore[attr-defined]
            for span in spans:
                text = span.text.strip()
                if "מחזור" in text:
                    form_number = text.replace("מחזור", "").strip()
                    break
        except Exception:
            pass

        # Regex for aria-label: "באירוע 1 מכבי נתניה נגד בית"ר ירושלים התוצאה היא 2"
        aria_pattern = re.compile(
            r"באירוע\s+(\d+)\s+(.+?)\s+נגד\s+(.+?)\s+התוצאה היא\s+(1|X|2)",
            re.IGNORECASE,
        )

        matches: list[Match] = []
        for box in boxes:
            try:
                toto_line = box.find_element(By.CSS_SELECTOR, ".toto-line")
                aria_label = toto_line.get_attribute("aria-label") or ""

                m = aria_pattern.search(aria_label)
                if m:
                    match_num = int(m.group(1))
                    home = m.group(2).strip()
                    away = m.group(3).strip()
                    result = m.group(4).upper()
                else:
                    # Fallback: parse from child elements
                    texts_el = toto_line.find_element(By.CSS_SELECTOR, ".toto-texts")
                    p_tags = texts_el.find_elements(By.TAG_NAME, "p")
                    # First <p> is the match number
                    match_num = int(p_tags[0].text.strip()) if p_tags else len(matches) + 1

                    # Team names are in the nested <div>'s <p> children
                    inner_div = texts_el.find_element(By.TAG_NAME, "div")
                    team_ps = inner_div.find_elements(By.TAG_NAME, "p")
                    # Filter out the separator " - "
                    team_names = [
                        p.text.strip() for p in team_ps if p.text.strip() not in ("", "-", " - ")
                    ]
                    home = team_names[0] if len(team_names) > 0 else "?"
                    away = team_names[1] if len(team_names) > 1 else "?"

                    # Result from the yellow button
                    result = None
                    try:
                        yellow = toto_line.find_element(
                            By.CSS_SELECTOR, ".toto-result-btns .yellow"
                        )
                        result_text = yellow.find_element(By.TAG_NAME, "span").text.strip()
                        if result_text in ("1", "X", "2"):
                            result = result_text
                    except Exception:
                        pass

                matches.append(
                    Match(
                        match_number=match_num,
                        home_team=home,
                        away_team=away,
                        result=result,
                    )
                )
            except Exception as e:
                console.print(f"[yellow]Failed to parse toto-box: {e}[/yellow]")
                continue

        if matches:
            return WinnerForm(form_number=form_number, matches=matches)
        return None

    def _extract_results_from_network_logs(self, driver: object) -> WinnerForm | None:
        """Extract match results from captured network performance logs."""
        try:
            logs = driver.get_log("performance")  # type: ignore[attr-defined]
        except Exception:
            return None

        interesting_patterns = [
            "/api/",
            "/games",
            "/events",
            "/toto",
            "/winner",
            "/matches",
            "/fixtures",
            "/program",
            "/results",
            "/draws",
        ]

        captured_data: list[tuple[str, object]] = []

        for entry in logs:
            try:
                msg = json.loads(entry["message"])["message"]
                if msg["method"] != "Network.responseReceived":
                    continue

                url = msg["params"]["response"]["url"]
                if not any(p in url.lower() for p in interesting_patterns):
                    continue

                request_id = msg["params"]["requestId"]
                try:
                    body = driver.execute_cdp_cmd(  # type: ignore[attr-defined]
                        "Network.getResponseBody", {"requestId": request_id}
                    )
                except Exception:
                    continue

                data = json.loads(body["body"])
                console.print(f"[dim]Captured API response: {url[:100]}...[/dim]")
                captured_data.append((url, data))

            except Exception:
                continue

        # Try to parse results from each captured response
        for url, data in captured_data:
            self._debug_dump_structure(data, url)

            form = self._try_parse_winner_results_api(data, url)
            if form and form.matches:
                return form

        # Save all captured data to a debug file
        if captured_data:
            try:
                debug_path = "debug_results_dump.json"
                with open(debug_path, "w", encoding="utf-8") as f:
                    dump = []
                    for url, data in captured_data:
                        dump.append({"url": url, "data": data})
                    json.dump(dump, f, ensure_ascii=False, indent=2)
                console.print(
                    f"[yellow]Saved {len(captured_data)} API responses to {debug_path}[/yellow]"
                )
            except Exception as e:
                console.print(f"[yellow]Could not save debug dump: {e}[/yellow]")

        return None

    def _try_parse_winner_results_api(self, data: object, url: str) -> WinnerForm | None:
        """Parse Winner.co.il results API response, extracting match results."""
        if not isinstance(data, dict):
            return None

        if "games" not in data or not isinstance(data["games"], list):
            return None

        for game in data["games"]:
            if not isinstance(game, dict):
                continue

            rows = game.get("rows", [])
            if not isinstance(rows, list):
                continue

            draw_number = game.get("drawNumber")
            close_dt = game.get("closeDateTime")
            form_number = str(draw_number) if draw_number else None

            matches: list[Match] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                team_a = row.get("teamA", "")
                team_b = row.get("teamB", "")
                if not team_a or not team_b:
                    continue

                match_date = None
                est = row.get("eventStartTime")
                if est:
                    try:
                        match_date = datetime.fromisoformat(str(est).replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                # Try to extract the result from various possible fields
                result = self._extract_result_from_row(row)

                matches.append(
                    Match(
                        match_number=row.get("rowNumber", len(matches) + 1),
                        home_team=team_a,
                        away_team=team_b,
                        league=row.get("league", ""),
                        match_date=match_date,
                        result=result,
                    )
                )

            if len(matches) >= 16:
                deadline = None
                if close_dt:
                    try:
                        deadline = datetime.fromisoformat(str(close_dt).replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass
                results_count = sum(1 for m in matches if m.result)
                console.print(
                    f"[green]Parsed draw #{form_number}: "
                    f"{len(matches)} matches, {results_count} results[/green]"
                )
                return WinnerForm(
                    form_number=form_number,
                    deadline=deadline,
                    matches=matches[:16],
                )

        return None

    def _extract_result_from_row(self, row: dict) -> str | None:
        """Try to derive a 1/X/2 result from a match row's score fields."""
        # Try direct result field
        for key in ("result", "Result", "resultTypeId", "ResultTypeId", "sign", "Sign"):
            val = row.get(key)
            if val is not None:
                val_str = str(val).strip()
                if val_str in ("1", "X", "2", "x"):
                    return val_str.upper() if val_str == "x" else val_str
                # resultTypeId might be numeric: 1=home, 0=draw, 2=away
                if val_str == "0":
                    return "X"

        # Try score-based derivation
        score_key_pairs = [
            ("scoreA", "scoreB"),
            ("ScoreA", "ScoreB"),
            ("homeScore", "awayScore"),
            ("HomeScore", "AwayScore"),
            ("homeGoals", "awayGoals"),
            ("HomeGoals", "AwayGoals"),
            ("goalA", "goalB"),
            ("GoalA", "GoalB"),
            ("resultA", "resultB"),
            ("ResultA", "ResultB"),
        ]
        for home_key, away_key in score_key_pairs:
            home_val = row.get(home_key)
            away_val = row.get(away_key)
            if home_val is not None and away_val is not None:
                try:
                    home_goals = int(home_val)
                    away_goals = int(away_val)
                    if home_goals > away_goals:
                        return "1"
                    elif home_goals == away_goals:
                        return "X"
                    else:
                        return "2"
                except (ValueError, TypeError):
                    continue

        return None


def create_mock_form() -> WinnerForm:
    """Create a mock Winner 16 form for testing without scraping."""
    return WinnerForm(
        form_number="mock-001",
        matches=[
            Match(
                match_number=1,
                home_team="מכבי תל אביב",
                away_team="הפועל באר שבע",
                league="ליגת העל",
                country="ישראל",
            ),
            Match(
                match_number=2,
                home_team="הפועל תל אביב",
                away_team="מכבי חיפה",
                league="ליגת העל",
                country="ישראל",
            ),
            Match(
                match_number=3,
                home_team="ברצלונה",
                away_team="ריאל מדריד",
                league="לה ליגה",
                country="ספרד",
            ),
            Match(
                match_number=4,
                home_team="ליברפול",
                away_team="מנצ'סטר סיטי",
                league="פרמיירליג",
                country="אנגליה",
            ),
            Match(
                match_number=5,
                home_team="באיירן מינכן",
                away_team="דורטמונד",
                league="בונדסליגה",
                country="גרמניה",
            ),
            Match(
                match_number=6,
                home_team="יובנטוס",
                away_team="אינטר מילאן",
                league="סריה א",
                country="איטליה",
            ),
            Match(
                match_number=7,
                home_team="פריז סן ז'רמן",
                away_team="מרסיי",
                league="ליג 1",
                country="צרפת",
            ),
            Match(
                match_number=8,
                home_team="אייאקס",
                away_team="פיינורד",
                league="ארדיוויזיה",
                country="הולנד",
            ),
            Match(
                match_number=9,
                home_team="בנפיקה",
                away_team="פורטו",
                league="ליגה פורטוגזית",
                country="פורטוגל",
            ),
            Match(
                match_number=10,
                home_team="סלטיק",
                away_team="ריינג'רס",
                league="ליגה סקוטית",
                country="סקוטלנד",
            ),
            Match(
                match_number=11,
                home_team="ארסנל",
                away_team="צ'לסי",
                league="פרמיירליג",
                country="אנגליה",
            ),
            Match(
                match_number=12,
                home_team="אתלטיקו מדריד",
                away_team="סביליה",
                league="לה ליגה",
                country="ספרד",
            ),
            Match(
                match_number=13,
                home_team="מילאן",
                away_team="נאפולי",
                league="סריה א",
                country="איטליה",
            ),
            Match(
                match_number=14,
                home_team="לייפציג",
                away_team="לברקוזן",
                league="בונדסליגה",
                country="גרמניה",
            ),
            Match(
                match_number=15,
                home_team="טוטנהאם",
                away_team="מנצ'סטר יונייטד",
                league="פרמיירליג",
                country="אנגליה",
            ),
            Match(
                match_number=16,
                home_team="ביתר ירושלים",
                away_team="מכבי נתניה",
                league="ליגת העל",
                country="ישראל",
            ),
        ],
    )
