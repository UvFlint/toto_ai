"""Sportradar widget exploration script.

Usage: uv run python scripts/explore_sportradar.py

This script:
1. Opens winner.co.il Winner16 page with Selenium
2. Clicks each bet-radar-N button and captures the new tab URL (window-handle approach)
3. Navigates to each Sportradar URL and extracts window.__INITIAL_STATE__
4. Dumps raw data to data/sportradar_exploration/match_N.json
5. Tests direct API calls to stats.fn.sportradar.com (no Selenium)
6. Prints a coverage report comparing Sportradar data vs API-Football needs
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows to avoid codec errors with special chars
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

WINNER_BASE_URL = "https://www.winner.co.il"
WINNER16_PATH = "/%D7%9E%D7%A9%D7%97%D7%A7%D7%99%D7%9D/%D7%95%D7%95%D7%99%D7%A0%D7%A8-16"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sportradar_exploration"


def _chrome_driver(headless: bool = False):
    """Create a Chrome WebDriver with anti-bot settings."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--lang=he-IL")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def extract_sportradar_urls(driver) -> dict[int, str]:
    """Navigate to Winner16 and capture Sportradar URLs by detecting new tabs after button clicks."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    print("Navigating to winner.co.il Winner16...")
    driver.get(WINNER_BASE_URL + WINNER16_PATH)

    # Wait for SPA content
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[id^='bet-radar-']"))
        )
        time.sleep(2)
    except Exception:
        print("Timeout waiting for bet-radar buttons, trying fallback sleep...")
        time.sleep(15)

    # Check how many bet-radar buttons exist
    buttons = driver.find_elements(By.CSS_SELECTOR, "[id^='bet-radar-']")
    print(f"Found {len(buttons)} bet-radar buttons")

    if not buttons:
        print("No bet-radar buttons found. Dumping page source snippet...")
        print(driver.page_source[:3000])
        return {}

    main_handle = driver.current_window_handle
    sportradar_urls: dict[int, str] = {}

    for i in range(len(buttons)):
        btn_id = f"bet-radar-{i}"
        try:
            btn = driver.find_element(By.ID, btn_id)
            # Scroll element into center of viewport to avoid nav-bar interception
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", btn
            )
            time.sleep(0.4)

            handles_before = set(driver.window_handles)
            # Use JS click to bypass overlay/interception issues
            driver.execute_script("arguments[0].click();", btn)

            # Poll for a new tab to appear (up to 4 seconds)
            new_handle = None
            deadline = time.time() + 4
            while time.time() < deadline:
                new_handles = set(driver.window_handles) - handles_before
                if new_handles:
                    new_handle = new_handles.pop()
                    break
                time.sleep(0.1)

            if new_handle:
                driver.switch_to.window(new_handle)
                # If it's a redirect, wait for the final sportradar URL
                try:
                    WebDriverWait(driver, 5).until(EC.url_contains("sportradar"))
                except Exception:
                    pass  # Already at the URL or not a sportradar URL
                url = driver.current_url
                driver.close()
                driver.switch_to.window(main_handle)

                if "sportradar" in url:
                    sportradar_urls[i] = url
                    print(f"  bet-radar-{i}: {url}")
                else:
                    print(f"  bet-radar-{i}: new tab opened but URL is not sportradar: {url[:80]}")
            else:
                print(f"  bet-radar-{i}: no new tab opened within 4s")

        except Exception as e:
            print(f"  bet-radar-{i}: error - {e}")
            # Ensure we're back on the main tab
            if main_handle in driver.window_handles:
                driver.switch_to.window(main_handle)

    return sportradar_urls


def extract_initial_state(driver, url: str) -> dict | None:
    """Navigate to a Sportradar widget URL and extract window.__INITIAL_STATE__."""
    print(f"\nNavigating to Sportradar URL:\n  {url}")
    driver.get(url)

    # Wait for React app to boot
    try:
        from selenium.webdriver.support.ui import WebDriverWait
        WebDriverWait(driver, 15).until(
            lambda d: d.execute_script("return typeof window.__INITIAL_STATE__ !== 'undefined'")
        )
    except Exception:
        print("  Timeout waiting for __INITIAL_STATE__, trying anyway...")
        time.sleep(5)

    try:
        raw = driver.execute_script("return JSON.stringify(window.__INITIAL_STATE__)")
        if raw:
            return json.loads(raw)
        print("  __INITIAL_STATE__ is undefined or null")
        return None
    except Exception as e:
        print(f"  Failed to extract __INITIAL_STATE__: {e}")
        return None


def try_direct_api_calls(sportradar_url: str) -> dict:
    """Try direct HTTP calls to stats.fn.sportradar.com without Selenium."""
    import httpx
    import re

    results: dict[str, object] = {}

    # Parse URL to get IDs
    # Pattern: /season/{seasonId}/headtohead/{homeId}/{awayId}/match/{matchId}
    m = re.search(
        r"/season/(\d+)/headtohead/(\d+)/(\d+)/match/(\d+)", sportradar_url
    )
    if not m:
        print(f"  Could not parse IDs from URL: {sportradar_url}")
        return results

    season_id, home_id, away_id, match_id = m.group(1), m.group(2), m.group(3), m.group(4)
    print(f"\nDirect API test — match={match_id}, season={season_id}, home={home_id}, away={away_id}")

    base = "https://stats.fn.sportradar.com"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://s5.sir.sportradar.com/",
        "Origin": "https://s5.sir.sportradar.com",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Endpoint patterns from widget JS analysis
    endpoints = [
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/match_get/{match_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/stats_match_get/{match_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/stats_h2h_versus/{away_id}/{home_id}/{match_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/stats_team_versus/{home_id}/{away_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/stats_team_streaks/{home_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/stats_season_meta/{season_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/season_standings/{season_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/match_bettingstimulation/{match_id}",
        f"/israelsportsbettingboard/en/Asia:Jerusalem/gismo/match_markets/{match_id}",
        # Without timezone path
        f"/israelsportsbettingboard/en/1/gismo/match_get/{match_id}",
        f"/israelsportsbettingboard/en/1/gismo/stats_h2h_versus/{away_id}/{home_id}/{match_id}",
    ]

    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        for endpoint in endpoints:
            url = base + endpoint
            try:
                resp = client.get(url, headers=headers)
                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                        results[endpoint] = data
                        # Show brief summary
                        if isinstance(data, dict):
                            keys = list(data.keys())[:5]
                            print(f"  ✅ {endpoint}")
                            print(f"     → keys: {keys}")
                        else:
                            print(f"  ✅ {endpoint} → {type(data).__name__}({len(data) if hasattr(data, '__len__') else '?'})")
                    except Exception:
                        print(f"  ✅ {endpoint} → {status} (non-JSON, {len(resp.content)} bytes)")
                        results[endpoint] = {"_raw_bytes": len(resp.content), "_status": status}
                else:
                    print(f"  ❌ {endpoint} → HTTP {status}")
                    results[endpoint] = {"_status": status}
            except Exception as e:
                print(f"  ❌ {endpoint} → error: {e}")
                results[endpoint] = {"_error": str(e)}

    return results


def analyze_initial_state(state: dict) -> dict:
    """Analyze __INITIAL_STATE__ and categorize what data is available."""
    analysis: dict[str, object] = {}

    fetched = state.get("fetchedData", {})
    if not isinstance(fetched, dict):
        fetched = {}

    analysis["fetched_data_keys"] = list(fetched.keys())

    # Categorize by data type
    h2h_keys = [k for k in fetched if "h2h" in k or "headtohead" in k or "versus" in k.lower()]
    form_keys = [k for k in fetched if "streak" in k or "form" in k.lower()]
    standing_keys = [k for k in fetched if "standing" in k or "league" in k.lower() or "season" in k.lower()]
    odds_keys = [k for k in fetched if "market" in k or "odds" in k.lower() or "betting" in k.lower()]
    injury_keys = [k for k in fetched if "injur" in k or "absence" in k.lower() or "lineup" in k.lower()]
    match_keys = [k for k in fetched if "match" in k]

    analysis["h2h_keys"] = h2h_keys
    analysis["form_keys"] = form_keys
    analysis["standing_keys"] = standing_keys
    analysis["odds_keys"] = odds_keys
    analysis["injury_keys"] = injury_keys
    analysis["match_keys"] = match_keys

    # Check what fields are inside each category
    for key in h2h_keys + form_keys + standing_keys + odds_keys + injury_keys:
        data = fetched.get(key, {})
        if isinstance(data, dict) and "doc" in data:
            # Sportradar response format: {doc: [{data: {...}}]}
            docs = data["doc"]
            if docs and isinstance(docs, list) and isinstance(docs[0], dict):
                inner = docs[0].get("data", {})
                if isinstance(inner, dict):
                    analysis[f"{key}_fields"] = list(inner.keys())[:20]

    # Client config
    options = state.get("options", {})
    client_opts = options.get("clientOptions", {})
    analysis["client"] = client_opts.get("client", {})
    analysis["urls"] = options.get("urls", {})

    # Routing
    routing = state.get("routing", {})
    analysis["routing"] = routing

    return analysis


def print_coverage_report(analyses: list[dict]) -> None:
    """Print a table comparing Sportradar data coverage vs API-Football needs."""
    print("\n" + "=" * 70)
    print("SPORTRADAR vs API-FOOTBALL COVERAGE REPORT")
    print("=" * 70)

    needs = {
        "H2H (last 10 matches)": "h2h_keys",
        "Team form (last 5 matches)": "form_keys",
        "League standings": "standing_keys",
        "Betting odds": "odds_keys",
        "Injuries/suspensions": "injury_keys",
        "Match details (referee, date)": "match_keys",
    }

    for need, key in needs.items():
        found_in = sum(1 for a in analyses if a.get(key))
        total = len(analyses)
        pct = (found_in / total * 100) if total > 0 else 0
        status = "✅" if pct >= 50 else ("⚠️" if pct > 0 else "❌")
        print(f"  {status} {need}: found in {found_in}/{total} matches ({pct:.0f}%)")
        # Show example keys
        for a in analyses:
            sample = a.get(key, [])
            if sample:
                print(f"       Example keys: {sample[:3]}")
                break

    print("\nAll unique fetchedData keys across all matches:")
    all_keys: set[str] = set()
    for a in analyses:
        all_keys.update(a.get("fetched_data_keys", []))
    for k in sorted(all_keys):
        print(f"  - {k}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Explore Sportradar widget data")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument("--max-matches", type=int, default=16, help="Max matches to explore (default: 16)")
    parser.add_argument("--skip-widget", action="store_true", help="Skip widget scraping, only test direct API")
    parser.add_argument("--url", help="Test a specific Sportradar URL directly")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analyses: list[dict] = []
    sportradar_urls: dict[int, str] = {}

    if args.url:
        # Single URL mode
        print(f"Testing single URL: {args.url}")
        driver = _chrome_driver(headless=args.headless)
        try:
            state = extract_initial_state(driver, args.url)
            if state:
                out_path = OUTPUT_DIR / "single_url.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                print(f"Saved to {out_path}")
                analysis = analyze_initial_state(state)
                analyses.append(analysis)
                print("\nAnalysis:")
                print(json.dumps(analysis, indent=2, default=str))
        finally:
            driver.quit()

        # Also test direct API
        api_results = try_direct_api_calls(args.url)
        if api_results:
            out_path = OUTPUT_DIR / "single_url_direct_api.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(api_results, f, ensure_ascii=False, indent=2)
            print(f"\nDirect API results saved to {out_path}")

        return

    if not args.skip_widget:
        # Step 1: Get Sportradar URLs from Winner16
        driver = _chrome_driver(headless=args.headless)
        try:
            sportradar_urls = extract_sportradar_urls(driver)
        finally:
            driver.quit()

        if not sportradar_urls:
            print("\nNo Sportradar URLs found. Cannot proceed with widget extraction.")
            print("Try running with --url to test a known Sportradar URL directly.")
            return

        print(f"\nFound {len(sportradar_urls)} Sportradar URLs")

        # Step 2: Extract __INITIAL_STATE__ from each widget URL
        driver = _chrome_driver(headless=args.headless)
        try:
            for i, url in sorted(sportradar_urls.items()):
                if i >= args.max_matches:
                    break

                state = extract_initial_state(driver, url)
                if state:
                    out_path = OUTPUT_DIR / f"match_{i}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False, indent=2)
                    print(f"  Saved match_{i}.json ({out_path.stat().st_size // 1024} KB)")

                    analysis = analyze_initial_state(state)
                    analyses.append(analysis)

                    # Show brief summary
                    print(f"  fetchedData keys: {analysis.get('fetched_data_keys', [])[:8]}")
                else:
                    print(f"  match_{i}: no __INITIAL_STATE__ found")

                time.sleep(1)  # Polite delay

        finally:
            driver.quit()

    # Step 3: Test direct API calls (using first available URL)
    first_url = next(iter(sportradar_urls.values())) if sportradar_urls else None

    if first_url:
        print(f"\n{'=' * 50}")
        print("Testing direct API calls to stats.fn.sportradar.com...")
        api_results = try_direct_api_calls(first_url)
        if api_results:
            out_path = OUTPUT_DIR / "direct_api_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(api_results, f, ensure_ascii=False, indent=2)
            print(f"Direct API results saved to {out_path}")

    # Step 4: Coverage report
    if analyses:
        print_coverage_report(analyses)

    print(f"\nAll output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
