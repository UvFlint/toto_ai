from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from pathlib import Path

from rich.table import Table

from toto_ai.analyzer.agent import analyze_matches
from toto_ai.analyzer.models import FullReport
from toto_ai.display.console import display_report
from toto_ai.display.file_writer import write_report_to_file
from toto_ai.news.collector import gather_news
from toto_ai.scraper.models import Match, WinnerForm
from toto_ai.scraper.winner_scraper import WinnerScraper, create_mock_form
from toto_ai.stats.api_football_stats import ApiFootballStatsCollector
from toto_ai.stats.models import MatchStats
from toto_ai.console import console

_JUNK_STRINGS: frozenset[str] = frozenset(
    {
        "משחק יקבע בהמשך",
        "ראשי",
        "משחקים",
        "מידע",
        "תוצאות",
        "מבצעים",
        "כללי המשחק",
        "שלח טופס",
        "למידע נוסף",
        "מילוי אוטומטי",
        "צמצומים",
        "אירוע",
        "מלא טור",
        "רגיל",
        "רב טורי",
        "משחקים באחריות",
    }
)

_MAX_TEAM_NAME_LEN = 40  # Real team names are never this long
_MIN_MATCHES = 16

_ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
_POLL_INTERVAL_HOURS = 4
_NEXT_FORM_WEEKDAY = 1  # Tuesday (Monday=0 … Sunday=6)
_NEXT_FORM_HOUR = 10  # 10:00 AM Israel time


def _seconds_until_next_tuesday_morning() -> tuple[float, datetime]:
    """Return (seconds, target_dt) until the next Tuesday 10 AM Israel time.

    If it's currently Tuesday before 10 AM, returns time until *today* 10 AM.
    """
    now = datetime.now(_ISRAEL_TZ)
    days_ahead = (_NEXT_FORM_WEEKDAY - now.weekday()) % 7
    if days_ahead == 0 and now.hour >= _NEXT_FORM_HOUR:
        days_ahead = 7
    target = (now + timedelta(days=days_ahead)).replace(
        hour=_NEXT_FORM_HOUR, minute=0, second=0, microsecond=0
    )
    return (target - now).total_seconds(), target


def _validate_form(form: WinnerForm) -> tuple[bool, str]:
    """Return (valid, reason) — True if the form looks like real match data."""
    if not form.matches:
        return False, "no matches found"

    if len(form.matches) < _MIN_MATCHES:
        return False, f"only {len(form.matches)} match(es) found (expected {_MIN_MATCHES})"

    for m in form.matches:
        for team, side in ((m.home_team, "home"), (m.away_team, "away")):
            if team in _JUNK_STRINGS:
                return False, f"{side} team is a known UI string: '{team}'"
            if len(team) > _MAX_TEAM_NAME_LEN:
                return False, (
                    f"{side} team name too long ({len(team)} chars): '{team[:60]}' — "
                    "likely a maintenance or error page"
                )

    return True, ""


async def _fetch_form(dry_run: bool) -> WinnerForm:
    """Fetch the current Winner 16 form."""
    if dry_run:
        console.print("[yellow]Dry run mode: using mock data[/yellow]")
        return create_mock_form()

    scraper = WinnerScraper()
    form = await scraper.fetch_current_form()

    if not form.matches:
        console.print("[yellow]No matches found from scraper. Falling back to mock data.[/yellow]")
        return create_mock_form()

    return form


async def _research_stats(
    matches: list[Match],
    skip_research: bool,
) -> list[MatchStats]:
    """Gather structured stats for all matches using API-Football v3.

    Fetches H2H, form, standings, injuries, odds per match.
    """
    if skip_research:
        console.print("[dim]Skipping match research[/dim]")
        return [MatchStats(home_team=m.home_team, away_team=m.away_team) for m in matches]

    from toto_ai.config import settings

    if not settings.API_FOOTBALL_API_KEY:
        console.print("[dim]API_FOOTBALL_API_KEY not set — skipping research[/dim]")
        return [MatchStats(home_team=m.home_team, away_team=m.away_team) for m in matches]

    collector = ApiFootballStatsCollector()

    match_tuples = [(m.home_team, m.away_team, m.league, m.match_date) for m in matches]
    stats = await collector.research_all_matches(match_tuples)

    console.print(f"[green]Stats research complete: {len(stats)} matches[/green]")
    return stats


async def _enrich_xg(
    matches: list[Match],
    stats: list[MatchStats],
    skip: bool,
) -> None:
    """Enrich stats with understat xG data for supported European leagues."""
    if skip:
        console.print("[dim]Skipping xG enrichment[/dim]")
        return

    from toto_ai.config import settings

    if not settings.UNDERSTAT_ENABLED:
        console.print("[dim]xG enrichment disabled[/dim]")
        return

    try:
        from toto_ai.stats.understat_stats import UnderstatCollector

        collector = UnderstatCollector()
        leagues = [m.league for m in matches]
        await collector.enrich_match_stats(stats, leagues)
    except Exception as e:
        console.print(f"[yellow]xG enrichment failed (non-critical): {e}[/yellow]")


async def _gather_news(
    matches: list[Match],
    stats: list[MatchStats],
    skip: bool,
) -> None:
    """Gather news for all matches using web search + Israeli scrapers."""
    if skip:
        console.print("[dim]Skipping news gathering[/dim]")
        return

    await gather_news(stats, matches)


def _verify_enrichment(matches: list[Match], stats: list[MatchStats]) -> tuple[bool, bool]:
    """Log a summary table of data coverage per match and warn about gaps.

    Returns (critical_gaps, warnings) where critical_gaps blocks the pipeline
    and warnings are informational only (e.g. missing news, empty H2H).
    """
    _UNAVAILABLE_PREFIX = "News unavailable for"

    table = Table(title="Data Coverage", show_lines=False, pad_edge=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Match", min_width=28)
    table.add_column("H2H", justify="center", width=5)
    table.add_column("Home Form", justify="center", width=10)
    table.add_column("Away Form", justify="center", width=10)
    table.add_column("Standings", justify="center", width=10)
    table.add_column("News", justify="center", width=5)
    table.add_column("xG", justify="center", width=4)
    table.add_column("Ref", justify="center", width=4)
    table.add_column("Rest", justify="center", width=5)

    ok = "[green]OK[/green]"
    gap = "[red]MISS[/red]"
    empty = "[yellow]EMPTY[/yellow]"
    warn = "[yellow]MISS[/yellow]"

    total = len(matches)
    critical_matches: list[str] = []
    warning_matches: list[str] = []

    for i, match in enumerate(matches):
        s = stats[i] if i < len(stats) else None
        label = f"{match.home_team} vs {match.away_team}"
        critical_issues: list[str] = []
        warning_issues: list[str] = []

        # H2H: present and has actual match history
        if s and s.h2h and s.h2h.matches:
            h2h_cell = ok
        elif s and s.h2h:
            # H2H object exists but no matches — teams never met, not an error
            h2h_cell = empty
            warning_issues.append("H2H (no history)")
        else:
            h2h_cell = gap
            critical_issues.append("H2H")

        # Home form
        if s and s.home_form and (s.home_form.form_string or s.home_form.recent_matches):
            hf_cell = ok
        elif s and s.home_form:
            hf_cell = empty
            critical_issues.append("Home form (empty)")
        else:
            hf_cell = gap
            critical_issues.append("Home form")

        # Away form
        if s and s.away_form and (s.away_form.form_string or s.away_form.recent_matches):
            af_cell = ok
        elif s and s.away_form:
            af_cell = empty
            critical_issues.append("Away form (empty)")
        else:
            af_cell = gap
            critical_issues.append("Away form")

        # Standings (at least one side)
        if s and (s.home_standing or s.away_standing):
            st_cell = ok
        else:
            st_cell = gap
            critical_issues.append("Standings")

        # News (non-critical — never blocks the pipeline)
        if s and s.news and not s.news.startswith(_UNAVAILABLE_PREFIX):
            nw_cell = ok
        else:
            nw_cell = warn
            warning_issues.append("News")

        # xG (non-critical, only available for supported European leagues)
        if s and (s.home_xg or s.away_xg):
            xg_cell = ok
        elif match.league and any(
            match.league.startswith(p)
            for p in (
                "פרמייר",
                "Premier",
                "לה ליגה",
                "La Liga",
                "בונדס",
                "Bundes",
                "סרייה",
                "Serie",
                "ליג 1",
                "Ligue",
            )
        ):
            xg_cell = warn
            warning_issues.append("xG")
        else:
            xg_cell = "[dim]N/A[/dim]"

        # Referee (non-critical)
        ref_cell = ok if (s and s.referee) else "[dim]N/A[/dim]"

        # Rest days (non-critical)
        rest_cell = (
            ok
            if (s and s.home_rest_days is not None and s.away_rest_days is not None)
            else "[dim]N/A[/dim]"
        )

        table.add_row(
            str(match.match_number),
            label,
            h2h_cell,
            hf_cell,
            af_cell,
            st_cell,
            nw_cell,
            xg_cell,
            ref_cell,
            rest_cell,
        )

        if critical_issues:
            critical_matches.append(
                f"  #{match.match_number} {label}: missing {', '.join(critical_issues)}"
            )
        if warning_issues:
            warning_matches.append(
                f"  #{match.match_number} {label}: missing {', '.join(warning_issues)}"
            )

    console.print(table)

    if critical_matches:
        console.print(
            f"\n[red]Critical data gaps in {len(critical_matches)}/{total} matches:[/red]"
        )
        for line in critical_matches:
            console.print(f"[red]{line}[/red]")
    if warning_matches:
        console.print(
            f"\n[yellow]Non-critical gaps in {len(warning_matches)}/{total} matches "
            f"(will not block analysis)[/yellow]"
        )
    if not critical_matches and not warning_matches:
        console.print(f"\n[green]All {total} matches have full data coverage.[/green]")

    return bool(critical_matches), bool(warning_matches)


async def _poll_for_deadline() -> datetime:
    """Poll winner.co.il every _POLL_INTERVAL_HOURS until a form deadline is found."""
    while True:
        scraper = WinnerScraper()
        try:
            form = await scraper.fetch_current_form()
            if form.deadline is not None:
                deadline_il = form.deadline.astimezone(_ISRAEL_TZ)
                console.print(
                    f"[green]Form deadline found: "
                    f"[bold]{deadline_il.strftime('%Y-%m-%d %H:%M')} (Israeli time)[/bold][/green]"
                )
                return form.deadline
        except Exception as exc:
            console.print(f"[yellow]Scrape attempt failed: {exc}[/yellow]")

        console.print(f"[dim]No deadline yet. Retrying in {_POLL_INTERVAL_HOURS}h...[/dim]")
        await asyncio.sleep(_POLL_INTERVAL_HOURS * 3600)


async def wait_until_4h_before_deadline(deadline: datetime) -> None:
    """
    Wait until 4 hours before the given deadline.

    Args:
        deadline: The deadline datetime (expected to be timezone-aware)
    """
    # Ensure deadline is timezone-aware (assume UTC if naive)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
        console.print("[yellow]Warning: Deadline was naive, assumed UTC[/yellow]")

    # Calculate target time (4 hours before deadline)
    target_time = deadline - timedelta(hours=4)

    # Get current time
    now = datetime.now(timezone.utc)

    # Calculate wait duration
    wait_seconds = (target_time - now).total_seconds()

    if wait_seconds > 0:
        # Convert to readable format
        days = int(wait_seconds // 86400)
        hours = int((wait_seconds % 86400) // 3600)
        minutes = int((wait_seconds % 3600) // 60)

        wait_parts = []
        if days > 0:
            wait_parts.append(f"{days} day{'s' if days > 1 else ''}")
        if hours > 0:
            wait_parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            wait_parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

        wait_str = ", ".join(wait_parts)

        # Format target time nicely
        target_str = target_time.strftime("%Y-%m-%d %H:%M UTC")
        deadline_str = deadline.strftime("%Y-%m-%d %H:%M UTC")

        console.print(
            f"[bold yellow]Schedule mode enabled[/bold yellow]\n"
            f"  Deadline: {deadline_str}\n"
            f"  Will continue at: {target_str} (4 hours before deadline)\n"
            f"  Waiting for: {wait_str}"
        )

        # Wait asynchronously
        await asyncio.sleep(wait_seconds)

        console.print("[bold green]Target time reached! Continuing pipeline...[/bold green]")
    elif wait_seconds < 0:
        # We're already within 4 hours of deadline
        hours_until_deadline = abs(wait_seconds) / 3600

        hours_past_target = abs(wait_seconds) / 3600
        hours_until_deadline = max(
            0.0, (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
        )
        if deadline > datetime.now(timezone.utc):
            console.print(
                f"[yellow]Already within 4-hour window "
                f"({hours_until_deadline:.1f}h until deadline). "
                f"Continuing immediately...[/yellow]"
            )
        else:
            # Past the deadline
            console.print(
                f"[red]Warning: Deadline has passed "
                f"({hours_past_target - 4:.1f}h ago). "
                f"Continuing anyway...[/red]"
            )
    else:
        # Exactly at target time
        console.print("[green]Exactly at target time! Continuing immediately...[/green]")


async def run_pipeline(
    dry_run: bool = False,
    no_research: bool = False,
    premium: bool = False,
    send_auto: bool = False,
    schedule: bool = False,
) -> FullReport:
    """Run the complete analysis pipeline.

    1. Fetch the current Winner 16 form (or mock data)
    2. Research matches via API-Football (stats per match) + one.co.il news
    3. Run AI models in parallel for predictions
    4. Build consensus and display results

    When schedule=True, loops continuously: poll -> check tracker -> run -> repeat.
    """
    from toto_ai.config import settings
    from toto_ai.tracker import SubmissionTracker

    tracker = SubmissionTracker(settings.TRACKER_FILE)
    report: FullReport | None = None

    while True:
        polled_deadline: datetime | None = None
        if schedule:
            console.rule("[bold]Schedule Mode: Waiting for Form[/bold]")
            polled_deadline = await _poll_for_deadline()

        console.rule("[bold]Step 1: Fetching Winner 16 Form[/bold]")
        form = await _fetch_form(dry_run)
        console.print(f"Form: {form.form_number or 'unknown'} with {len(form.matches)} matches\n")

        deadline = form.deadline or polled_deadline

        # ── Skip if already submitted ──────────────────────────────────
        if schedule and form.form_number and tracker.is_submitted(form.form_number):
            wait_secs, target = _seconds_until_next_tuesday_morning()
            target_str = target.strftime("%A %Y-%m-%d %H:%M %Z")
            days = int(wait_secs // 86400)
            hours = int((wait_secs % 86400) // 3600)
            console.print(
                f"[yellow]Form {form.form_number} already submitted. "
                f"Sleeping until {target_str} ({days}d {hours}h)…[/yellow]"
            )
            await asyncio.sleep(wait_secs)
            continue

        valid, reason = _validate_form(form)
        if not valid:
            console.print(f"[bold red]Error: Scraped form data is invalid — {reason}.[/bold red]")
            console.print(
                "[yellow]The site may be showing a maintenance/error page instead of matches. "
                "Try running again or check winner.co.il manually.[/yellow]"
            )
            if schedule:
                console.print("[yellow]Retrying in 60 s…[/yellow]")
                await asyncio.sleep(60)
                continue
            raise SystemExit(1)

        if schedule:
            if deadline is not None:
                await wait_until_4h_before_deadline(deadline)
            else:
                console.print("[yellow]No deadline available — running immediately.[/yellow]")

        console.rule("[bold]Step 2: Researching Matches (API-Football)[/bold]")
        stats = await _research_stats(
            form.matches,
            no_research,
        )
        console.print()

        # ── Step 2b: xG Enrichment (Understat) ────────────────────────
        console.rule("[bold]Step 2b: xG Enrichment (Understat)[/bold]")
        await _enrich_xg(form.matches, stats, no_research)
        console.print()

        # ── Step 2c: Statistical Model (Poisson/Dixon-Coles) ─────────
        console.rule("[bold]Step 2c: Statistical Model (Poisson/Dixon-Coles)[/bold]")
        from toto_ai.stats.poisson import compute_all_probabilities

        poisson_probs = compute_all_probabilities(form.matches, stats)
        computed = 0
        for i, p in enumerate(poisson_probs):
            if p and i < len(stats):
                stats[i].poisson_probs = p
                computed += 1
        console.print(
            f"[green]Poisson probabilities computed for {computed}/{len(form.matches)} matches[/green]"
        )
        console.print()

        # ── Step 2d: CatBoost ML Probabilities ──────────────────────
        console.rule("[bold]Step 2d: CatBoost ML Probabilities[/bold]")
        from toto_ai.ml.catboost_model import enrich_stats_with_catboost

        enrich_stats_with_catboost(form.matches, stats)
        cb_count = sum(1 for s in stats if s.catboost_probs)
        console.print(
            f"[green]CatBoost probabilities computed for {cb_count}/{len(form.matches)} matches[/green]"
        )
        console.print()

        # ── Step 2e: XGBoost ML Probabilities ──────────────────────
        console.rule("[bold]Step 2e: XGBoost ML Probabilities[/bold]")
        from toto_ai.ml.xgboost_model import enrich_stats_with_xgboost

        enrich_stats_with_xgboost(form.matches, stats)
        xgb_count = sum(1 for s in stats if s.xgboost_probs)
        console.print(
            f"[green]XGBoost probabilities computed for {xgb_count}/{len(form.matches)} matches[/green]"
        )
        console.print()

        # ── Step 3: News Gathering ────────────────────────────────────
        console.rule("[bold]Step 3: Gathering News[/bold]")
        await _gather_news(form.matches, stats, no_research)
        console.print()

        console.rule("[bold]Data Verification[/bold]")
        critical_gaps, _warnings = _verify_enrichment(form.matches, stats)
        console.print()

        # Abort if research was attempted but critical data is missing
        if critical_gaps and not no_research and not dry_run:
            if schedule:
                # Retry research once before giving up
                console.print(
                    "[yellow]Critical data gaps detected. Retrying research once...[/yellow]"
                )
                stats = await _research_stats(form.matches, no_research)
                await _gather_news(form.matches, stats, no_research)
                console.rule("[bold]Data Verification (Retry)[/bold]")
                critical_gaps, _warnings = _verify_enrichment(form.matches, stats)
                console.print()

            if critical_gaps:
                console.print(
                    "[bold red]Aborting: cannot run AI analysis with missing research data.[/bold red]"
                )
                if schedule:
                    console.print("[yellow]Will retry on next schedule cycle.[/yellow]")
                    wait_secs, target = _seconds_until_next_tuesday_morning()
                    target_str = target.strftime("%A %Y-%m-%d %H:%M %Z")
                    days = int(wait_secs // 86400)
                    hours = int((wait_secs % 86400) // 3600)
                    console.print(f"[dim]Sleeping until {target_str} ({days}d {hours}h)…[/dim]")
                    await asyncio.sleep(wait_secs)
                    continue
                raise SystemExit(1)

        from toto_ai.analyzer.agent import MODELS_PREMIUM, MODELS_STANDARD

        models = MODELS_PREMIUM if premium else MODELS_STANDARD
        tier = "Premium" if premium else "Standard"
        console.rule(f"[bold]Step 4: AI Analysis ({len(models)} Models — {tier})[/bold]")
        report = await analyze_matches(form.matches, stats, premium=premium)
        report.form_number = form.form_number
        console.print()

        console.rule("[bold]Step 5: Results[/bold]")
        display_report(report)

        saved_path = write_report_to_file(report)
        console.print(f"\n[dim]Report saved to: {saved_path}[/dim]")

        # ── Step 6: Email Notification ─────────────────────────────────
        if not dry_run:
            _send_email_report(report, saved_path)

        # ── Step 7: Form Submission ────────────────────────────────────
        results = await _submit_predictions(report, dry_run=dry_run, send_auto=send_auto)

        # Record successful submission in tracker
        if results and any(r.success for r in results) and form.form_number:
            submitted_count = sum(1 for r in results if r.success)
            tracker.record_submission(
                form.form_number, deadline=deadline, columns_count=submitted_count
            )
            console.print(f"[dim]Submission recorded in tracker for form {form.form_number}[/dim]")

        if not schedule:
            break

        wait_secs, target = _seconds_until_next_tuesday_morning()
        target_str = target.strftime("%A %Y-%m-%d %H:%M %Z")
        days = int(wait_secs // 86400)
        hours = int((wait_secs % 86400) // 3600)
        console.rule("[bold]Round complete[/bold]")
        console.print(f"[dim]Next form expected {target_str} ({days}d {hours}h). Sleeping…[/dim]")
        await asyncio.sleep(wait_secs)

    assert report is not None
    return report


async def run_test_pipeline(
    url: str,
    no_research: bool = False,
    premium: bool = False,
) -> None:
    """Run backtesting pipeline against a past results form.

    Scrapes a past form with results, runs the normal AI prediction pipeline,
    then compares predictions against actual results. No email or submission.
    """
    from toto_ai.display.console import display_test_report

    # Step 1: Scrape past results
    console.rule("[bold]Step 1: Fetching Past Results Form[/bold]")
    scraper = WinnerScraper()
    form = await scraper.fetch_results_form(url)

    if not form.matches:
        console.print("[bold red]Error: No matches found from the results page.[/bold red]")
        console.print(
            "[yellow]Check the URL and try again, or run with HEADLESS=false to inspect.[/yellow]"
        )
        raise SystemExit(1)

    valid, reason = _validate_form(form)
    if not valid:
        console.print(f"[bold red]Error: Scraped form data is invalid — {reason}.[/bold red]")
        raise SystemExit(1)

    results_count = sum(1 for m in form.matches if m.result)
    console.print(
        f"Form: {form.form_number or 'unknown'} with {len(form.matches)} matches "
        f"({results_count} with results)"
    )
    console.print()
    for m in form.matches:
        result_str = m.result or "?"
        console.print(f"  #{m.match_number} {m.home_team} vs {m.away_team} → {result_str}")
    console.print()

    if results_count < 16:
        console.print(
            f"[bold red]Only {results_count} results found (need 16). "
            f"Cannot run backtest.[/bold red]"
        )
        raise SystemExit(1)

    # Derive reference date from earliest match date (cutoff for stats/news)
    match_dates = [m.match_date for m in form.matches if m.match_date]
    reference_date = min(match_dates) if match_dates else None
    if reference_date:
        console.print(
            f"[dim]Reference date for backtest: {reference_date.strftime('%Y-%m-%d')} "
            f"(stats/news filtered to before this date)[/dim]"
        )

    # Step 2: Research matches
    console.rule("[bold]Step 2: Researching Matches (API-Football)[/bold]")
    stats = await _research_stats(
        form.matches,
        no_research,
    )
    console.print()

    # Step 2b: xG Enrichment
    console.rule("[bold]Step 2b: xG Enrichment (Understat)[/bold]")
    await _enrich_xg(form.matches, stats, no_research)
    console.print()

    # Step 2c: Statistical Model (Poisson/Dixon-Coles)
    console.rule("[bold]Step 2c: Statistical Model (Poisson/Dixon-Coles)[/bold]")
    from toto_ai.stats.poisson import compute_all_probabilities

    poisson_probs = compute_all_probabilities(form.matches, stats)
    computed = 0
    for i, p in enumerate(poisson_probs):
        if p and i < len(stats):
            stats[i].poisson_probs = p
            computed += 1
    console.print(
        f"[green]Poisson probabilities computed for {computed}/{len(form.matches)} matches[/green]"
    )
    console.print()

    # Step 2d: CatBoost ML Probabilities
    console.rule("[bold]Step 2d: CatBoost ML Probabilities[/bold]")
    from toto_ai.ml.catboost_model import enrich_stats_with_catboost

    enrich_stats_with_catboost(form.matches, stats)
    cb_count = sum(1 for s in stats if s.catboost_probs)
    console.print(
        f"[green]CatBoost probabilities computed for {cb_count}/{len(form.matches)} matches[/green]"
    )
    console.print()

    # Step 2e: XGBoost ML Probabilities
    console.rule("[bold]Step 2e: XGBoost ML Probabilities[/bold]")
    from toto_ai.ml.xgboost_model import enrich_stats_with_xgboost

    enrich_stats_with_xgboost(form.matches, stats)
    xgb_count = sum(1 for s in stats if s.xgboost_probs)
    console.print(
        f"[green]XGBoost probabilities computed for {xgb_count}/{len(form.matches)} matches[/green]"
    )
    console.print()

    # Step 3: News gathering
    console.rule("[bold]Step 3: Gathering News[/bold]")
    await _gather_news(form.matches, stats, no_research)
    console.print()

    # Data verification
    console.rule("[bold]Data Verification[/bold]")
    critical_gaps, _warnings = _verify_enrichment(form.matches, stats)
    console.print()

    if critical_gaps and not no_research:
        console.print(
            "[bold red]Aborting: cannot run AI analysis with missing research data.[/bold red]"
        )
        raise SystemExit(1)

    # Step 4: AI Analysis
    from toto_ai.analyzer.agent import MODELS_PREMIUM, MODELS_STANDARD

    models = MODELS_PREMIUM if premium else MODELS_STANDARD
    tier = "Premium" if premium else "Standard"
    console.rule(f"[bold]Step 4: AI Analysis ({len(models)} Models — {tier})[/bold]")
    report = await analyze_matches(
        form.matches, stats, premium=premium, reference_date=reference_date
    )
    report.form_number = form.form_number
    console.print()

    # Step 5: Display predictions
    console.rule("[bold]Step 5: Predictions[/bold]")
    display_report(report)

    # Save report
    saved_path = write_report_to_file(report)
    console.print(f"\n[dim]Report saved to: {saved_path}[/dim]")

    # Step 7: Backtesting comparison
    display_test_report(report, form.matches)


def _send_email_report(report: FullReport, report_path: Path) -> None:
    """Send the report markdown to the configured email recipient."""
    from toto_ai.config import settings
    from toto_ai.notifier.email_sender import send_report_email

    if not settings.EMAIL_SENDER or not settings.EMAIL_PASSWORD:
        console.print(
            "[dim]Skipping email notification (EMAIL_SENDER/EMAIL_PASSWORD not set)[/dim]"
        )
        return

    recipient = settings.EMAIL_RECIPIENT
    console.print(f"[dim]Sending report email to {recipient}…[/dim]")
    try:
        send_report_email(
            report,
            report_path,
            sender=settings.EMAIL_SENDER,
            password=settings.EMAIL_PASSWORD,
            recipient=recipient,
        )
        console.print(f"[green]Email sent to {recipient}[/green]")
    except Exception as exc:
        console.print(f"[yellow]Warning: failed to send email — {exc}[/yellow]")


async def _submit_predictions(report: FullReport, *, dry_run: bool, send_auto: bool) -> list | None:
    """Optionally submit predictions to winner.co.il."""
    from rich.prompt import Confirm

    from toto_ai.config import settings
    from toto_ai.submitter.winner_submitter import WinnerSubmitter, deduplicate_columns

    if dry_run:
        return

    if not settings.WINNER_USERNAME or not settings.WINNER_PASSWORD:
        console.print(
            "\n[dim]Skipping form submission (WINNER_USERNAME/WINNER_PASSWORD not set)[/dim]"
        )
        return

    console.rule("[bold]Step 6: Form Submission[/bold]")

    columns = deduplicate_columns(report)
    if not columns:
        console.print("[yellow]No columns to submit[/yellow]")
        return

    total_models = sum(len(c.source_models) for c in columns)
    duplicates_removed = total_models - len(columns)
    cost = len(columns) * 3

    console.print(f"\nUnique columns to submit: {len(columns)}")
    for col in columns:
        models_str = ", ".join(col.source_models)
        preds_str = " ".join(col.predictions)
        console.print(f"  Column {col.column_index} ({models_str}): {preds_str}")
    if duplicates_removed > 0:
        console.print(f"{duplicates_removed} duplicate column(s) removed")
    console.print(f"Cost: {cost} ₪\n")

    if not send_auto:
        confirmed = Confirm.ask(
            f"Submit {len(columns)} column(s) to winner.co.il? ({cost} ₪)",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Submission skipped[/dim]")
            return

    submitter = WinnerSubmitter()
    results = submitter.submit_predictions(columns)

    console.print()
    for r in results:
        if r.success:
            console.print(f"[green]  Column {r.column_index}: {r.message}[/green]")
        else:
            console.print(f"[red]  Column {r.column_index}: {r.message}[/red]")

    return results
