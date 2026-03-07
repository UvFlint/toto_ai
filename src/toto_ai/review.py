from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
from rich.table import Table
from rich.text import Text

from toto_ai.analyzer.models import FullReport
from toto_ai.calibration import (
    CalibrationManager,
    MatchReviewRecord,
    WeeklyReview,
)
from toto_ai.console import console


def _find_report_json(form_number: str | None) -> Path | None:
    """Find the latest report JSON for the given form number (or overall latest)."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None

    if form_number:
        pattern = f"report_{form_number}*.json"
    else:
        pattern = "report_*.json"

    matches = sorted(reports_dir.glob(pattern))
    return matches[-1] if matches else None


async def run_review(form_number: str | None = None) -> None:
    """Review predictions vs actual results and update calibration."""
    from toto_ai.config import settings
    from toto_ai.scraper.winner_scraper import WinnerScraper

    # 1. Find report JSON
    report_path = _find_report_json(form_number)
    if not report_path:
        target = f"form {form_number}" if form_number else "any form"
        console.print(f"[red]No report JSON found for {target} in reports/[/red]")
        raise SystemExit(1)

    console.print(f"[dim]Loading report: {report_path}[/dim]")
    report = FullReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    console.print(
        f"Found report for form [bold]{report.form_number}[/bold] "
        f"with {len(report.columns)} columns, "
        f"{len(report.news_snapshots)} news snapshots"
    )
    console.print()

    # 2. Prompt for results URL
    url = click.prompt("Enter the results form URL from winner.co.il")

    # 3. Scrape actual results
    console.rule("[bold]Fetching Results[/bold]")
    scraper = WinnerScraper()
    results_form = await scraper.fetch_results_form(url)

    results_map: dict[int, str] = {}
    match_info: dict[int, tuple[str, str]] = {}
    for m in results_form.matches:
        if m.result:
            results_map[m.match_number] = m.result
            match_info[m.match_number] = (m.home_team, m.away_team)

    if len(results_map) < 16:
        console.print(f"[yellow]Warning: only {len(results_map)}/16 results available[/yellow]")
    console.print(f"[green]Fetched {len(results_map)} results[/green]")
    console.print()

    # 4. Build news snapshot lookup
    snapshot_map = {ns.match_number: ns for ns in report.news_snapshots}

    # 5. Build consensus lookup
    consensus_map = {c.match_number: c.prediction for c in report.consensus}

    # 6. Build review records
    calibration = CalibrationManager(settings.CALIBRATION_FILE)
    current_multiplier = calibration.post_odds_multiplier

    records: list[MatchReviewRecord] = []
    for match_num, actual in sorted(results_map.items()):
        home, away = match_info[match_num]

        # Collect predictions from each column
        preds: dict[str, str] = {}
        for col in report.columns:
            for p in col.predictions:
                if p.match_number == match_num:
                    preds[col.model_name] = p.prediction
                    break

        ns = snapshot_map.get(match_num)
        records.append(
            MatchReviewRecord(
                match_number=match_num,
                home_team=home,
                away_team=away,
                predictions=preds,
                consensus=consensus_map.get(match_num),
                actual_result=actual,
                had_news_analysis=ns is not None,
                has_x_factor=ns.has_x_factor if ns else False,
                net_impact=ns.net_impact if ns else 0.0,
                post_odds_item_count=ns.post_odds_item_count if ns else 0,
                multiplier_used=ns.multiplier_used if ns else current_multiplier,
            )
        )

    # 7. Compute aggregate metrics
    total = len(records)
    consensus_correct = sum(1 for r in records if r.consensus and r.consensus == r.actual_result)
    consensus_acc = consensus_correct / total if total else 0.0

    xf_records = [r for r in records if r.has_x_factor]
    nxf_records = [r for r in records if not r.has_x_factor]

    xf_correct = sum(1 for r in xf_records if r.consensus and r.consensus == r.actual_result)
    xf_acc = xf_correct / len(xf_records) if xf_records else 0.0

    nxf_correct = sum(1 for r in nxf_records if r.consensus and r.consensus == r.actual_result)
    nxf_acc = nxf_correct / len(nxf_records) if nxf_records else 0.0

    # 8. Build WeeklyReview
    review = WeeklyReview(
        form_number=report.form_number or "unknown",
        reviewed_at=datetime.now(timezone.utc),
        multiplier_before=current_multiplier,
        multiplier_after=current_multiplier,  # placeholder, computed below
        matches=records,
        total_matches=total,
        consensus_correct=consensus_correct,
        consensus_accuracy=round(consensus_acc, 4),
        xfactor_matches=len(xf_records),
        xfactor_correct=xf_correct,
        xfactor_accuracy=round(xf_acc, 4),
        non_xfactor_matches=len(nxf_records),
        non_xfactor_correct=nxf_correct,
        non_xfactor_accuracy=round(nxf_acc, 4),
    )

    # Compute new multiplier
    new_multiplier = calibration.compute_new_multiplier(review)
    review.multiplier_after = new_multiplier

    # 9. Display comparison table
    _display_review_table(records, report)

    # 10. Display accuracy summary
    _display_accuracy_summary(review)

    # 11. Display multiplier change
    console.print()
    if new_multiplier != current_multiplier:
        console.print(f"[bold]Post-odds multiplier: {current_multiplier} → {new_multiplier}[/bold]")
    else:
        console.print(f"[dim]Post-odds multiplier unchanged: {current_multiplier}[/dim]")

    xf_lift = xf_acc - nxf_acc if xf_records and nxf_records else 0.0
    if xf_records:
        console.print(f"[dim]X-factor lift: {xf_lift:+.1%}[/dim]")

    # 12. Save review
    calibration.record_review(review)
    console.print(f"\n[green]Review saved to {calibration.path}[/green]")


def _display_review_table(records: list[MatchReviewRecord], report: FullReport) -> None:
    """Display a Rich table comparing predictions to actual results."""
    console.rule("[bold blue]Prediction Review[/bold blue]")
    console.print()

    table = Table(
        title="Predictions vs Actual Results",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )

    table.add_column("#", justify="center", width=3)
    table.add_column("Home", justify="right", min_width=12)
    table.add_column("Away", justify="left", min_width=12)
    table.add_column("Actual", justify="center", width=7, style="bold")
    table.add_column("Consensus", justify="center", width=10)
    table.add_column("X-Factor", justify="center", width=8)

    for r in records:
        correct = r.consensus == r.actual_result if r.consensus else False
        style = "bold green" if correct else "red"
        mark = "V" if correct else "X"
        cons_text = (
            Text(f"{r.consensus} {mark}", style=style) if r.consensus else Text("-", style="dim")
        )
        xf_text = Text("*", style="bold yellow") if r.has_x_factor else Text("")

        table.add_row(
            str(r.match_number),
            r.home_team,
            r.away_team,
            r.actual_result,
            cons_text,
            xf_text,
        )

    console.print(table)


def _display_accuracy_summary(review: WeeklyReview) -> None:
    """Display accuracy breakdown panel."""
    console.print()
    table = Table(
        title="Accuracy Summary",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Category", min_width=14)
    table.add_column("Correct", justify="center", width=8)
    table.add_column("Total", justify="center", width=6)
    table.add_column("Accuracy", justify="center", width=10)

    table.add_row(
        "Consensus",
        str(review.consensus_correct),
        str(review.total_matches),
        f"{review.consensus_accuracy:.0%}",
    )

    table.add_section()
    table.add_row(
        Text("X-Factor matches", style="bold yellow"),
        str(review.xfactor_correct),
        str(review.xfactor_matches),
        f"{review.xfactor_accuracy:.0%}" if review.xfactor_matches else "-",
    )
    table.add_row(
        "Non-X-Factor matches",
        str(review.non_xfactor_correct),
        str(review.non_xfactor_matches),
        f"{review.non_xfactor_accuracy:.0%}" if review.non_xfactor_matches else "-",
    )

    console.print(table)
