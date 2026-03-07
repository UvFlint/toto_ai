"""
Standalone script to submit predictions from a saved report JSON.
Skips the entire pipeline — uses the latest (or specified) report file.

Usage:
    uv run python scripts/submit.py                  # latest report
    uv run python scripts/submit.py path/to/report.json
    uv run python scripts/submit.py --auto           # skip confirmation prompt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(override=True)

from rich.console import Console
from rich.prompt import Confirm

from toto_ai.analyzer.models import FullReport
from toto_ai.config import settings
from toto_ai.submitter.models import SubmissionColumn
from toto_ai.submitter.winner_submitter import WinnerSubmitter, deduplicate_columns

console = Console()


def find_latest_report() -> Path:
    """Find the most recent report JSON in the reports/ directory."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        console.print("[bold red]No reports/ directory found.[/bold red]")
        raise SystemExit(1)

    json_files = sorted(reports_dir.glob("report_*.json"), reverse=True)
    if not json_files:
        console.print(
            "[bold red]No report JSON files found in reports/.[/bold red]\n"
            "[dim]Run the full pipeline once to generate a JSON report.[/dim]"
        )
        raise SystemExit(1)

    return json_files[0]


def load_report(path: Path) -> FullReport:
    """Load a FullReport from a JSON file."""
    console.print(f"Loading report: [cyan]{path}[/cyan]")
    data = path.read_text(encoding="utf-8")
    return FullReport.model_validate_json(data)


def display_columns(columns: list[SubmissionColumn], total_models: int) -> None:
    """Show the columns that will be submitted."""
    duplicates_removed = total_models - len(columns)
    cost = len(columns) * 3

    console.print(f"\nUnique columns to submit: [bold]{len(columns)}[/bold]")
    for col in columns:
        models_str = ", ".join(col.source_models)
        preds_str = " ".join(col.predictions)
        console.print(f"  Column {col.column_index} ({models_str}): {preds_str}")
    if duplicates_removed > 0:
        console.print(f"{duplicates_removed} duplicate column(s) removed")
    console.print(f"Cost: [bold]{cost} ₪[/bold]\n")


def main() -> None:
    args = sys.argv[1:]
    auto = "--auto" in args
    args = [a for a in args if a != "--auto"]

    if not settings.WINNER_USERNAME or not settings.WINNER_PASSWORD:
        console.print("[bold red]WINNER_USERNAME / WINNER_PASSWORD not set in .env[/bold red]")
        raise SystemExit(1)

    report_path = Path(args[0]) if args else find_latest_report()
    if not report_path.exists():
        console.print(f"[bold red]File not found: {report_path}[/bold red]")
        raise SystemExit(1)

    report = load_report(report_path)
    console.print(
        f"Form: [bold]{report.form_number or 'unknown'}[/bold]  |  "
        f"Models: {', '.join(c.model_name for c in report.columns)}"
    )

    columns = deduplicate_columns(report)
    if not columns:
        console.print("[yellow]No columns to submit[/yellow]")
        return

    total_models = sum(len(c.source_models) for c in columns)
    display_columns(columns, total_models)

    cost = len(columns) * 3
    if not auto:
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


if __name__ == "__main__":
    main()
