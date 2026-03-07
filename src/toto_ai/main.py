from __future__ import annotations

import asyncio
import sys

import click
from dotenv import load_dotenv

# Force UTF-8 output on Windows (Hebrew characters crash with cp1252 default)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env into os.environ so pydantic-ai providers can find API keys.
# Use CWD-based search (no path arg) to avoid issues with Hebrew chars in paths.
load_dotenv(override=True)

from toto_ai.console import console


@click.command()
@click.option("--dry-run", is_flag=True, help="Use mock match data (no scraping)")
@click.option("--no-research", is_flag=True, help="Skip API-Football match research (stats + news)")
@click.option(
    "--premium", is_flag=True, help="Use high-end models (GPT-5.2, Gemini Pro, Claude Opus)"
)
@click.option(
    "--send-auto",
    is_flag=True,
    help="Submit predictions to winner.co.il without confirmation prompt",
)
@click.option(
    "--schedule", is_flag=True, help="Schedule the pipeline according to winner16 form end of life"
)
@click.option(
    "--mark-submitted",
    type=str,
    default=None,
    help="Mark a form number as already submitted and exit",
)
@click.option("--test", is_flag=True, help="Backtest against a past results form (prompts for URL)")
@click.option("--download-data", is_flag=True, help="Download historical match CSVs from football-data.co.uk")
@click.option("--force-download", is_flag=True, help="Re-download existing CSV files")
@click.option("--train-model", is_flag=True, help="Train CatBoost models from historical data")
def main(
    dry_run: bool,
    no_research: bool,
    premium: bool,
    send_auto: bool,
    schedule: bool,
    mark_submitted: str | None,
    test: bool,
    download_data: bool,
    force_download: bool,
    train_model: bool,
) -> None:
    """Analyze the current Winner 16 form and predict outcomes using AI models."""
    if test:
        if dry_run or send_auto or schedule:
            console.print(
                "[red]--test cannot be combined with --dry-run, --send-auto, or --schedule[/red]"
            )
            raise SystemExit(1)

        url = click.prompt("Enter the past results form URL from winner.co.il")
        from toto_ai.pipeline import run_test_pipeline

        console.print("[bold blue]Toto AI - Backtesting Mode[/bold blue]")
        console.print()
        asyncio.run(run_test_pipeline(url=url, no_research=no_research, premium=premium))
        return

    if download_data:
        from toto_ai.config import settings
        from toto_ai.data_collector.football_data_downloader import FootballDataDownloader

        console.print("[bold blue]Toto AI - Historical Data Download[/bold blue]")
        console.print()
        downloader = FootballDataDownloader(data_dir=settings.FOOTBALL_DATA_DIR)
        asyncio.run(downloader.download_all(force=force_download))
        return

    if train_model:
        from toto_ai.config import settings
        from toto_ai.ml.train import run_training

        console.print("[bold blue]Toto AI - CatBoost Model Training[/bold blue]")
        console.print()
        run_training(data_dir=settings.FOOTBALL_DATA_DIR, model_dir=settings.MODEL_DIR)
        return

    if mark_submitted:
        from toto_ai.config import settings
        from toto_ai.tracker import SubmissionTracker

        tracker = SubmissionTracker(settings.TRACKER_FILE)
        tracker.mark_as_submitted(mark_submitted)
        console.print(f"[green]Form {mark_submitted} marked as submitted.[/green]")
        return

    from toto_ai.pipeline import run_pipeline

    console.print("[bold blue]Toto AI - Winner 16 Analyzer[/bold blue]")
    console.print()

    asyncio.run(
        run_pipeline(
            dry_run=dry_run,
            no_research=no_research,
            premium=premium,
            send_auto=send_auto,
            schedule=schedule,
        )
    )


if __name__ == "__main__":
    main()
