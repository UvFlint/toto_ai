from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from toto_ai.console import console

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{division}.csv"

ALL_DIVISIONS: dict[str, str] = {
    # England
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    # Germany
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    # Italy
    "I1": "Serie A",
    "I2": "Serie B",
    # Spain
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    # France
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    # Scotland
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    # Netherlands
    "N1": "Eredivisie",
    # Belgium
    "B1": "Jupiler League",
    # Portugal
    "P1": "Liga Portugal",
    # Turkey
    "T1": "Super Lig",
    # Greece
    "G1": "Super League Greece",
    # Israel
    "ISR1": "Israeli Premier League",
    "ISR_CUP": "Israeli State Cup",
}

EXTRA_LEAGUES_URL = "https://www.football-data.co.uk/new/{code}.csv"

EXTRA_LEAGUE_CODES: dict[str, str] = {
    "ARG": "Argentina",
    "AUT": "Austria",
    "BRA": "Brazil",
    "CHN": "China",
    "DNK": "Denmark",
    "FIN": "Finland",
    "IRL": "Ireland",
    "JPN": "Japan",
    "MEX": "Mexico",
    "NOR": "Norway",
    "POL": "Poland",
    "ROU": "Romania",
    "RUS": "Russia",
    "SWE": "Sweden",
    "SWZ": "Switzerland",
    "USA": "USA",
}


def _generate_season_codes(start_year: int = 1993, end_year: int = 2025) -> list[str]:
    """Generate season codes like '9394', '0001', '2526'."""
    codes: list[str] = []
    for year in range(start_year, end_year + 1):
        s = f"{year % 100:02d}{(year + 1) % 100:02d}"
        codes.append(s)
    return codes


class DownloadResult(BaseModel):
    season_code: str
    division: str
    status: Literal["downloaded", "skipped", "failed"]
    error: str | None = None


class DownloadSummary(BaseModel):
    total: int
    downloaded: int
    skipped: int
    failed: int
    results: list[DownloadResult]


class FootballDataDownloader:
    """Downloads historical match CSVs from football-data.co.uk."""

    def __init__(
        self,
        data_dir: str = "data/football_data",
        divisions: list[str] | None = None,
        start_year: int = 1993,
        end_year: int = 2025,
        max_concurrent: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.divisions = divisions or list(ALL_DIVISIONS.keys())
        self.season_codes = _generate_season_codes(start_year, end_year)
        self.max_concurrent = max_concurrent

    def _build_url(self, season_code: str, division: str) -> str:
        return BASE_URL.format(season_code=season_code, division=division)

    def _file_path(self, season_code: str, division: str) -> Path:
        return self.data_dir / division / f"{season_code}.csv"

    async def _download_file(
        self,
        client: httpx.AsyncClient,
        season_code: str,
        division: str,
        semaphore: asyncio.Semaphore,
        force: bool,
        progress: Progress,
        task_id: int,
    ) -> DownloadResult:
        path = self._file_path(season_code, division)

        if path.exists() and not force:
            progress.advance(task_id)
            return DownloadResult(season_code=season_code, division=division, status="skipped")

        url = self._build_url(season_code, division)
        async with semaphore:
            for attempt in range(2):
                try:
                    resp = await client.get(url, follow_redirects=True)
                    if resp.status_code == 404:
                        progress.advance(task_id)
                        return DownloadResult(
                            season_code=season_code,
                            division=division,
                            status="failed",
                            error="404 Not Found",
                        )
                    resp.raise_for_status()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(resp.content)
                    progress.advance(task_id)
                    return DownloadResult(
                        season_code=season_code, division=division, status="downloaded"
                    )
                except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                    if attempt == 0:
                        await asyncio.sleep(1)
                        continue
                    progress.advance(task_id)
                    return DownloadResult(
                        season_code=season_code,
                        division=division,
                        status="failed",
                        error=str(e),
                    )

        progress.advance(task_id)
        return DownloadResult(
            season_code=season_code, division=division, status="failed", error="Unknown error"
        )

    async def _download_extra_league(
        self,
        client: httpx.AsyncClient,
        code: str,
        semaphore: asyncio.Semaphore,
        force: bool,
        progress: Progress,
        task_id: int,
    ) -> DownloadResult:
        path = self.data_dir / "extra" / f"{code}.csv"

        if path.exists() and not force:
            progress.advance(task_id)
            return DownloadResult(season_code="all", division=code, status="skipped")

        url = EXTRA_LEAGUES_URL.format(code=code)
        async with semaphore:
            for attempt in range(2):
                try:
                    resp = await client.get(url, follow_redirects=True)
                    if resp.status_code == 404:
                        progress.advance(task_id)
                        return DownloadResult(
                            season_code="all", division=code, status="failed", error="404 Not Found"
                        )
                    resp.raise_for_status()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(resp.content)
                    progress.advance(task_id)
                    return DownloadResult(season_code="all", division=code, status="downloaded")
                except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                    if attempt == 0:
                        await asyncio.sleep(1)
                        continue
                    progress.advance(task_id)
                    return DownloadResult(
                        season_code="all", division=code, status="failed", error=str(e)
                    )

        progress.advance(task_id)
        return DownloadResult(
            season_code="all", division=code, status="failed", error="Unknown error"
        )

    async def download_all(self, force: bool = False) -> DownloadSummary:
        """Download all configured CSVs (main leagues + extra leagues)."""
        pairs = [
            (season, div) for div in self.divisions for season in self.season_codes
        ]
        extra_codes = list(EXTRA_LEAGUE_CODES.keys())
        total = len(pairs) + len(extra_codes)

        console.print(
            f"[bold]Downloading {total} items "
            f"({len(self.divisions)} divisions x {len(self.season_codes)} seasons "
            f"+ {len(extra_codes)} extra leagues)[/bold]"
        )
        console.print(f"Target directory: [cyan]{self.data_dir}[/cyan]\n")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Downloading", total=total)

            async with httpx.AsyncClient(timeout=30) as client:
                main_tasks = [
                    self._download_file(client, season, div, semaphore, force, progress, task_id)
                    for season, div in pairs
                ]
                extra_tasks = [
                    self._download_extra_league(client, code, semaphore, force, progress, task_id)
                    for code in extra_codes
                ]
                results = await asyncio.gather(*main_tasks, *extra_tasks)

        downloaded = sum(1 for r in results if r.status == "downloaded")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")

        console.print()
        console.print(f"[green]Downloaded:[/green] {downloaded}")
        console.print(f"[yellow]Skipped:[/yellow]    {skipped}")
        console.print(f"[red]Failed:[/red]      {failed}")

        if failed > 0:
            failed_results = [r for r in results if r.status == "failed"]
            console.print(f"\n[dim]Failed files ({len(failed_results)}):[/dim]")
            for r in failed_results:
                div_name = ALL_DIVISIONS.get(r.division, EXTRA_LEAGUE_CODES.get(r.division, r.division))
                console.print(f"  [dim]{div_name} {r.season_code}: {r.error}[/dim]")

        return DownloadSummary(
            total=total,
            downloaded=downloaded,
            skipped=skipped,
            failed=failed,
            results=list(results),
        )
