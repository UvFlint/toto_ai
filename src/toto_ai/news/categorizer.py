from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from toto_ai.console import console
from toto_ai.news.models import MatchNewsAnalysis
from toto_ai.stats.models import MatchStats

_CATEGORIZER_AGENT = Agent(
    "google-gla:gemini-2.5-flash",
    output_type=MatchNewsAnalysis,
    system_prompt="""\
You are a football news analyst. Given raw news text about an upcoming match,
extract and categorize each distinct news item.

CATEGORIES (use exactly these values):
- star_player_absence: Top-3 player confirmed out (by goals, assists, or minutes played)
- goalkeeper_change: Starting goalkeeper unavailable
- manager_sacking: Manager fired or resigned this week
- mass_absence: 3+ first-team players unavailable simultaneously
- key_player_return: Important player returning from injury/suspension
- tactical_shift: Confirmed formation or system change from press conference
- motivation_shift: Change in competitive context (relegated, title clinched, must-win)
- transfer_disruption: Key player sold/transferred or major signing this week
- minor_injury: Rotation player or backup unavailable
- travel_weather: Extreme weather forecast or long-distance travel issues
- fan_atmosphere: Stadium ban, derby atmosphere, fan protests
- general_context: Background news with no direct outcome impact

RULES:
- Only include CONFIRMED news, not rumors or speculation
- Set is_post_odds=true if the news clearly emerged after the start of the week (Monday)
- For direction: 'positive' means it HELPS the affected_team, 'negative' means it HURTS them
- affected_team must be 'home', 'away', or 'both'
- confidence: 0.9+ for official club announcements, 0.6-0.8 for reputable journalists, 0.3-0.5 for rumors
- If no meaningful news exists, return an empty items list
- Do NOT fabricate news items — only extract what is present in the text
""",
)


async def categorize_all_news(
    stats: list[MatchStats],
    matches: list,
    *,
    max_concurrent: int = 4,
) -> None:
    """Categorize raw news text into structured MatchNewsAnalysis for each match."""
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(stats)
    completed_count = 0
    count_lock = asyncio.Lock()

    async def _categorize_one(idx: int) -> tuple[int, MatchNewsAnalysis | None]:
        nonlocal completed_count
        s = stats[idx]
        if not s.news:
            async with count_lock:
                completed_count += 1
            return idx, None

        async with semaphore:
            try:
                home = s.home_team_english or matches[idx].home_team
                away = s.away_team_english or matches[idx].away_team
                prompt = f"Match: {home} (home) vs {away} (away)\n\nRaw news:\n{s.news}"
                result = await _CATEGORIZER_AGENT.run(prompt)
                analysis = result.output
            except Exception as e:
                home = matches[idx].home_team
                away = matches[idx].away_team
                console.print(f"[yellow]Categorization failed for {home} vs {away}: {e}[/yellow]")
                analysis = None

        async with count_lock:
            completed_count += 1
            n = completed_count

        home = s.home_team_english or matches[idx].home_team
        away = s.away_team_english or matches[idx].away_team
        label = f"{home} vs {away}"
        if analysis and analysis.items:
            console.print(f"[dim]  [{n}/{total}] {label} — {len(analysis.items)} items[/dim]")
        else:
            console.print(f"[dim]  [{n}/{total}] {label} — no items[/dim]")

        return idx, analysis

    tasks = [_categorize_one(i) for i in range(total)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    categorized = 0
    for result in results:
        if isinstance(result, Exception):
            console.print(f"[yellow]Categorization task error: {result}[/yellow]")
            continue
        idx, analysis = result
        if analysis and analysis.items:
            stats[idx].news_analysis = analysis
            categorized += 1

    console.print(
        f"[green]News categorized: {categorized}/{total} matches have structured news[/green]"
    )
