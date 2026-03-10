from __future__ import annotations

import asyncio
from datetime import datetime

from pydantic_ai import Agent

from toto_ai.console import console
from toto_ai.news.models import MatchNewsAnalysis
from toto_ai.stats.models import MatchStats

_CATEGORIZER_SYSTEM_PROMPT = """\
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
- POST-ODDS DATE: {post_odds_cutoff}. Set is_post_odds=true if the news emerged AFTER this date.
  This means the bookmaker could NOT have priced this information into the odds — treat it with
  higher importance. If no date can be inferred for a news item, use your best judgment.
- For direction: 'positive' means it HELPS the affected_team, 'negative' means it HURTS them
- affected_team must be 'home', 'away', or 'both'
- confidence: 0.9+ for official club announcements, 0.6-0.8 for reputable journalists, 0.3-0.5 for rumors
- If no meaningful news exists, return an empty items list
- Do NOT fabricate news items — only extract what is present in the text
"""

_CATEGORIZER_AGENT_NO_DATE = Agent(
    "google-gla:gemini-2.5-flash",
    output_type=MatchNewsAnalysis,
    system_prompt=_CATEGORIZER_SYSTEM_PROMPT.format(
        post_odds_cutoff="start of the current week (Monday)"
    ),
)


def _make_categorizer_agent(odds_date: datetime | None) -> Agent:
    """Create a categorizer agent with the correct post-odds cutoff date."""
    if odds_date is None:
        return _CATEGORIZER_AGENT_NO_DATE
    cutoff_str = odds_date.strftime("%A %Y-%m-%d")
    system_prompt = _CATEGORIZER_SYSTEM_PROMPT.format(post_odds_cutoff=cutoff_str)
    return Agent("google-gla:gemini-2.5-flash", output_type=MatchNewsAnalysis, system_prompt=system_prompt)


async def categorize_all_news(
    stats: list[MatchStats],
    matches: list,
    *,
    max_concurrent: int = 4,
    odds_date: datetime | None = None,
) -> None:
    """Categorize raw news text into structured MatchNewsAnalysis for each match.

    Args:
        odds_date: The date when the betting form was published (odds were set).
                   News after this date is flagged is_post_odds=True and weighted higher.
    """
    agent = _make_categorizer_agent(odds_date)
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
                result = await agent.run(prompt)
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
