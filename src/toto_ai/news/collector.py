from __future__ import annotations

import asyncio

import httpx
from pydantic_ai import Agent, WebSearchTool

from toto_ai.console import console
from toto_ai.news.football_co_il_scraper import fetch_football_co_il_news
from toto_ai.news.one_scraper import fetch_one_news
from toto_ai.stats.models import MatchStats

_ISRAELI_LEAGUE_NAMES = {"ליגת העל", "ליגה לאומית", "גביע המדינה", "גביע הטוטו"}

_NEWS_AGENT = Agent(
    "google-gla:gemini-3-flash-preview",
    builtin_tools=[WebSearchTool()],
    system_prompt=(
        "You are a football news researcher. "
        "Search for relevant pre-match news and provide concise, accurate summaries. "
        "Always respond in English."
    ),
)

_SEARCH_PROMPT = (
    "Search for the latest news about the upcoming football match: "
    "{home} vs {away}. "
    "Focus on: key injuries & suspensions, recent team news, "
    "managerial changes, motivation factors (title race, relegation battle), "
    "and fixture congestion. "
    "Provide a concise summary of the most relevant findings in 3-5 bullet points. "
    "Only include confirmed, recent information."
)


async def _search_match_news(
    home: str,
    away: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Use Gemini with WebSearchTool to find news for a single match."""
    async with semaphore:
        try:
            prompt = _SEARCH_PROMPT.format(home=home, away=away)
            result = await _NEWS_AGENT.run(prompt)
            return result.output
        except Exception as e:
            console.print(f"[yellow]Web search failed for {home} vs {away}: {e}[/yellow]")
            return ""


async def gather_news(
    stats: list[MatchStats],
    matches: list,
    *,
    max_concurrent: int = 4,
) -> None:
    """Gather news for all matches and populate stats[i].news.

    Sources:
    1. Gemini WebSearchTool (all matches)
    2. one.co.il + football.co.il (Israeli teams only)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # Phase 1: Web search for all matches in parallel
    console.print("[dim]Searching web for match news (Gemini)...[/dim]")
    total = len(stats)
    completed_count = 0
    count_lock = asyncio.Lock()

    async def _web_search_task(idx: int) -> tuple[int, str]:
        nonlocal completed_count
        s = stats[idx]
        home = s.home_team_english or matches[idx].home_team
        away = s.away_team_english or matches[idx].away_team
        news = await _search_match_news(home, away, semaphore)
        async with count_lock:
            completed_count += 1
            n = completed_count
        label = f"{home} vs {away}"
        if news:
            console.print(f"[dim]  [{n}/{total}] {label} — OK[/dim]")
        else:
            console.print(f"[yellow]  [{n}/{total}] {label} — FAILED[/yellow]")
        return idx, news

    web_tasks = [_web_search_task(i) for i in range(len(stats))]
    web_results = await asyncio.gather(*web_tasks, return_exceptions=True)

    web_success = 0
    for result in web_results:
        if isinstance(result, Exception):
            console.print(f"[yellow]News task error: {result}[/yellow]")
            continue
        idx, news_text = result
        if news_text:
            stats[idx].news = news_text
            web_success += 1

    console.print(f"[dim]Web search complete: {web_success}/{total} matches[/dim]")

    # Phase 2: Israeli sources (one.co.il + football.co.il) for Israeli teams only
    israeli_indices: list[int] = [
        i for i in range(len(stats)) if matches[i].league in _ISRAELI_LEAGUE_NAMES
    ]

    if israeli_indices:
        console.print(
            f"[dim]Fetching Israeli news for {len(israeli_indices)} matches "
            f"(one.co.il + football.co.il)...[/dim]"
        )
        il_total = len(israeli_indices)
        async with httpx.AsyncClient() as client:
            for il_n, i in enumerate(israeli_indices, 1):
                home_name = matches[i].home_team
                away_name = matches[i].away_team

                one_home = await fetch_one_news(home_name, client)
                one_away = await fetch_one_news(away_name, client)
                fc_home = await fetch_football_co_il_news(home_name, client)
                fc_away = await fetch_football_co_il_news(away_name, client)

                israeli_articles: list[str] = []
                if one_home or one_away:
                    israeli_articles.append("[one.co.il]")
                    israeli_articles.extend(f"- {a}" for a in one_home + one_away)
                if fc_home or fc_away:
                    israeli_articles.append("[football.co.il]")
                    israeli_articles.extend(f"- {a}" for a in fc_home + fc_away)

                if israeli_articles:
                    israeli_section = "\n\n[Israeli sources]\n" + "\n".join(israeli_articles)
                    stats[i].news = (stats[i].news or "") + israeli_section

                article_count = len(one_home) + len(one_away) + len(fc_home) + len(fc_away)
                label = f"{home_name} vs {away_name}"
                if article_count:
                    console.print(
                        f"[dim]  [{il_n}/{il_total}] {label} — {article_count} articles[/dim]"
                    )
                else:
                    console.print(f"[yellow]  [{il_n}/{il_total}] {label} — no articles[/yellow]")

    total_with_news = sum(1 for s in stats if s.news)
    console.print(
        f"[green]News gathering complete: {total_with_news}/{len(stats)} matches have news[/green]"
    )
