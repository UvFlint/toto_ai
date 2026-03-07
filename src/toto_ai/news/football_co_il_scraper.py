from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from toto_ai.console import console

_FEED_URL = "https://www.football.co.il/feed/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


async def fetch_football_co_il_news(
    team_name: str, client: httpx.AsyncClient
) -> list[str]:
    """Fetch recent articles mentioning *team_name* from football.co.il RSS feed.

    Returns up to 5 matching article titles. Returns an empty list on any error.
    """
    try:
        response = await client.get(
            _FEED_URL, headers=_HEADERS, timeout=10.0, follow_redirects=True
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)
        channel = root.find("channel")
        if channel is None:
            return []

        articles: list[str] = []
        for item in channel.findall("item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            title = title_el.text if title_el is not None and title_el.text else ""
            desc = desc_el.text if desc_el is not None and desc_el.text else ""

            if team_name in title or team_name in desc:
                entry = title
                if desc and desc != title:
                    # Trim long descriptions
                    short_desc = desc[:150].strip()
                    if len(desc) > 150:
                        short_desc += "..."
                    entry += f" — {short_desc}"
                articles.append(entry)

            if len(articles) >= 5:
                break

        return articles
    except Exception as exc:
        console.print(
            f"[dim]football.co.il fetch failed for '{team_name}': {exc}[/dim]"
        )
        return []
