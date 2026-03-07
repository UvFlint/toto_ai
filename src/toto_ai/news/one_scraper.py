from __future__ import annotations

import urllib.parse
from html.parser import HTMLParser

import httpx

from toto_ai.console import console

_ARTICLE_CLASSES = {"one-article-secondary", "one-article-plain"}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class _ArticleParser(HTMLParser):
    """Extracts article titles (h2) and snippets (h3) from one.co.il search results."""

    def __init__(self) -> None:
        super().__init__()
        self.articles: list[str] = []
        self._in_article = False
        self._depth = 0  # div nesting depth inside an article container
        self._capture_tag: str | None = None  # "h2" or "h3" when capturing text
        self._current_h2 = ""
        self._current_h3 = ""
        self._buf = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)

        if tag == "div":
            classes = set((attr_dict.get("class") or "").split())
            if classes & _ARTICLE_CLASSES:
                self._in_article = True
                self._depth = 0
                self._current_h2 = ""
                self._current_h3 = ""
            elif self._in_article:
                self._depth += 1

        if self._in_article and tag in ("h2", "h3"):
            self._capture_tag = tag
            self._buf = ""

    def handle_endtag(self, tag: str) -> None:
        if self._capture_tag and tag == self._capture_tag:
            text = self._buf.strip()
            if tag == "h2":
                self._current_h2 = text
            elif tag == "h3":
                self._current_h3 = text
            self._capture_tag = None
            self._buf = ""

        if self._in_article and tag == "div":
            if self._depth == 0:
                # Leaving the article container
                if self._current_h2:
                    entry = self._current_h2
                    if self._current_h3:
                        entry += f" — {self._current_h3}"
                    self.articles.append(entry)
                self._in_article = False
            else:
                self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._capture_tag:
            self._buf += data


async def fetch_one_news(team_name: str, client: httpx.AsyncClient) -> list[str]:
    """Fetch top article titles/snippets about *team_name* from one.co.il search.

    Returns up to 5 article strings. Returns an empty list on any error.
    """
    url = f"https://www.one.co.il/search/?q={urllib.parse.quote(team_name)}"
    try:
        response = await client.get(url, headers=_HEADERS, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        parser = _ArticleParser()
        parser.feed(response.text)
        return parser.articles[:5]
    except Exception as exc:
        console.print(f"[dim]one.co.il fetch failed for '{team_name}': {exc}[/dim]")
        return []
