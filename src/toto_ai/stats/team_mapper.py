from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import NamedTuple

import httpx

from toto_ai.console import console

# Hebrew prefix → English transliteration for API search fallback
_HEBREW_PREFIXES: dict[str, str] = {
    "הפועל": "Hapoel",
    "מכבי": "Maccabi",
    'בית"ר': "Beitar",
    "ביתר": "Beitar",
    "עירוני": "Ironi",
    "בני": "Bnei",
    "מ.ס.": "MS",
    "שמשון": "Shimshon",
}


class TeamInfo(NamedTuple):
    v3_id: int
    english_name: str
    league_id: int | None


def _pick_best_league(league_results: list[dict]) -> int | None:
    """Pick the best domestic league ID from API-Football /leagues response.

    Prefers leagues with type="League" (domestic) over cups/friendlies,
    and filters for current season.
    """
    candidates: list[dict] = []
    for entry in league_results:
        league = entry.get("league", {})
        league_type = league.get("type", "")
        seasons = entry.get("seasons", [])
        # Check if the league has a current season
        is_current = any(s.get("current", False) for s in seasons)
        if is_current and league_type == "League":
            candidates.append(league)

    if not candidates:
        # Fallback: any current league regardless of type
        for entry in league_results:
            league = entry.get("league", {})
            seasons = entry.get("seasons", [])
            if any(s.get("current", False) for s in seasons):
                candidates.append(league)

    if not candidates:
        return None

    # If multiple domestic leagues, pick the first one (API usually returns
    # the primary league first)
    return candidates[0].get("id")


class TeamMapper:
    """Resolve Hebrew team names to API-Football v3 team IDs."""

    def __init__(self, cache_path: str = "data/team_cache.json") -> None:
        self._cache_path = Path(cache_path)
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                console.print(f"[yellow]Warning: failed to load team cache: {e}[/yellow]")
                self._cache = {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def resolve_cached(self, hebrew_name: str) -> TeamInfo | None:
        """Synchronous cache-only lookup (exact + fuzzy)."""
        # Exact match
        entry = self._cache.get(hebrew_name)
        if entry:
            return TeamInfo(
                v3_id=entry["v3_id"],
                english_name=entry["name"],
                league_id=entry.get("league_id"),
            )

        # Fuzzy match against cached Hebrew keys
        matches = difflib.get_close_matches(hebrew_name, self._cache.keys(), n=1, cutoff=0.7)
        if matches:
            entry = self._cache[matches[0]]
            console.print(
                f"[dim]Fuzzy match: '{hebrew_name}' → '{matches[0]}' ({entry['name']})[/dim]"
            )
            return TeamInfo(
                v3_id=entry["v3_id"],
                english_name=entry["name"],
                league_id=entry.get("league_id"),
            )

        return None

    async def resolve(
        self,
        hebrew_name: str,
        client: httpx.AsyncClient,
        *,
        api_headers: dict[str, str],
        base_url: str,
    ) -> TeamInfo | None:
        """Resolve Hebrew name → TeamInfo. Tries cache, then API search."""
        # Try cache first
        cached = self.resolve_cached(hebrew_name)
        if cached:
            return cached

        # Transliterate and search via API
        english_guess = self._transliterate(hebrew_name)
        if not english_guess:
            console.print(f"[yellow]Cannot transliterate team name: '{hebrew_name}'[/yellow]")
            return None

        console.print(f"[dim]Searching API-Football for '{english_guess}'...[/dim]")
        try:
            resp = await client.get(
                f"{base_url}/teams",
                params={"search": english_guess},
                headers=api_headers,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("response", [])

            if not results:
                console.print(
                    f"[yellow]No API results for '{english_guess}' "
                    f"(Hebrew: '{hebrew_name}')[/yellow]"
                )
                return None

            # Pick the first result
            team = results[0]["team"]
            team_id = team["id"]

            # Try to discover league for this team
            league_id = await self.refresh_league_id(
                team_id, client, api_headers=api_headers, base_url=base_url
            )

            team_info = TeamInfo(
                v3_id=team_id,
                english_name=team["name"],
                league_id=league_id,
            )

            # Auto-persist to cache
            self._cache[hebrew_name] = {
                "v3_id": team_info.v3_id,
                "name": team_info.english_name,
                "league_id": team_info.league_id,
            }
            self._save_cache()
            console.print(
                f"[green]Resolved '{hebrew_name}' → {team_info.english_name} "
                f"(ID: {team_info.v3_id})[/green]"
            )
            return team_info

        except Exception as e:
            console.print(f"[yellow]API team search failed for '{hebrew_name}': {e}[/yellow]")
            return None

    def update_league_id(self, hebrew_name: str, league_id: int) -> None:
        """Update a cached team's league_id and persist."""
        if hebrew_name in self._cache:
            self._cache[hebrew_name]["league_id"] = league_id
            self._save_cache()

    async def refresh_league_id(
        self,
        team_id: int,
        client: httpx.AsyncClient,
        *,
        api_headers: dict[str, str],
        base_url: str,
    ) -> int | None:
        """Fetch current league_id for a team from the API."""
        try:
            resp = await client.get(
                f"{base_url}/leagues",
                params={"team": team_id, "current": "true"},
                headers=api_headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return _pick_best_league(data.get("response", []))
        except Exception as e:
            console.print(f"[dim]League refresh failed for team {team_id}: {e}[/dim]")
            return None

    def _transliterate(self, hebrew_name: str) -> str | None:
        """Best-effort transliteration of Hebrew team name to English."""
        name = hebrew_name.strip()

        # Try known prefix patterns
        for heb_prefix, eng_prefix in _HEBREW_PREFIXES.items():
            if name.startswith(heb_prefix):
                suffix = name[len(heb_prefix) :].strip()
                if suffix:
                    # Return prefix + transliterated suffix as-is
                    # (the suffix is usually a city name that the API can fuzzy-match)
                    return f"{eng_prefix} {suffix}"
                return eng_prefix

        # No known prefix — return the Hebrew name as-is for the API to try
        # (API-Football sometimes handles non-Latin search terms)
        return hebrew_name
