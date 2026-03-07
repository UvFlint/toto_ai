from __future__ import annotations

import asyncio

import httpx

from toto_ai.console import console


class ApiFootballClient:
    """Async HTTP client for API-Football v3 with retry and rate-limit tracking."""

    def __init__(
        self,
        api_key: str,
        host: str = "v3.football.api-sports.io",
        use_rapidapi: bool = False,
        max_concurrent: int = 3,
    ) -> None:
        self._api_key = api_key
        self._host = host
        self._use_rapidapi = use_rapidapi
        self._base_url = f"https://{host}"
        self._remaining: int | None = None
        self._call_count = 0
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _headers(self) -> dict[str, str]:
        if self._use_rapidapi:
            return {
                "x-rapidapi-key": self._api_key,
                "x-rapidapi-host": self._host,
            }
        return {"x-apisports-key": self._api_key}

    @property
    def call_count(self) -> int:
        return self._call_count

    async def get(
        self,
        endpoint: str,
        params: dict | None = None,
        *,
        client: httpx.AsyncClient,
        max_retries: int = 5,
    ) -> dict:
        """Make a GET request to API-Football with retry on 429/5xx.

        Uses a semaphore to limit concurrent requests and avoid burst rate limits.
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers = self._headers()

        for attempt in range(max_retries + 1):
            try:
                async with self._semaphore:
                    resp = await client.get(url, params=params, headers=headers)

                # Track rate limits from response headers
                remaining = resp.headers.get("x-ratelimit-requests-remaining")
                if remaining is not None:
                    self._remaining = int(remaining)
                    if self._remaining <= 5:
                        console.print(
                            f"[yellow]API-Football rate limit warning: "
                            f"{self._remaining} requests remaining[/yellow]"
                        )

                if resp.status_code == 429:
                    if attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        console.print(f"[yellow]Rate limited. Retrying in {wait}s...[/yellow]")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()

                if resp.status_code >= 500:
                    if attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                self._call_count += 1
                data = resp.json()

                # Handle in-body rate limit errors (API returns 200 but empty response)
                errors = data.get("errors")
                if errors and ("rateLimit" in errors or "requests" in errors):
                    if attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        console.print(
                            f"[yellow]Rate limited ({endpoint}). "
                            f"Retrying in {wait}s...[/yellow]"
                        )
                        await asyncio.sleep(wait)
                        continue
                    console.print(
                        f"[yellow]Rate limit exhausted for {endpoint} "
                        f"after {max_retries} retries[/yellow]"
                    )
                elif errors:
                    console.print(
                        f"[yellow]API-Football error for {endpoint} "
                        f"params={params}: {errors}[/yellow]"
                    )

                return data

            except httpx.TimeoutException:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                raise

        raise RuntimeError(f"API-Football request failed after {max_retries} retries: {endpoint}")
