from __future__ import annotations


def has_data_gaps(match_data: list[dict]) -> bool:
    """Return True if any match is missing h2h, home_form, or away_form."""
    return any(
        not m.get("h2h") or not m.get("home_form") or not m.get("away_form") for m in match_data
    )
