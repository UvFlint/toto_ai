from __future__ import annotations

from datetime import datetime

MATCH_ANALYSIS_SYSTEM_PROMPT = """\
You are a professional football betting analyst specializing in the Israeli Winner 16 \
(Toto) game. Your task is to predict the outcomes of 16 football matches.

## Winner 16 Rules
- For each match, predict: 1 (home team wins), X (draw), 2 (away team wins)
- Results are based on 90 MINUTES OF REGULAR TIME ONLY (no extra time, no penalties)
- This means cup matches and knockout games are judged on 90-minute result only

## Analysis Framework
For each match, consider these factors in order of importance:

1. **Recent Form** (most important): Last 5 matches performance, goals scored/conceded
2. **Home/Away Advantage**: Home teams statistically win ~45% of matches
3. **Head-to-Head Record**: Historical matchups between these teams
4. **League Position Gap**: Larger gaps suggest clearer favorites
5. **News & Context**: Injuries, suspensions, transfers, managerial changes
6. **Motivation**: Title race, relegation battle, nothing to play for
7. **Fixture Congestion**: Midweek games, rotation risk
8. **Injuries/Suspensions**: Missing key players significantly affect match outcomes

## Important Guidelines
- Be HONEST about uncertainty. Don't force a prediction if the match is truly unpredictable
- DRAWS are valuable in Toto - don't shy away from predicting X when appropriate
- Consider that ~25-30% of football matches end in draws
- Lower league matches have higher draw rates
- Derby matches are often tighter than expected
- Don't be biased toward big teams - form matters more than reputation

## Output
Provide your predictions as a structured list of 16 MatchPrediction objects.
Each prediction must include:
- match_number (1-16)
- home_team and away_team names
- prediction ("1", "X", or "2")
- confidence (0.0 to 1.0)
- reasoning (1-2 sentences explaining your pick)
- key_factors (top 2-3 factors)
- Always respond in English regardless of the language of team names in the input.
"""


def build_match_data_prompt(
    matches_data: list[dict],
    flag_gaps: bool = False,
    reference_date: datetime | None = None,
) -> str:
    """Build the user prompt with all match data for analysis."""
    sections: list[str] = []
    sections.append("# Winner 16 Form - Match Analysis Request\n")
    if reference_date:
        date_str = reference_date.strftime("%B %d, %Y")
        sections.append(
            f"**BACKTEST MODE**: All data below is from before {date_str}. "
            f"Do NOT use knowledge of events after this date in your analysis.\n"
        )
    sections.append("Analyze the following 16 matches and predict each outcome.\n")

    for match in matches_data:
        mn = match["match_number"]
        section = f"\n## Match {mn}: {match['home_team']} vs {match['away_team']}\n"

        if match.get("league"):
            section += f"**League:** {match['league']}\n"

        if match.get("h2h"):
            h2h = match["h2h"]
            section += f"\n### Head-to-Head (last {len(h2h.get('matches', []))} meetings)\n"
            section += f"- Home wins: {h2h.get('home_wins', 'N/A')}\n"
            section += f"- Draws: {h2h.get('draws', 'N/A')}\n"
            section += f"- Away wins: {h2h.get('away_wins', 'N/A')}\n"
        elif flag_gaps:
            section += "\n### Head-to-Head\n"
            section += "[Data not available]\n"

        if match.get("home_form"):
            hf = match["home_form"]
            section += f"\n### {match['home_team']} Recent Form\n"
            section += f"- Form: {hf.get('form_string', 'N/A')} (last 5)\n"
            section += (
                f"- Record: {hf.get('wins', 0)}W {hf.get('draws', 0)}D {hf.get('losses', 0)}L\n"
            )
            section += (
                f"- Goals: {hf.get('goals_for', 0)} scored, {hf.get('goals_against', 0)} conceded\n"
            )
        elif flag_gaps:
            section += f"\n### {match['home_team']} Recent Form\n"
            section += "[Data not available]\n"

        if match.get("away_form"):
            af = match["away_form"]
            section += f"\n### {match['away_team']} Recent Form\n"
            section += f"- Form: {af.get('form_string', 'N/A')} (last 5)\n"
            section += (
                f"- Record: {af.get('wins', 0)}W {af.get('draws', 0)}D {af.get('losses', 0)}L\n"
            )
            section += (
                f"- Goals: {af.get('goals_for', 0)} scored, {af.get('goals_against', 0)} conceded\n"
            )
        elif flag_gaps:
            section += f"\n### {match['away_team']} Recent Form\n"
            section += "[Data not available]\n"

        if match.get("home_standing") or match.get("away_standing"):
            section += "\n### League Standings\n"
            if match.get("home_standing"):
                hs = match["home_standing"]
                section += f"- {match['home_team']}: #{hs.get('position', '?')} ({hs.get('points', '?')} pts)\n"
            if match.get("away_standing"):
                as_ = match["away_standing"]
                section += f"- {match['away_team']}: #{as_.get('position', '?')} ({as_.get('points', '?')} pts)\n"

        if match.get("match_date"):
            section += f"\n**Match Date:** {match['match_date']}\n"

        if match.get("home_injuries") or match.get("away_injuries"):
            section += "\n### Injuries & Suspensions\n"
            for inj in match.get("home_injuries", []):
                section += f"- [{match['home_team']}] {inj.get('player_name', '?')} — {inj.get('type', '?')}: {inj.get('reason', 'N/A')}\n"
            for inj in match.get("away_injuries", []):
                section += f"- [{match['away_team']}] {inj.get('player_name', '?')} — {inj.get('type', '?')}: {inj.get('reason', 'N/A')}\n"

        if match.get("odds"):
            od = match["odds"]
            section += "\n### Betting Odds\n"
            section += f"- {od.get('bookmaker', 'N/A')}: 1={od.get('home_odds', 'N/A')} X={od.get('draw_odds', 'N/A')} 2={od.get('away_odds', 'N/A')}\n"

        if (
            match.get("news")
            and match["news"]
            != f"News unavailable for {match['home_team']} vs {match['away_team']}"
        ):
            section += f"\n### Recent News\n{match['news']}\n"

        sections.append(section)

    sections.append(
        "\n---\nNow provide your predictions for all 16 matches. "
        "Remember: results based on 90 minutes only."
    )
    return "\n".join(sections)
