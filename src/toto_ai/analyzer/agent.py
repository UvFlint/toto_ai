from __future__ import annotations

import asyncio
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from toto_ai.analyzer.models import (
    FullColumn,
    FullReport,
    ConsensusMatch,
    MatchPrediction,
    ModelCost,
)
from toto_ai.analyzer.prompts import MATCH_ANALYSIS_SYSTEM_PROMPT, build_match_data_prompt
from toto_ai.analyzer.tools import has_data_gaps
from toto_ai.scraper.models import Match
from toto_ai.stats.models import MatchStats
from toto_ai.console import console

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_DEEPSEEK_MODEL_ID = "deepseek-reasoner"

# Model configurations — (model_id, display_name)
# Use "deepseek:<model>" prefix to signal DeepSeek custom provider
MODELS_STANDARD = [
    ("openai:gpt-4o", "GPT-4o (gpt-4o)"),
    ("google-gla:gemini-2.5-flash", "Gemini Flash (gemini-2.5-flash)"),
    ("anthropic:claude-sonnet-4-6", "Claude Sonnet (claude-sonnet-4-6)"),
    (f"deepseek:{_DEEPSEEK_MODEL_ID}", f"DeepSeek ({_DEEPSEEK_MODEL_ID})"),
]

MODELS_PREMIUM = [
    ("openai:gpt-5.2", "GPT-5.2 (gpt-5.2)"),
    ("google-gla:gemini-3.1-pro-preview", "Gemini Pro (gemini-3.1-pro-preview)"),
    ("anthropic:claude-opus-4-6", "Claude Opus (claude-opus-4-6)"),
    (f"deepseek:{_DEEPSEEK_MODEL_ID}", f"DeepSeek ({_DEEPSEEK_MODEL_ID})"),
]

# Per-model extra settings passed to the agent call
_MODEL_SETTINGS: dict[str, dict] = {
    "claude-opus-4-6": {"thinking": {"type": "enabled", "budget_tokens": 16000}},
    "gpt-5.2": {"reasoning_effort": "high"},
    "gemini-3.1-pro-preview": {"google_thinking_config": {"thinking_budget": -1}},
}


def _resolve_model(model_id: str):
    """Return a pydantic-ai model string or object for the given model_id."""
    if model_id.startswith("deepseek:"):
        from toto_ai.config import settings

        deepseek_name = model_id[len("deepseek:") :]
        return OpenAIModel(
            deepseek_name,
            provider=OpenAIProvider(
                base_url=_DEEPSEEK_BASE_URL,
                api_key=settings.DEEPSEEK_API_KEY or "placeholder",
            ),
            profile=OpenAIModelProfile(openai_supports_tool_choice_required=False),
        )
    return model_id


def _model_settings_for(model_id: str) -> dict | None:
    """Return extra model_settings for models that need them (reasoning/thinking)."""
    for key, settings in _MODEL_SETTINGS.items():
        if key in model_id:
            return settings
    return None


def _create_agent(
    model_id: str, model_settings: dict | None = None
) -> Agent[None, list[MatchPrediction]]:
    """Create a pydantic-ai agent for match analysis."""
    model = _resolve_model(model_id)
    kwargs: dict = dict(
        output_type=list[MatchPrediction],
        system_prompt=MATCH_ANALYSIS_SYSTEM_PROMPT,
    )
    if model_settings:
        kwargs["model_settings"] = model_settings
    return Agent(model, **kwargs)


def _prepare_match_data(matches: list[Match], stats: list[MatchStats]) -> list[dict]:
    """Prepare match data dicts for the prompt builder."""
    stats_map = {(s.home_team, s.away_team): s for s in stats}

    data: list[dict] = []
    for match in matches:
        s = stats_map.get((match.home_team, match.away_team))
        entry: dict = {
            "match_number": match.match_number,
            "home_team": (s.home_team_english if s and s.home_team_english else match.home_team),
            "away_team": (s.away_team_english if s and s.away_team_english else match.away_team),
            "league": match.league,
        }
        if s:
            if s.h2h:
                entry["h2h"] = s.h2h.model_dump()
            if s.home_form:
                entry["home_form"] = s.home_form.model_dump()
            if s.away_form:
                entry["away_form"] = s.away_form.model_dump()
            if s.home_standing:
                entry["home_standing"] = s.home_standing.model_dump()
            if s.away_standing:
                entry["away_standing"] = s.away_standing.model_dump()
            if s.news:
                entry["news"] = s.news
            if s.match_date:
                entry["match_date"] = s.match_date.strftime("%Y-%m-%d")
            if s.home_injuries:
                entry["home_injuries"] = [inj.model_dump() for inj in s.home_injuries]
            if s.away_injuries:
                entry["away_injuries"] = [inj.model_dump() for inj in s.away_injuries]
            if s.odds:
                entry["odds"] = s.odds.model_dump()
            if s.referee:
                entry["referee"] = s.referee
            if s.home_rest_days is not None:
                entry["home_rest_days"] = s.home_rest_days
            if s.away_rest_days is not None:
                entry["away_rest_days"] = s.away_rest_days
            if s.home_avg_days_between is not None:
                entry["home_avg_days_between"] = s.home_avg_days_between
            if s.away_avg_days_between is not None:
                entry["away_avg_days_between"] = s.away_avg_days_between
            if s.home_xg:
                entry["home_xg"] = s.home_xg.model_dump()
            if s.away_xg:
                entry["away_xg"] = s.away_xg.model_dump()
            if s.poisson_probs:
                entry["poisson_probs"] = s.poisson_probs.model_dump()
        data.append(entry)
    return data


async def _run_single_model(
    model_id: str,
    model_name: str,
    prompt: str,
) -> FullColumn:
    """Run analysis with a single AI model."""
    console.print(f"[blue]Running analysis with {model_name}...[/blue]")
    try:
        agent = _create_agent(model_id, model_settings=_model_settings_for(model_id))
        result = await agent.run(prompt)
        predictions = result.output

        run_usage = result.usage()
        cost_usd = 0.0
        try:
            price = result.response.cost()
            cost_usd = float(price.total_price)
        except Exception:
            pass  # model not in genai-prices DB

        console.print(f"[green]{model_name} completed: {len(predictions)} predictions[/green]")
        return FullColumn(
            model_name=model_name,
            predictions=predictions,
            usage=ModelCost(
                input_tokens=run_usage.input_tokens,
                output_tokens=run_usage.output_tokens,
                cost_usd=cost_usd,
            ),
        )
    except Exception as e:
        console.print(f"[red]{model_name} failed: {e}[/red]")
        return FullColumn(model_name=model_name, predictions=[])


def _build_consensus(columns: list[FullColumn], matches: list[Match]) -> list[ConsensusMatch]:
    """Build consensus from multiple model predictions."""
    from collections import Counter

    total_models = len(columns)
    consensus: list[ConsensusMatch] = []

    for match in matches:
        predictions_for_match: list[str] = []
        for col in columns:
            for pred in col.predictions:
                if pred.match_number == match.match_number:
                    predictions_for_match.append(pred.prediction)
                    break

        # Find majority prediction
        # With 1 model: use its prediction directly
        # With 2-3 models: require at least 2 to agree
        if predictions_for_match:
            counter = Counter(predictions_for_match)
            most_common = counter.most_common(1)[0]
            min_agreement = 1 if total_models == 1 else 2
            consensus_pred = most_common[0] if most_common[1] >= min_agreement else None
            agreement = most_common[1]
        else:
            consensus_pred = None
            agreement = 0

        consensus.append(
            ConsensusMatch(
                match_number=match.match_number,
                home_team=match.home_team,
                away_team=match.away_team,
                prediction=consensus_pred,
                agreement_count=agreement,
            )
        )

    return consensus


def _find_bankers_and_upsets(
    columns: list[FullColumn], consensus: list[ConsensusMatch]
) -> tuple[list[int], list[int]]:
    """Identify banker picks (high agreement + confidence) and upset alerts."""
    bankers: list[int] = []
    upsets: list[int] = []

    total_models = len(columns)
    for cm in consensus:
        if cm.agreement_count == total_models:
            # All models agree - check average confidence
            confidences: list[float] = []
            for col in columns:
                for pred in col.predictions:
                    if pred.match_number == cm.match_number:
                        confidences.append(pred.confidence)
                        break
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            if avg_conf >= 0.7:
                bankers.append(cm.match_number)
        elif cm.agreement_count <= 1:
            # No agreement - upset alert
            upsets.append(cm.match_number)

    return bankers, upsets


async def analyze_matches(
    matches: list[Match],
    stats: list[MatchStats],
    premium: bool = False,
    reference_date: datetime | None = None,
) -> FullReport:
    """Run all three AI models in parallel and build the report."""
    models = MODELS_PREMIUM if premium else MODELS_STANDARD
    match_data = _prepare_match_data(matches, stats)

    gaps = has_data_gaps(match_data)
    prompt = build_match_data_prompt(match_data, flag_gaps=gaps, reference_date=reference_date)

    # Run all models in parallel
    tasks = [_run_single_model(model_id, model_name, prompt) for model_id, model_name in models]
    columns = await asyncio.gather(*tasks)
    columns = [c for c in columns if c.predictions]  # Filter out failed models

    # Add Poisson/Dixon-Coles statistical model as a voting column
    from toto_ai.stats.poisson import build_poisson_column

    poisson_col = build_poisson_column(matches, stats)
    if poisson_col.predictions:
        columns.append(poisson_col)

    if not columns:
        console.print("[red]All models failed. Cannot produce report.[/red]")
        return FullReport()

    # Build consensus
    consensus = _build_consensus(columns, matches)
    bankers, upsets = _find_bankers_and_upsets(columns, consensus)

    return FullReport(
        columns=list(columns),
        consensus=consensus,
        banker_picks=bankers,
        upset_alerts=upsets,
    )
