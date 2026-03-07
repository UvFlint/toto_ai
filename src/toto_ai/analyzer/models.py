from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelCost(BaseModel):
    """Token usage and estimated cost for a single model run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class MatchPrediction(BaseModel):
    match_number: int
    home_team: str
    away_team: str
    prediction: Literal["1", "X", "2"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level 0-1")
    reasoning: str = Field(description="Brief explanation for the prediction")
    key_factors: list[str] = Field(
        default_factory=list,
        description="Top factors driving this prediction",
    )


class FullColumn(BaseModel):
    """A complete column of 16 predictions from one AI model."""

    model_name: str
    predictions: list[MatchPrediction] = Field(description="Exactly 16 match predictions")
    usage: ModelCost = Field(default_factory=ModelCost)


class ConsensusMatch(BaseModel):
    match_number: int
    home_team: str
    away_team: str
    prediction: Literal["1", "X", "2"] | None = None
    agreement_count: int = 0  # How many models agree


class ResearchCost(BaseModel):
    """Cost entry for a research model (news, stats, etc.)."""

    label: str
    usage: ModelCost = Field(default_factory=ModelCost)


class FullReport(BaseModel):
    """Complete analysis report with all three model columns."""

    form_number: str | None = None
    columns: list[FullColumn] = []
    external_column: FullColumn | None = None
    consensus: list[ConsensusMatch] = []
    research_costs: list[ResearchCost] = Field(default_factory=list)
    banker_picks: list[int] = Field(
        default_factory=list,
        description="Match numbers where all 3 models agree with high confidence",
    )
    upset_alerts: list[int] = Field(
        default_factory=list,
        description="Match numbers with high disagreement between models",
    )

    @property
    def total_cost_usd(self) -> float:
        prediction_cost = sum(col.usage.cost_usd for col in self.columns)
        research_cost = sum(rc.usage.cost_usd for rc in self.research_costs)
        return prediction_cost + research_cost
