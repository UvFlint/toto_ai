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
    column_type: Literal["ai", "statistical", "ml"] = "ai"


class StabilizedPrediction(BaseModel):
    """Stabilized prediction for one match from one model across N runs."""

    match_number: int
    stable_prediction: Literal["1", "X", "2"]
    stability: str = ""  # e.g. "3/3", "2/3"
    run_predictions: list[str] = Field(default_factory=list)  # e.g. ["1", "X", "1"]
    is_fallback: bool = False
    fallback_source: str | None = None  # "poisson" or "catboost"


class MatchDrawInfo(BaseModel):
    """Draw probability from the binary draw classifier for one match."""

    match_number: int
    draw_prob: float  # 0.0–1.0


class MatchNewsSnapshot(BaseModel):
    """News analysis snapshot saved with report for later review."""

    match_number: int
    home_team: str = ""
    away_team: str = ""
    has_x_factor: bool = False
    net_impact: float = 0.0
    post_odds_item_count: int = 0
    multiplier_used: float = 1.5


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
    news_snapshots: list[MatchNewsSnapshot] = Field(default_factory=list)
    draw_probs: list[MatchDrawInfo] = Field(default_factory=list)
    stabilize_runs: int | None = None
    run_columns: list[list[FullColumn]] = Field(default_factory=list)
    stabilized_predictions: dict[str, list[StabilizedPrediction]] = Field(
        default_factory=dict,
        description="model_name -> list of StabilizedPrediction per match",
    )

    @property
    def total_cost_usd(self) -> float:
        prediction_cost = sum(col.usage.cost_usd for col in self.columns)
        research_cost = sum(rc.usage.cost_usd for rc in self.research_costs)
        return prediction_cost + research_cost
