from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class MatchReviewRecord(BaseModel):
    """Per-match review data within a weekly review."""

    match_number: int
    home_team: str
    away_team: str
    predictions: dict[str, str] = Field(description="model_name -> '1'/'X'/'2'")
    consensus: str | None = None
    actual_result: str
    had_news_analysis: bool = False
    has_x_factor: bool = False
    net_impact: float = 0.0
    post_odds_item_count: int = 0
    multiplier_used: float = 1.5


class WeeklyReview(BaseModel):
    """One week's review record."""

    form_number: str
    reviewed_at: datetime
    multiplier_before: float
    multiplier_after: float
    matches: list[MatchReviewRecord] = []
    # Aggregate metrics
    total_matches: int = 0
    consensus_correct: int = 0
    consensus_accuracy: float = 0.0
    # X-factor specific
    xfactor_matches: int = 0
    xfactor_correct: int = 0
    xfactor_accuracy: float = 0.0
    # Non-x-factor
    non_xfactor_matches: int = 0
    non_xfactor_correct: int = 0
    non_xfactor_accuracy: float = 0.0


class CalibrationData(BaseModel):
    """Root model for calibration.json."""

    post_odds_multiplier: float = 1.5
    reviews: list[WeeklyReview] = []


_MIN_XFACTOR_SAMPLE = 3
_LIFT_THRESHOLD = 0.05
_ADJUSTMENT_STEP = 0.1
_MULTIPLIER_MIN = 1.0
_MULTIPLIER_MAX = 3.0
_ROLLING_WINDOW = 4


class CalibrationManager:
    """Manages calibration data and multiplier adjustments."""

    def __init__(self, path: Path | str = "data/calibration.json") -> None:
        self.path = Path(path)

    def _load(self) -> CalibrationData:
        if not self.path.exists():
            return CalibrationData()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return CalibrationData.model_validate(data)

    def _save(self, data: CalibrationData) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def post_odds_multiplier(self) -> float:
        return self._load().post_odds_multiplier

    def compute_new_multiplier(self, current_review: WeeklyReview) -> float:
        """Compute adjusted multiplier using rolling 4-week window."""
        data = self._load()
        # Build window: existing reviews + current
        all_reviews = data.reviews + [current_review]
        window = all_reviews[-_ROLLING_WINDOW:]

        # Pool x-factor and non-x-factor metrics across the window
        xf_total = sum(r.xfactor_matches for r in window)
        xf_correct = sum(r.xfactor_correct for r in window)
        nxf_total = sum(r.non_xfactor_matches for r in window)
        nxf_correct = sum(r.non_xfactor_correct for r in window)

        current = data.post_odds_multiplier

        if xf_total < _MIN_XFACTOR_SAMPLE:
            return current

        xf_acc = xf_correct / xf_total
        nxf_acc = nxf_correct / nxf_total if nxf_total > 0 else 0.0
        lift = xf_acc - nxf_acc

        if lift > _LIFT_THRESHOLD:
            new = current + _ADJUSTMENT_STEP
        elif lift < -_LIFT_THRESHOLD:
            new = current - _ADJUSTMENT_STEP
        else:
            new = current

        return max(_MULTIPLIER_MIN, min(_MULTIPLIER_MAX, round(new, 1)))

    def record_review(self, review: WeeklyReview) -> None:
        """Append a review and update the multiplier."""
        data = self._load()
        data.reviews.append(review)
        data.post_odds_multiplier = review.multiplier_after
        self._save(data)
