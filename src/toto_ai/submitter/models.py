from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SubmissionColumn(BaseModel):
    """A unique column of 16 predictions to submit."""

    column_index: int
    source_models: list[str]
    predictions: list[Literal["1", "X", "2"]]


class SubmissionResult(BaseModel):
    """Result of submitting a single column."""

    column_index: int
    success: bool
    message: str
