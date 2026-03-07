from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class SubmissionRecord(BaseModel):
    form_number: str
    deadline: datetime | None = None
    submitted_at: datetime
    columns_submitted: int = 0


class SubmissionTracker:
    """Tracks which Winner 16 forms have already been submitted."""

    def __init__(self, path: Path | str = "data/submissions.json") -> None:
        self.path = Path(path)

    def _load(self) -> list[SubmissionRecord]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [SubmissionRecord.model_validate(r) for r in data.get("submissions", [])]

    def _save(self, records: list[SubmissionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"submissions": [r.model_dump(mode="json") for r in records]}
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def is_submitted(self, form_number: str) -> bool:
        return any(r.form_number == form_number for r in self._load())

    def record_submission(
        self,
        form_number: str,
        deadline: datetime | None = None,
        columns_count: int = 0,
    ) -> None:
        records = self._load()
        records.append(
            SubmissionRecord(
                form_number=form_number,
                deadline=deadline,
                submitted_at=datetime.now(timezone.utc),
                columns_submitted=columns_count,
            )
        )
        self._save(records)

    def mark_as_submitted(self, form_number: str, deadline: datetime | None = None) -> None:
        """Manually mark a form as submitted (e.g., to skip this week)."""
        self.record_submission(form_number, deadline=deadline, columns_count=0)
