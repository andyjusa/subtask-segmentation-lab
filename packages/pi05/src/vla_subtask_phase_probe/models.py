from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSignal:
    """A phase transition represented by the OpenArmSciEdu Task API contract."""

    task_id: str
    task_sub_id: str
    phase: str

    def payload(self) -> dict[str, str]:
        return {"task_id": self.task_id, "task_sub_id": self.task_sub_id}
