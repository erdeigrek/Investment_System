from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ProgressDict = Dict[str, Dict[str, Any]]
# struktura:
# {
#   "E0_T1": {"status": "done", "note": "...", "updated_at": "..." },
#   ...
# }


@dataclass(frozen=True)
class ProgressStore:
    path: Path

    def load(self) -> ProgressDict:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data  # type: ignore[return-value]

    def save(self, progress: ProgressDict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def get_status(self, task_id: str, default: str = "todo") -> str:
        p = self.load()
        entry = p.get(task_id, {})
        status = entry.get("status", default)
        if status not in {"todo", "in_progress", "done"}:
            return default
        return status

    def set_status(self, task_id: str, status: str) -> None:
        if status not in {"todo", "in_progress", "done"}:
            raise ValueError(f"Nieznany status: {status}")
        p = self.load()
        entry = p.get(task_id, {})
        entry["status"] = status
        p[task_id] = entry
        self.save(p)

    def set_note(self, task_id: str, note: str) -> None:
        p = self.load()
        entry = p.get(task_id, {})
        entry["note"] = note
        p[task_id] = entry
        self.save(p)

    def get_note(self, task_id: str, default: str = "") -> str:
        p = self.load()
        entry = p.get(task_id, {})
        note = entry.get("note", default)
        return note if isinstance(note, str) else default
