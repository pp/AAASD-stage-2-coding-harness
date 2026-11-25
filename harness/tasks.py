"""Task requests: what the user asks for, which files may change, and which checks decide the result."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_REQUIRED = ("acceptance", "regression_core", "regression_reversal")


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    request: str
    starting_commit: str
    allowed_write_paths: Tuple[str, ...]
    required_checks: Tuple[str, ...] = DEFAULT_REQUIRED
    project: Optional[str] = None
    acceptance_script: Optional[str] = None   # path inside the sandbox, e.g. /acceptance/check_x.py

    @classmethod
    def load(cls, path: Path) -> "Task":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Task":
        for key in ("id", "title", "request", "starting_commit", "allowed_write_paths"):
            if key not in raw:
                raise ValueError("Task is missing '{}'".format(key))
        allowed = tuple(p.strip("/") for p in raw["allowed_write_paths"])
        if not allowed:
            raise ValueError("Task must list at least one allowed write path")
        return cls(
            id=raw["id"],
            title=raw["title"],
            request=raw["request"],
            starting_commit=raw["starting_commit"],
            allowed_write_paths=allowed,
            required_checks=tuple(raw.get("required_checks", DEFAULT_REQUIRED)),
            project=raw.get("project"),
            acceptance_script=raw.get("acceptance_script"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "request": self.request,
            "starting_commit": self.starting_commit,
            "allowed_write_paths": list(self.allowed_write_paths),
            "required_checks": list(self.required_checks),
            "project": self.project, "acceptance_script": self.acceptance_script,
        }

    def scope_text(self) -> str:
        return ", ".join(self.allowed_write_paths)
