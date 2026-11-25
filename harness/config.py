"""Settings for the harness. Keep secrets out of this file."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CHECKS: Dict[str, List[str]] = {
    "regression_core": ["python", "tests/test_core.py"],
    "regression_reversal": ["python", "tests/test_reversal_advances.py"],
    "build": ["python", "/acceptance/check_build.py"],
}

DEFAULT_DENIED = (
    ".git", "data", "backups", "documents", "logs", "venv", ".venv", "vendor",
    "__pycache__", "config_local.json", ".env", ".vscode", ".idea",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    runs_dir: Path = Path("runs")
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen3.8"
    model_timeout_s: int = 600
    num_ctx: int = 16384
    max_actions: int = 80
    max_retries: int = 3
    max_repair_rounds: int = 3
    max_output_chars: int = 4000
    max_read_chars: int = 6000
    max_list_entries: int = 200
    max_search_hits: int = 50
    max_symbol_hits: int = 50
    max_context_chars: int = 24000
    max_map_chars: int = 4000
    max_snippet_chars: int = 6000
    verbatim_turns: int = 8
    max_message_chars: int = 2000
    docker_image: str = "python:3.12-slim"
    check_timeout_s: int = 300
    container_memory: str = "1g"
    container_pids: int = 256
    cancel_poll_s: float = 0.5
    stale_heartbeat_s: int = 30
    checks: Dict[str, List[str]] = field(default_factory=lambda: dict(DEFAULT_CHECKS))
    denied_paths: Tuple[str, ...] = DEFAULT_DENIED
    projects: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "openstock": {"path": "targets/openstock", "description": "OpenStock (Flask + SQLite)"},
    })

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError("Unknown settings: {}".format(", ".join(sorted(unknown))))
        if "runs_dir" in raw:
            raw["runs_dir"] = Path(raw["runs_dir"])
        if "denied_paths" in raw:
            raw["denied_paths"] = tuple(raw["denied_paths"])
        return cls(**raw)

    def project_path(self, name: str) -> Path:
        """Absolute path of an allowed project. Relative paths are taken from the harness folder."""
        if name not in self.projects:
            raise KeyError(name)
        p = Path(self.projects[name]["path"])
        return p if p.is_absolute() else (PROJECT_DIR / p).resolve()

    def runs_path(self) -> Path:
        return self.runs_dir if self.runs_dir.is_absolute() else (PROJECT_DIR / self.runs_dir).resolve()
