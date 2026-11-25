"""Saved sessions: task details, the action journal (checkpoints), conversation, events, and cancellation.

Layout of one session folder (runs/<session-id>/):

  session.json   task, starting commit, permission snapshot, state, counters, last fingerprint, lock
  repo/          the disposable working copy the agent edits
  journal.jsonl  one record per side-effecting action: "started", then "finished" (or "interrupted")
  messages.json  the conversation, saved after each model reply and tool result
  events.jsonl   progress events for display and later inspection
  cancel.request written by `harness cancel`; the running process polls for it

Journal rule for resume: an action with a "started" record and no "finished" or "acknowledged"
record is uncertain. It is never replayed automatically.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSION_FILE = "session.json"
JOURNAL_FILE = "journal.jsonl"
MESSAGES_FILE = "messages.json"
EVENTS_FILE = "events.jsonl"
CANCEL_FILE = "cancel.request"
REPO_DIR = "repo"

RESUMABLE_STATES = ("paused", "cancelled", "running")   # "running" only when its heartbeat is stale
FINAL_STATES = ("completed", "stopped")


class SessionError(Exception):
    pass


class SessionBusy(SessionError):
    pass


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _parse_time(text: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


@dataclass
class Session:
    dir: Path
    data: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def state(self) -> str:
        return self.data["state"]

    @property
    def repo_dir(self) -> Path:
        return self.dir / REPO_DIR

    @property
    def task(self) -> Dict[str, Any]:
        return self.data["task"]


class SessionStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    # ── creating and opening ────────────────────────────────
    def new_id(self, task_id: str) -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in task_id)
        candidate = "{}-{}".format(stamp, safe)
        n = 2
        while (self.root / candidate).exists():
            candidate = "{}-{}-{}".format(stamp, safe, n)
            n += 1
        return candidate

    def create(self, *, session_id: str, data: Dict[str, Any]) -> Session:
        folder = self.root / session_id
        folder.mkdir(parents=True, exist_ok=False)
        data = dict(data, id=session_id, created=now_iso(), updated=now_iso(), state="new")
        s = Session(folder, data)
        self.save(s)
        return s

    def open(self, session_id: str) -> Session:
        folder = self.root / session_id
        path = folder / SESSION_FILE
        if not path.exists():
            raise SessionError("no session named '{}' in {}".format(session_id, self.root))
        return Session(folder, json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> List[Dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for folder in sorted(self.root.iterdir()):
            path = folder / SESSION_FILE
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                out.append({"id": data.get("id", folder.name), "state": data.get("state", "?"),
                            "task": (data.get("task") or {}).get("title", ""),
                            "updated": data.get("updated", ""), "status": data.get("status", "")})
        return out

    # ── session record ──────────────────────────────────────
    def save(self, s: Session) -> None:
        with self._lock:
            s.data["updated"] = now_iso()
            tmp = s.dir / (SESSION_FILE + ".tmp")
            tmp.write_text(json.dumps(s.data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, s.dir / SESSION_FILE)

    def acquire(self, s: Session, stale_s: int) -> None:
        """Mark the session as running for this process. Refuses if another live process holds it."""
        with self._lock:
            if s.state == "running" and not self.is_stale(s, stale_s):
                raise SessionBusy("session {} is running in another process (pid {})".format(
                    s.id, s.data.get("owner_pid")))
            s.data.update(state="running", owner_pid=os.getpid(), heartbeat=now_iso())
            self.save(s)

    def heartbeat(self, s: Session) -> None:
        with self._lock:
            s.data["heartbeat"] = now_iso()
            self.save(s)

    def release(self, s: Session, state: str, **extra: Any) -> None:
        with self._lock:
            s.data.update(state=state, owner_pid=None, **extra)
            self.save(s)

    @staticmethod
    def is_stale(s: Session, stale_s: int) -> bool:
        beat = _parse_time(s.data.get("heartbeat", ""))
        if beat is None:
            return True
        return (datetime.datetime.now() - beat).total_seconds() > stale_s

    # ── journal (checkpoints) ───────────────────────────────
    def journal_append(self, s: Session, entry: Dict[str, Any]) -> None:
        with self._lock:
            with (s.dir / JOURNAL_FILE).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(entry, time=now_iso()), ensure_ascii=False) + "\n")

    def journal_read(self, s: Session) -> List[Dict[str, Any]]:
        path = s.dir / JOURNAL_FILE
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    def next_seq(self, s: Session) -> int:
        return sum(1 for e in self.journal_read(s) if e.get("phase") == "started") + 1

    @staticmethod
    def uncertain(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Started or interrupted actions with no finished or acknowledged record."""
        pending: Dict[int, Dict[str, Any]] = {}
        closed = set()
        for e in entries:
            seq = e.get("seq")
            if e.get("phase") in ("started", "interrupted"):
                pending[seq] = e
            elif e.get("phase") in ("finished", "acknowledged"):
                closed.add(seq)
        return [e for seq, e in sorted(pending.items()) if seq not in closed]

    # ── conversation and events ─────────────────────────────
    def messages_save(self, s: Session, messages: List[Dict[str, Any]]) -> None:
        with self._lock:
            tmp = s.dir / (MESSAGES_FILE + ".tmp")
            tmp.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, s.dir / MESSAGES_FILE)

    def messages_load(self, s: Session) -> List[Dict[str, Any]]:
        path = s.dir / MESSAGES_FILE
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def events_append(self, s: Session, kind: str, data: Dict[str, Any]) -> None:
        with self._lock:
            with (s.dir / EVENTS_FILE).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"time": now_iso(), "event": kind, **data}, ensure_ascii=False) + "\n")

    def events_read(self, s: Session) -> List[Dict[str, Any]]:
        path = s.dir / EVENTS_FILE
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ── cancellation ────────────────────────────────────────
    def request_cancel(self, s: Session) -> None:
        (s.dir / CANCEL_FILE).write_text(now_iso(), encoding="utf-8")

    def cancel_requested(self, s: Session) -> bool:
        return (s.dir / CANCEL_FILE).exists()

    def clear_cancel(self, s: Session) -> None:
        path = s.dir / CANCEL_FILE
        if path.exists():
            path.unlink()
