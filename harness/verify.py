"""Verification output: the final diff, the report, and the summary shown by `show`.

Nothing here runs repository code. Git is read on the working copy, and the report files are written
into the session folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import workspace
from .session import Session, SessionStore


def unfinished_work(store: SessionStore, session: Session) -> List[str]:
    """Plain-language list of what is not done: interrupted actions, failed checks, state problems."""
    items: List[str] = []
    for e in SessionStore.uncertain(store.journal_read(session)):
        items.append("interrupted, outcome unknown: {} {}".format(e.get("tool"), e.get("args", "")))
    for c in session.data.get("checks") or []:
        if c["status"] != "passed":
            items.append("check {} is {} (exit {})".format(c["name"], c["status"], c["exit_code"]))
    state = session.state
    if state == "running":
        items.append("session was left running: its process stopped without recording an end. Resume to continue.")
    if state in ("cancelled", "paused"):
        items.append("session is {}: {}".format(state, session.data.get("stop_reason") or "no reason recorded"))
    if state == "stopped":
        items.append("run stopped: {}".format(session.data.get("stop_reason") or "no reason recorded"))
    return items


def write_outputs(store: SessionStore, session: Session) -> Path:
    """Writes diff.patch and report.md into the session folder. Returns the report path."""
    changed, diff = workspace.changed_state(session.repo_dir)
    (session.dir / "diff.patch").write_text(diff, encoding="utf-8")
    report = render_report(store, session, changed)
    path = session.dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path


def render_report(store: SessionStore, session: Session, changed: List[str]) -> str:
    d = session.data
    c = d.get("counters") or {}
    task = d.get("task") or {}
    lines = [
        "# Session {}".format(session.id),
        "",
        "Task: {} ({})".format(task.get("title", ""), task.get("id", "")),
        "Project: {}".format((d.get("project") or {}).get("name", "")),
        "Starting commit: `{}`".format(d.get("starting_commit", "")),
        "Model: {}".format(d.get("model", "")),
        "",
        "**State:** {}".format(session.state),
        "",
        "**Status:** {}".format(d.get("status") or "(not finished)"),
        "",
        "Agent summary: {}".format(d.get("finish_summary") or "(none)"),
        "",
        "## Counters",
        "- actions: {}".format(c.get("actions", 0)),
        "- invalid requests: {}".format(c.get("invalid_requests", 0)),
        "- denied actions: {}".format(c.get("denied_actions", 0)),
        "- model retries: {}".format(c.get("model_retries", 0)),
        "- repair rounds: {}".format(c.get("repair_rounds", 0)),
        "- shortened outputs: {}".format(c.get("shortened_outputs", 0)),
        "",
        "## Checks",
    ]
    if d.get("baseline"):
        b = d["baseline"]
        lines.append("- baseline {} on starting code: {} (exit {})".format(b["check"], b["status"], b["exit_code"]))
    for ch in d.get("checks") or []:
        lines.append("- {}: {} (exit {})".format(ch["name"], ch["status"], ch["exit_code"]))
    lines += ["", "## Changed files"] + (["- " + f for f in changed] or ["(none)"])
    lines += ["", "## Unfinished work"] + (["- " + w for w in unfinished_work(store, session)] or ["(nothing)"])
    lines += ["", "Session folder: `{}`".format(session.dir), ""]
    return "\n".join(lines)


def summary(store: SessionStore, session: Session) -> List[str]:
    d = session.data
    c = d.get("counters") or {}
    task = d.get("task") or {}
    out = [
        "Session:   {}".format(session.id),
        "State:     {}".format(session.state),
        "Task:      {} ({})".format(task.get("title", ""), task.get("id", "")),
        "Project:   {}".format((d.get("project") or {}).get("name", "")),
        "Commit:    {}".format(d.get("starting_commit", "")),
        "Updated:   {}".format(d.get("updated", "")),
        "Status:    {}".format(d.get("status") or "(not finished)"),
        "Counters:  actions {}/{}, invalid {}, denied {}, model retries {}/{}, repair rounds {}/{}".format(
            c.get("actions", 0), d.get("max_actions", "?"), c.get("invalid_requests", 0), c.get("denied_actions", 0),
            c.get("model_retries", 0), d.get("max_retries", "?"), c.get("repair_rounds", 0), d.get("max_repair_rounds", "?")),
    ]
    if d.get("baseline"):
        b = d["baseline"]
        out.append("Baseline:  {} on starting code: {}".format(b["check"], b["status"]))
    if d.get("checks"):
        out.append("Checks:")
        for ch in d["checks"]:
            out.append("  {:<22} {:<12} exit={}".format(ch["name"], ch["status"], ch["exit_code"]))
    out.append("Changed files: " + (", ".join(d.get("changed_files") or []) or "(none)"))
    out.append("Unfinished work:")
    items = unfinished_work(store, session)
    out += ["  - " + i for i in items] or ["  (nothing)"]
    events = store.events_read(session)[-8:]
    if events:
        out.append("Recent events:")
        for e in events:
            detail = {k: v for k, v in e.items() if k not in ("time", "event")}
            out.append("  {} {} {}".format(e.get("time", "")[11:19], e.get("event", ""), str(detail)[:120]))
    return out
