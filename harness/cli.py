"""Command-line interface. It selects a project, starts or resumes a session, and shows the result.

The interface only calls the core modules. It holds no agent logic.

  python -m harness run    --task tasks/advance_expiry.json [--project openstock]
  python -m harness resume <session> [--acknowledge]
  python -m harness cancel <session>
  python -m harness list
  python -m harness show   <session>
  python -m harness diff   <session>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import verify, workspace
from .config import PROJECT_DIR, Settings
from .controller import AgentController, ResumeRefused
from .model import ModelError, OllamaClient
from .navigation import SymbolIndex
from .sandbox import ExecutionEnvironment
from .session import Session, SessionError, SessionStore
from .tasks import Task
from .tools import RepositoryTools

EXIT_OK = 0
EXIT_NOT_VERIFIED = 1
EXIT_SETUP = 2
EXIT_REFUSED = 3
EXIT_CANCELLED = 130


# ── wiring ──────────────────────────────────────────────────
def build_checks(settings: Settings, task: Task) -> Dict[str, List[str]]:
    checks = dict(settings.checks)
    if task.acceptance_script:
        checks["acceptance"] = ["python", task.acceptance_script]
    return checks


def build_env(settings: Settings, checks: Dict[str, List[str]], root: Path) -> ExecutionEnvironment:
    return ExecutionEnvironment(
        workspace=root, acceptance_dir=(PROJECT_DIR / "acceptance").resolve(), checks=checks,
        image=settings.docker_image, timeout_s=settings.check_timeout_s,
        max_output_chars=settings.max_output_chars, memory=settings.container_memory,
        pids=settings.container_pids,
    )


def build_tools(settings: Settings, task: Task, root: Path) -> RepositoryTools:
    return RepositoryTools(
        root=root, allowed_write_paths=task.allowed_write_paths, denied_paths=settings.denied_paths,
        max_read_chars=settings.max_read_chars, max_list_entries=settings.max_list_entries,
        max_search_hits=settings.max_search_hits, starting_commit=task.starting_commit,
        max_symbol_hits=settings.max_symbol_hits,
    )


def make_controller(settings: Settings, store: SessionStore, session: Session, task: Task,
                    model: OllamaClient, focus: Optional[List[str]] = None) -> AgentController:
    checks = build_checks(settings, task)
    tools = build_tools(settings, task, session.repo_dir)
    env = build_env(settings, checks, session.repo_dir)
    return AgentController(model=model, tools=tools, env=env, settings=settings, store=store,
                           session=session, task=task, on_event=_print_event, focus=focus)


def _load_settings(path: str) -> Settings:
    cfg = Path(path)
    if not cfg.is_absolute():
        cfg = (Path.cwd() / cfg) if cfg.exists() else (PROJECT_DIR / cfg)
    return Settings.load(cfg)


def _print_event(kind: str, data: Dict[str, Any]) -> None:
    if kind == "baseline":
        print("[baseline] {} on starting code: {}".format(data["check"], str(data["status"]).upper()))
    elif kind == "context":
        print("[context] {} of {} characters{}".format(
            data["chars"], data["limit"],
            "; {} older messages dropped".format(data["dropped"]) if data.get("dropped") else ""))
    elif kind == "tool":
        tag = " (parsed from text)" if data.get("from_text") else ""
        first = (data.get("result") or "").splitlines()[:1]
        print("[step {}] {}{} -> {}".format(data.get("step", "-"), data["name"], tag, first[0] if first else ""))
    elif kind == "checks":
        print("[checks] " + ", ".join("{}={}".format(r["name"], r["status"]) for r in data["results"]))
    elif kind == "repair":
        print("[repair] round {} of {}: failed checks {}".format(data["round"], data["max"], ", ".join(data["failed"])))
    elif kind == "no_tool_call":
        print("[warn] model replied without a tool call; asking again")
    elif kind == "model_error":
        print("[warn] model error: {}".format(data["error"]))
    elif kind == "cancel_requested":
        print("[cancel] cancel requested; stopping at the next safe point")
    elif kind == "acknowledged":
        print("[resume] acknowledged {}".format(data.get("what")))
    elif kind == "paused":
        print("[paused] resume refused:")
        for p in data.get("problems", []):
            print("  - " + p)


def _finish(store: SessionStore, session: Session) -> int:
    verify.write_outputs(store, session)
    print("\n=== RESULT ===")
    print(session.data.get("status") or "(no status)")
    print("State: {}".format(session.state))
    print("\nAgent summary: {}".format(session.data.get("finish_summary") or "(none)"))
    changed = session.data.get("changed_files") or []
    print("\nChanged files:")
    for f in changed or ["(none)"]:
        print("  " + f)
    print("\n=== CHECKS ===")
    for ch in session.data.get("checks") or []:
        print("{:<22} {:<12} exit={}".format(ch["name"], ch["status"], ch["exit_code"]))
    if session.state == "completed" and str(session.data.get("status", "")).startswith("PASSED"):
        code = EXIT_OK
    elif session.state == "cancelled":
        code = EXIT_CANCELLED
    else:
        code = EXIT_NOT_VERIFIED
    print("\nSession folder: {}".format(session.dir))
    print("Diff:           {}".format(session.dir / "diff.patch"))
    print("Report:         {}".format(session.dir / "report.md"))
    print("Show again:     python -m harness show {}".format(session.id))
    return code


def new_session(settings: Settings, store: SessionStore, task: Task, project: str, source: Path):
    """Create a session folder, its permission snapshot, and the disposable working copy.

    Returns (session, starting commit). Raises WorkspaceError if the copy cannot be made.
    """
    data = {
        "task": task.to_dict(),
        "project": {"name": project, "source": str(source)},
        "starting_commit": task.starting_commit,
        "model": settings.model,
        "max_actions": settings.max_actions,
        "max_retries": settings.max_retries,
        "max_repair_rounds": settings.max_repair_rounds,
        "snapshot": {
            "allowed_write_paths": list(task.allowed_write_paths),
            "denied_paths": list(settings.denied_paths),
            "required_checks": list(task.required_checks),
            "checks": build_checks(settings, task),
        },
    }
    session = store.create(session_id=store.new_id(task.id), data=data)
    try:
        head = workspace.create_workspace(source, session.repo_dir, task.starting_commit)
    except workspace.WorkspaceError:
        store.release(session, "stopped", stop_reason="setup failed", status="STOPPED: setup failed")
        raise
    focus = [rel for rel, _ in SymbolIndex(session.repo_dir, settings.denied_paths).rank(
        task.title + " " + task.request, limit=4)]
    session.data.update(focus=focus, counters={})
    store.save(session)
    return session, head


# ── commands ────────────────────────────────────────────────
def cmd_run(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    if args.model:
        settings.model = args.model
    if args.max_actions:
        settings.max_actions = args.max_actions
    task = Task.load(Path(args.task))
    project = args.project or task.project
    if not project:
        print("error: choose an allowed project with --project ({})".format(", ".join(sorted(settings.projects))),
              file=sys.stderr)
        return EXIT_SETUP
    if project not in settings.projects:
        print("error: '{}' is not an allowed project. Allowed: {}".format(project, ", ".join(sorted(settings.projects))),
              file=sys.stderr)
        return EXIT_SETUP
    source = settings.project_path(project)
    model = OllamaClient(settings.ollama_url, settings.model, settings.model_timeout_s, settings.num_ctx)
    try:
        model.check_model()
    except ModelError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP

    store = SessionStore(settings.runs_path())
    try:
        session, head = new_session(settings, store, task, project, source)
    except workspace.WorkspaceError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    print("Task:            {}".format(task.title))
    print("Project:         {} ({})".format(project, source))
    print("Starting commit: {}".format(head))
    print("Session:         {}".format(session.dir))
    print("Model:           {}".format(settings.model))
    focus = session.data["focus"]
    print("Likely relevant files: {}".format(", ".join(focus) or "(none found)"))

    controller = make_controller(settings, store, session, task, model, focus=focus)
    try:
        controller.start()
    except SessionError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    return _finish(store, session)


def cmd_resume(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    store = SessionStore(settings.runs_path())
    try:
        session = store.open(args.session)
    except SessionError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    task = Task.from_dict(session.data["task"])
    model = OllamaClient(settings.ollama_url, session.data.get("model", settings.model),
                         settings.model_timeout_s, settings.num_ctx)
    try:
        model.check_model()
    except ModelError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    controller = make_controller(settings, store, session, task, model)
    print("Resuming session {} (state {})".format(session.id, session.state))
    try:
        controller.resume(acknowledge=args.acknowledge)
    except ResumeRefused as exc:
        print("Resume refused. Nothing was replayed. Fix the problem above, or re-run with --acknowledge if you checked the state.")
        return EXIT_REFUSED
    except SessionError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    return _finish(store, session)


def cmd_cancel(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    store = SessionStore(settings.runs_path())
    try:
        session = store.open(args.session)
    except SessionError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    if session.state == "running" and not store.is_stale(session, settings.stale_heartbeat_s):
        store.request_cancel(session)
        print("Cancel requested for {}. The running process stops at its next safe point and removes its checks.".format(
            session.id))
        return EXIT_OK
    print("Session {} is {}; there is no active run to cancel.".format(session.id, session.state))
    return EXIT_OK


def cmd_list(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    rows = SessionStore(settings.runs_path()).list()
    if not rows:
        print("No sessions yet.")
        return EXIT_OK
    print("{:<40} {:<10} {:<19} {}".format("SESSION", "STATE", "UPDATED", "TASK"))
    for r in rows:
        print("{:<40} {:<10} {:<19} {}".format(r["id"], r["state"], r["updated"][:19], r["task"]))
    return EXIT_OK


def cmd_show(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    store = SessionStore(settings.runs_path())
    try:
        session = store.open(args.session)
    except SessionError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    for line in verify.summary(store, session):
        print(line)
    return EXIT_OK


def cmd_diff(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    store = SessionStore(settings.runs_path())
    try:
        session = store.open(args.session)
    except SessionError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_SETUP
    _, diff = workspace.changed_state(session.repo_dir)
    print(diff or "(no changes)")
    return EXIT_OK


# ── argument parsing ────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description="Coding harness for a local model (Stage 2).")
    parser.add_argument("--config", default=str(PROJECT_DIR / "harness.json"),
                        help="settings file (optional; defaults are used if it does not exist)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a new session for a task")
    run.add_argument("--task", required=True, help="task JSON file, e.g. tasks/advance_expiry.json")
    run.add_argument("--project", help="allowed project name (default: the task's project)")
    run.add_argument("--model", help="override the model")
    run.add_argument("--max-actions", type=int, help="override the action limit")
    run.set_defaults(func=cmd_run)

    res = sub.add_parser("resume", help="resume a paused, cancelled or interrupted session")
    res.add_argument("session")
    res.add_argument("--acknowledge", action="store_true",
                     help="accept the current working copy and mark interrupted actions as known (never replays them)")
    res.set_defaults(func=cmd_resume)

    can = sub.add_parser("cancel", help="ask a running session to stop")
    can.add_argument("session")
    can.set_defaults(func=cmd_cancel)

    sub.add_parser("list", help="list sessions").set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="show a session's state, checks, counters and unfinished work")
    show.add_argument("session")
    show.set_defaults(func=cmd_show)

    diff = sub.add_parser("diff", help="print the current diff of a session's working copy")
    diff.add_argument("session")
    diff.set_defaults(func=cmd_diff)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Any running check was stopped; the session can be resumed.", file=sys.stderr)
        return EXIT_CANCELLED
