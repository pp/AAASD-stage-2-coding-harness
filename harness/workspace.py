"""Disposable copy of the target repository, host-side git reads, and a fingerprint of the working copy."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class WorkspaceError(Exception):
    pass


COPY_IGNORE = shutil.ignore_patterns(
    "venv", ".venv", "vendor", "__pycache__", "*.pyc", "backups", "documents", "logs",
)


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise WorkspaceError("git {} failed: {}".format(" ".join(args), proc.stderr.strip()))
    return proc.stdout


def head_of(repo: Path) -> str:
    return _git(Path(repo), "rev-parse", "HEAD").strip()


def create_workspace(source: Path, dest: Path, expected_commit: Optional[str]) -> str:
    """Copy the target repository into dest and return its commit hash."""
    source = Path(source).resolve()
    if not (source / ".git").exists():
        raise WorkspaceError("{} is not a git repository".format(source))
    head = head_of(source)
    if expected_commit and head != expected_commit:
        raise WorkspaceError("Target repo is at {} but the task expects {}".format(head, expected_commit))
    if dest.exists():
        raise WorkspaceError("Workspace already exists: {}".format(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, ignore=COPY_IGNORE, symlinks=True)
    return head


def changed_files(workspace: Path) -> List[str]:
    out = _git(workspace, "status", "--porcelain")
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def diff(workspace: Path) -> str:
    return _git(workspace, "diff")


def changed_state(workspace: Path) -> Tuple[List[str], str]:
    return changed_files(workspace), diff(workspace)


def git_state(workspace: Path, starting_commit: str) -> Dict[str, object]:
    """Read-only summary of the copy's git state, compared with the task's starting commit."""
    head = head_of(workspace)
    branch = _git(workspace, "rev-parse", "--abbrev-ref", "HEAD").strip()
    return {
        "branch": branch,
        "head": head,
        "starting_commit": starting_commit,
        "at_starting_commit": head == starting_commit,
        "changed_files": changed_files(workspace),
    }


def fingerprint(root: Path, denied: Iterable[str]) -> str:
    """Hash of every file the agent can reach (denied folders such as data/ and .git are skipped).

    Stored after each harness edit. On resume, a different value means the working copy was
    changed by something other than the recorded actions.
    """
    root = Path(root)
    denied = frozenset(denied)
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in denied)
        for name in sorted(filenames):
            if name in denied:
                continue
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            h.update(rel.encode("utf-8") + b"\0")
            if p.is_symlink():
                h.update(b"symlink:" + os.readlink(p).encode("utf-8", errors="replace"))
            else:
                h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
