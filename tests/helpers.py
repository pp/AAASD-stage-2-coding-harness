"""Shared test helpers: a scripted model, a fake execution environment, and session factories.

The scripted model and fake environment need no Ollama, Docker, or API key.
"""
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

from harness.config import Settings
from harness.model import ModelCancelled
from harness.sandbox import CheckResult


class ScriptedModel:
    """Returns scripted messages in order. With repeat=True the last message repeats forever."""

    def __init__(self, replies, repeat=False):
        self.replies = list(replies)
        self.repeat = repeat
        self.calls = 0
        self.seen_messages = []

    def chat(self, messages, tools):
        self.calls += 1
        self.seen_messages.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError("ScriptedModel ran out of replies")
        if self.repeat and len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)

    def chat_cancellable(self, messages, tools, cancel, poll_s=0.2):
        if cancel.is_set():
            raise ModelCancelled("cancelled before the request")
        return self.chat(messages, tools)


class BlockingModel:
    """Simulates a slow model: every request waits until cancelled."""

    def __init__(self):
        self.calls = 0
        self.started = threading.Event()

    def chat(self, messages, tools):
        raise AssertionError("not used")

    def chat_cancellable(self, messages, tools, cancel, poll_s=0.05):
        self.calls += 1
        self.started.set()
        while not cancel.is_set():
            cancel.wait(poll_s)
        raise ModelCancelled("cancelled while waiting")


def call(_tool, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": _tool, "arguments": args}}]}


def text(content):
    return {"role": "assistant", "content": content}


class FakeEnv:
    """Stands in for ExecutionEnvironment.

    results maps a check name to a CheckResult, a list of CheckResults (used in order), or a callable.
    Each run is recorded in ran. cancel() makes later runs raise SandboxCancelled, like the real one.
    """

    def __init__(self, checks=None, results=None):
        self.checks = checks or {"acceptance": ["x"], "regression_core": ["y"]}
        self.results = results or {}
        self.ran: List[str] = []
        self.cancelled = False

    def run_check(self, name):
        from harness.sandbox import SandboxCancelled
        if self.cancelled:
            raise SandboxCancelled("cancelled")
        self.ran.append(name)
        spec = self.results.get(name)
        if callable(spec):
            return spec(name)
        if isinstance(spec, list):
            return spec.pop(0) if spec else CheckResult(name, [], "passed", 0, "ok")
        if spec is not None:
            return spec
        return CheckResult(name, [], "passed", 0, "ok")

    def cancel(self):
        self.cancelled = True


def make_repo(files: Dict[str, str]) -> Path:
    """Create a throwaway git repo with the given files (written as LF on every platform)."""
    root = Path(tempfile.mkdtemp(prefix="harness-test-"))
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))
    git = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf"]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(git + ["add", "-A"], cwd=root, check=True)
    subprocess.run(git + ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "start"],
                   cwd=root, check=True)
    return root


def head_of(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()


def settings(**over) -> Settings:
    base = dict(max_actions=10, max_retries=2, max_repair_rounds=2, cancel_poll_s=0.05, stale_heartbeat_s=30,
                max_context_chars=24000, max_map_chars=2000, projects={"test": {"path": "unused"}})
    base.update(over)
    return Settings(**base)


def cleanup(path):
    shutil.rmtree(path, ignore_errors=True)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
