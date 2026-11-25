"""Contained execution of configured checks, using a disposable Docker container per check."""
from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Docker exit codes that mean the check never ran: 125 docker run failed, 126 not executable, 127 not found.
NOT_RUN_CODES = (125, 126, 127)


class SandboxCancelled(Exception):
    """The environment was cancelled. No further checks may run."""


@dataclass
class CheckResult:
    name: str
    argv: List[str]
    status: str                      # passed | failed | timed_out | unavailable | cancelled
    exit_code: Optional[int]
    output: str
    output_shortened: bool = False

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def bound_output(text: str, limit: int) -> Tuple[str, bool]:
    """Keep the start and end of long output and mark that it was shortened."""
    if len(text) <= limit:
        return text, False
    head = limit * 3 // 4
    tail = limit - head
    marker = "\n... [output shortened: {} of {} characters shown] ...\n".format(limit, len(text))
    return text[:head] + marker + text[-tail:], True


@dataclass
class ExecutionEnvironment:
    workspace: Path
    acceptance_dir: Path
    checks: Dict[str, List[str]]
    image: str
    timeout_s: int
    max_output_chars: int
    memory: str
    pids: int
    _running: Set[str] = field(default_factory=set)
    _procs: Dict[str, subprocess.Popen] = field(default_factory=dict)
    _cancelled: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def docker_available(self) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            return False, detail[-1] if detail else "docker info failed"
        return True, proc.stdout.strip()

    def _command(self, container: str, argv: List[str]) -> List[str]:
        cmd = [
            "docker", "run", "--rm", "--name", container,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=256m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", self.memory, "--pids-limit", str(self.pids),
        ]
        if hasattr(os, "getuid"):          # Linux/macOS hosts: keep workspace files owned by the host user
            cmd += ["--user", "{}:{}".format(os.getuid(), os.getgid())]
        cmd += [
            "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "REPO_DIR=/work",
            "--mount", "type=bind,source={},target=/work".format(self.workspace),
            "--mount", "type=bind,source={},target=/acceptance,readonly".format(self.acceptance_dir),
            "-w", "/work",
            self.image, *argv,
        ]
        return cmd

    def run_check(self, name: str) -> CheckResult:
        if name not in self.checks:
            raise KeyError(name)
        if self._cancelled:
            raise SandboxCancelled("the run was cancelled; no further checks are started")
        argv = list(self.checks[name])
        ok, info = self.docker_available()
        if not ok:
            return CheckResult(name, argv, "unavailable", None,
                               "Docker is not available, so the check did not run: {}".format(info))
        container = "harness-{}".format(uuid.uuid4().hex[:12])
        with self._lock:
            if self._cancelled:
                raise SandboxCancelled("the run was cancelled; no further checks are started")
            self._running.add(container)
            proc = subprocess.Popen(self._command(container, argv), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            self._procs[container] = proc
        timed_out = False
        try:
            try:
                stdout_b, stderr_b = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._remove(container)
                proc.kill()
                stdout_b, stderr_b = proc.communicate()
            exit_code = proc.returncode
        finally:
            self._reap(container)
            with self._lock:
                self._running.discard(container)
                self._procs.pop(container, None)
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if self._cancelled:
            return CheckResult(name, argv, "cancelled", None,
                               "Check was cancelled while running; its result is not a check result.")
        text = (stdout + stderr).strip() or "(no output)"
        text, shortened = bound_output(text, self.max_output_chars)
        if timed_out:
            status = "timed_out"
            exit_code = None
            text = "Check stopped after {} s and its container was removed.\n{}".format(self.timeout_s, text)
        elif exit_code == 0:
            status = "passed"
        elif exit_code in NOT_RUN_CODES or self._daemon_error(stderr):
            status = "unavailable"
            text = "The check could not be started in the sandbox (exit code {}).\n{}".format(exit_code, text)
        else:
            status = "failed"
        return CheckResult(name, argv, status, exit_code, text, shortened)

    @staticmethod
    def _daemon_error(stderr: str) -> bool:
        """Docker itself failed (daemon down or unreachable). The check did not run, so it is not a failure."""
        markers = ("error response from daemon", "cannot connect to the docker daemon",
                   "docker desktop is unable to start", "is the docker daemon running")
        low = stderr.lower()
        return any(m in low for m in markers)

    def _remove(self, container: str) -> None:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=60)

    def _reap(self, container: str, attempts: int = 5) -> None:
        """Remove the container and confirm it is gone.

        A docker client that was killed can leave the daemon still creating the container, so one
        removal may run too early. The check is repeated until docker no longer lists the name.
        """
        for attempt in range(attempts):
            self._remove(container)
            left = subprocess.run(["docker", "ps", "-aq", "--filter", "name=^{}$".format(container)],
                                  capture_output=True, text=True, timeout=60).stdout.strip()
            if not left:
                return
            time.sleep(0.5)

    def cancel(self) -> None:
        """Stop every running check: remove its container and end the docker client process.

        After this call no new check starts. Used for cancellation and for Ctrl-C.
        """
        with self._lock:
            self._cancelled = True
            running = list(self._running)
            procs = list(self._procs.values())
        for proc in procs:
            try:
                proc.kill()
            except OSError:
                pass
        for container in running:
            self._reap(container)

    def stop_processes(self) -> None:
        """Backward-compatible name for cancel()."""
        self.cancel()

    @property
    def cancelled(self) -> bool:
        return self._cancelled
