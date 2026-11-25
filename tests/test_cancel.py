"""Cancellation tests: one representative cancel during a model request and one during a command.

The model case uses the real Ollama client against a local HTTP server that never answers. The command
case uses the real Docker sandbox and needs a running Docker daemon (skipped otherwise). After a cancel,
no further actions or writes happen, and no child process or container is left behind.
"""
import http.server
import json
import subprocess
import threading
import time
import unittest

from harness.model import OllamaClient
from harness.sandbox import ExecutionEnvironment
from harness.session import SessionStore
from tests.helpers import FakeEnv, ScriptedModel, call, settings
from tests.test_controller import Setup

IMAGE = "python:3.12-slim"


def docker_ready():
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class HangingOllama(http.server.BaseHTTPRequestHandler):
    """Accepts /api/chat and never answers within the test (until the server is closed)."""
    hits = 0

    def do_POST(self):
        HangingOllama.hits += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(30)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class CancelDuringModelRequestTest(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HangingOllama)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        HangingOllama.hits = 0

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_cancel_while_the_model_is_answering_stops_the_run(self):
        url = "http://127.0.0.1:{}".format(self.server.server_address[1])
        model = OllamaClient(url, "any-model", timeout_s=60, num_ctx=2048)
        s = Setup(self, ScriptedModel([]))
        self.addCleanup(s.close)
        s.controller.model = model
        s.controller.settings.cancel_poll_s = 0.05
        canceller = threading.Timer(1.0, lambda: SessionStore(s.runs).request_cancel(s.session))
        canceller.start()
        started = time.monotonic()
        state = s.run()
        elapsed = time.monotonic() - started
        self.assertEqual(state, "cancelled")
        self.assertLess(elapsed, 10, "the run should stop soon after the cancel request")
        self.assertEqual(HangingOllama.hits, 1)
        self.assertEqual(s.session.data["counters"]["actions"], 0)     # no further actions
        self.assertEqual(s.env.ran, ["acceptance"])                   # only the baseline ran; no final checks
        self.assertTrue(s.data["status"].startswith("CANCELLED"))
        self.assertEqual(s.data["changed_files"], [])
        self.assertEqual(SessionStore(s.runs).journal_read(s.session)[-1]["phase"], "finished")  # baseline only


@unittest.skipUnless(docker_ready(), "Docker daemon not running")
class CancelDuringCommandTest(unittest.TestCase):
    def test_cancel_stops_a_stuck_check_and_its_container(self):
        import tempfile
        from pathlib import Path
        acc = Path(tempfile.mkdtemp(prefix="harness-acc-"))
        env = ExecutionEnvironment(
            workspace=Path("."), acceptance_dir=acc,
            checks={"acceptance": ["python", "-c", "print('ok')"],
                    "regression_core": ["python", "-c", "import time; time.sleep(120)"]},
            image=IMAGE, timeout_s=120, max_output_chars=2000, memory="512m", pids=128)
        model = ScriptedModel([call("run_check", name="regression_core"), call("list_files", path=".")])
        s = Setup(self, model, env, required=("acceptance", "regression_core"))
        self.addCleanup(s.close)
        env.workspace = s.session.repo_dir

        def cancel_when_the_stuck_container_exists():
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                ids = subprocess.run(["docker", "ps", "-q", "--filter", "name=harness-"],
                                     capture_output=True, text=True).stdout.strip()
                if ids:
                    time.sleep(1.0)            # let the check run for a moment
                    SessionStore(s.runs).request_cancel(s.session)
                    return
                time.sleep(0.2)

        threading.Thread(target=cancel_when_the_stuck_container_exists, daemon=True).start()
        started = time.monotonic()
        state = s.run()
        elapsed = time.monotonic() - started
        left = subprocess.run(["docker", "ps", "-aq", "--filter", "name=harness-"],
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(state, "cancelled")
        self.assertLess(elapsed, 60, "a stuck check must not hold the run")
        self.assertEqual(left, "", "no check container may be left behind")
        self.assertEqual(model.calls, 1, "no further model requests after the cancel")
        entries = SessionStore(s.runs).journal_read(s.session)
        self.assertEqual(SessionStore.uncertain(entries)[0]["tool"], "run_check")
        self.assertTrue(s.data["status"].startswith("CANCELLED"))


class CancelCommandTest(unittest.TestCase):
    def test_cancel_command_only_signals_a_running_session(self):
        import tempfile
        from pathlib import Path
        from harness import cli
        runs = Path(tempfile.mkdtemp(prefix="harness-cli-runs-"))
        cfg = Path(tempfile.mkdtemp(prefix="harness-cli-cfg-")) / "harness.json"
        cfg.write_text(json.dumps({"runs_dir": str(runs)}), encoding="utf-8")
        store = SessionStore(runs)
        running = store.create(session_id="running-one", data={"task": {"title": "t"}})
        store.acquire(running, stale_s=30)
        paused = store.create(session_id="paused-one", data={"task": {"title": "t"}})
        store.release(paused, "paused")
        try:
            self.assertEqual(cli.main(["--config", str(cfg), "cancel", "running-one"]), 0)
            self.assertTrue(store.cancel_requested(running))
            self.assertEqual(cli.main(["--config", str(cfg), "cancel", "paused-one"]), 0)
            self.assertFalse(store.cancel_requested(paused))
        finally:
            import shutil
            shutil.rmtree(runs, ignore_errors=True)
            shutil.rmtree(cfg.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
