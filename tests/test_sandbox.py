"""Sandbox tests. Docker-backed tests are skipped when the Docker daemon is not running."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from harness.sandbox import ExecutionEnvironment, SandboxCancelled, bound_output
from tests.helpers import cleanup

IMAGE = "python:3.12-slim"


class BoundOutputTest(unittest.TestCase):
    def test_short_output_is_unchanged(self):
        self.assertEqual(bound_output("hello", 100), ("hello", False))

    def test_long_output_is_bounded_and_marked(self):
        text, shortened = bound_output("a" * 5000, 1000)
        self.assertTrue(shortened)
        self.assertIn("output shortened", text)
        self.assertLess(len(text), 1200)


def docker_ready():
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(docker_ready(), "Docker daemon not running")
class DockerExecutionTest(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp(prefix="harness-ws-"))
        self.acc = Path(tempfile.mkdtemp(prefix="harness-acc-"))
        self.env = ExecutionEnvironment(
            workspace=self.ws, acceptance_dir=self.acc,
            checks={
                "ok": ["python", "-c", "print('all good')"],
                "fail": ["python", "-c", "import sys; print('boom'); sys.exit(3)"],
                "big": ["python", "-c", "print('x' * 50000)"],
                "stuck": ["python", "-c", "import time; time.sleep(120)"],
                "writes": ["python", "-c", "open('/work/new.txt','w').write('x')"],
                "no_network": ["python", "-c",
                               "import socket; socket.create_connection(('1.1.1.1', 53), 3)"],
                "missing": ["harness-no-such-command"],
            },
            image=IMAGE, timeout_s=40, max_output_chars=2000, memory="512m", pids=128,
        )

    def tearDown(self):
        cleanup(self.ws)
        cleanup(self.acc)

    def test_passing_check(self):
        res = self.env.run_check("ok")
        self.assertEqual(res.status, "passed")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("all good", res.output)

    def test_failing_check_keeps_exit_code_and_output(self):
        res = self.env.run_check("fail")
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.exit_code, 3)
        self.assertIn("boom", res.output)
        self.assertFalse(res.passed)

    def test_missing_command_is_unavailable_not_failed(self):
        res = self.env.run_check("missing")
        self.assertEqual(res.status, "unavailable")
        self.assertFalse(res.passed)

    def test_large_output_is_bounded(self):
        res = self.env.run_check("big")
        self.assertTrue(res.output_shortened)
        self.assertLess(len(res.output), 2200)

    def test_stuck_check_times_out_and_leaves_no_container(self):
        self.env.timeout_s = 15
        res = self.env.run_check("stuck")
        self.assertEqual(res.status, "timed_out")
        left = subprocess.run(["docker", "ps", "-aq", "--filter", "name=harness-"],
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(left, "")

    def test_container_can_write_only_to_workspace(self):
        res = self.env.run_check("writes")
        self.assertEqual(res.status, "passed")
        self.assertTrue((self.ws / "new.txt").exists())

    def test_container_has_no_network(self):
        res = self.env.run_check("no_network")
        self.assertEqual(res.status, "failed")

    def test_no_check_starts_after_cancel(self):
        self.env.cancel()
        with self.assertRaises(SandboxCancelled):
            self.env.run_check("ok")


class DaemonErrorClassificationTest(unittest.TestCase):
    def test_daemon_errors_are_not_check_failures(self):
        self.assertTrue(ExecutionEnvironment._daemon_error("Error response from daemon: Docker Desktop is unable to start"))
        self.assertTrue(ExecutionEnvironment._daemon_error("Cannot connect to the Docker daemon at npipe"))
        self.assertFalse(ExecutionEnvironment._daemon_error("Traceback: AssertionError"))


class UnavailableDockerTest(unittest.TestCase):
    def test_check_reports_unavailable_when_docker_is_down(self):
        ws = Path(tempfile.mkdtemp(prefix="harness-ws-"))
        try:
            env = ExecutionEnvironment(workspace=ws, acceptance_dir=ws, checks={"ok": ["python", "-V"]},
                                       image=IMAGE, timeout_s=5, max_output_chars=500, memory="256m", pids=64)
            env.docker_available = lambda: (False, "daemon down")
            res = env.run_check("ok")
            self.assertEqual(res.status, "unavailable")
            self.assertFalse(res.passed)
        finally:
            cleanup(ws)


if __name__ == "__main__":
    unittest.main()
