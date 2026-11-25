"""Controller tests with a scripted model. They need no Ollama, Docker, or API key.

Earlier (Stage 1) behaviour is kept: correct tool calls, invalid requests, scope, failed checks,
action and retry limits. Stage 2 adds the edit-test-fix loop with bounded repair rounds.
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

from harness import workspace
from harness.cli import new_session
from harness.controller import AgentController
from harness.model import ModelError, extract_tool_calls
from harness.sandbox import CheckResult
from harness.session import SessionStore
from harness.tasks import Task
from harness.tools import RepositoryTools
from tests.helpers import FakeEnv, ScriptedModel, call, cleanup, head_of, make_repo, settings, text

SOURCE = "def sell():\n    return 1\n\ndef other():\n    return 1\n"


class ErrorModel:
    def chat_cancellable(self, messages, tools, cancel, poll_s=0.2):
        raise ModelError("connection refused")


class Setup:
    """One throwaway repository, runs folder, session, and controller for a test."""

    def __init__(self, test, model, env=None, *, required=("acceptance", "regression_core"), **over):
        self.root = make_repo({"operations.py": SOURCE, "db.py": "class DB:\n    pass\n"})
        self.runs = Path(tempfile.mkdtemp(prefix="harness-runs-"))
        self.events = []
        head = head_of(self.root)
        self.settings = settings(**over)
        self.store = SessionStore(self.runs)
        self.task = Task(id="t", title="Test task", request="Change sell.", starting_commit=head,
                         allowed_write_paths=("operations.py",), required_checks=tuple(required))
        self.session, _ = new_session(self.settings, self.store, self.task, "test", self.root)
        self.env = env or FakeEnv()
        # the saved permission snapshot must describe the checks this test's environment runs
        self.session.data["snapshot"]["checks"] = dict(self.env.checks)
        self.store.save(self.session)
        tools = RepositoryTools(root=self.session.repo_dir, allowed_write_paths=self.task.allowed_write_paths,
                                denied_paths=self.settings.denied_paths, max_read_chars=2000,
                                max_list_entries=50, max_search_hits=50, starting_commit=head)
        self.controller = AgentController(model=model, tools=tools, env=self.env, settings=self.settings,
                                          store=self.store, session=self.session, task=self.task,
                                          on_event=lambda k, d: self.events.append((k, d)))
        self.test = test

    def run(self):
        return self.controller.start()

    def repo_text(self, rel):
        return (self.session.repo_dir / rel).read_text(encoding="utf-8")

    @property
    def data(self):
        return self.session.data

    def close(self):
        cleanup(self.root)
        cleanup(self.runs)


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.made = []

    def tearDown(self):
        for s in self.made:
            s.close()

    def make(self, model, env=None, **over):
        s = Setup(self, model, env, **over)
        self.made.append(s)
        return s

    # ── Stage 1 behaviour ───────────────────────────────────
    def test_calls_correct_tool_and_receives_result(self):
        model = ScriptedModel([call("read_file", path="operations.py"), call("finish", summary="Read the file.")])
        s = self.make(model)
        s.run()
        self.assertEqual(s.data["finish_summary"], "Read the file.")
        tool_msgs = [m for m in model.seen_messages[1] if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["name"], "read_file")
        self.assertIn("1 | def sell():", tool_msgs[0]["content"])

    def test_unknown_tool_is_refused_without_action(self):
        model = ScriptedModel([call("delete_repo", path="."), call("finish", summary="x")])
        s = self.make(model)
        s.run()
        self.assertEqual(s.data["counters"]["invalid_requests"], 1)
        tool_msgs = [m for m in model.seen_messages[1] if m["role"] == "tool"]
        self.assertIn("Unknown tool", tool_msgs[0]["content"])
        self.assertEqual(workspace.changed_files(s.session.repo_dir), [])

    def test_invalid_arguments_are_refused(self):
        model = ScriptedModel([
            call("read_file", path=5),
            call("edit_file", path="operations.py", old_string="return 1"),
            call("run_check", name="rm -rf /"),
            call("finish", summary="x"),
        ])
        s = self.make(model)
        s.run()
        self.assertEqual(s.data["counters"]["invalid_requests"], 3)
        self.assertEqual(s.data["counters"]["denied_actions"], 0)
        self.assertEqual(workspace.changed_files(s.session.repo_dir), [])

    def test_edit_outside_scope_is_denied_and_counted(self):
        model = ScriptedModel([call("edit_file", path="db.py", old_string="pass", new_string="return 1"),
                               call("finish", summary="x")])
        s = self.make(model)
        s.run()
        self.assertEqual(s.data["counters"]["denied_actions"], 1)
        self.assertIn("pass", s.repo_text("db.py"))

    def test_failed_command_stays_visible_and_is_not_reported_as_passed(self):
        failing = CheckResult("regression_core", [], "failed", 1, "FAILED: sales total wrong")
        env = FakeEnv(results={"regression_core": failing})
        model = ScriptedModel([call("run_check", name="regression_core"), call("finish", summary="All good.")])
        s = self.make(model, env=env, max_repair_rounds=0)
        s.run()
        tool_msgs = [m for m in model.seen_messages[1] if m["role"] == "tool"]
        self.assertIn("exit code 1", tool_msgs[0]["content"])
        self.assertIn("FAILED: sales total wrong", tool_msgs[0]["content"])
        self.assertTrue(s.data["status"].startswith("NOT VERIFIED"), s.data["status"])
        self.assertIn("regression_core", s.data["status"])

    def test_action_limit_stops_repeating_reply(self):
        model = ScriptedModel([call("list_files", path=".")], repeat=True)
        s = self.make(model, max_actions=3)
        s.run()
        self.assertEqual(s.data["counters"]["actions"], 3)
        self.assertIn("action limit", s.data["stop_reason"])
        self.assertTrue(s.data["status"].startswith("STOPPED"))
        self.assertEqual(s.session.state, "stopped")

    def test_model_that_never_calls_a_tool_is_stopped(self):
        model = ScriptedModel([text("I think I am done."), text("Still done."), text("Done again.")], repeat=True)
        s = self.make(model, max_retries=2)
        s.run()
        self.assertIn("without calling a tool", s.data["stop_reason"])
        self.assertEqual(s.data["counters"]["model_retries"], 3)

    def test_model_errors_are_counted_and_stop_the_run(self):
        s = self.make(ErrorModel(), max_retries=2)
        s.run()
        self.assertEqual(s.data["counters"]["model_retries"], 3)
        self.assertIn("model error", s.data["stop_reason"])
        self.assertEqual(s.session.state, "stopped")

    def test_json_tool_call_written_as_text_is_accepted(self):
        content = json.dumps({"name": "finish", "arguments": {"summary": "from text"}})
        s = self.make(ScriptedModel([{"role": "assistant", "content": content}]))
        s.run()
        self.assertEqual(s.data["finish_summary"], "from text")
        self.assertTrue(any(k == "tool" and d.get("from_text") for k, d in s.events))

    def test_passing_run_reports_passed(self):
        model = ScriptedModel([
            call("edit_file", path="operations.py", old_string="    return 1\n\ndef other", new_string="    return 2\n\ndef other"),
            call("finish", summary="Changed return value."),
        ])
        s = self.make(model)
        s.run()
        self.assertEqual(s.session.state, "completed")
        self.assertEqual(s.data["changed_files"], ["operations.py"])
        self.assertEqual(s.data["status"], "PASSED: agent finished and all required checks passed")

    def test_finish_without_checks_is_not_passed(self):
        env = FakeEnv(results={"acceptance": CheckResult("acceptance", [], "unavailable", None, "docker off")})
        s = self.make(ScriptedModel([call("finish", summary="Done.")]), env=env, max_repair_rounds=0)
        s.run()
        self.assertTrue(s.data["status"].startswith("NOT VERIFIED"), s.data["status"])
        self.assertNotIn("PASSED", s.data["status"])

    def test_invalid_finish_summary_is_refused_and_the_run_goes_on(self):
        model = ScriptedModel([call("finish", summary="   "), call("finish", summary="Done now.")])
        s = self.make(model)
        s.run()
        self.assertEqual(s.data["counters"]["invalid_requests"], 1)
        self.assertEqual(s.data["finish_summary"], "Done now.")
        self.assertEqual(s.session.state, "completed")

    # ── Stage 2: edit-test-fix loop ─────────────────────────
    def test_failed_check_guides_a_fix_then_the_run_passes(self):
        failed = CheckResult("regression_core", [], "failed", 1, "FAILED: sales total wrong")
        passed = CheckResult("regression_core", [], "passed", 0, "ok")
        env = FakeEnv(results={"regression_core": [failed, passed]})
        model = ScriptedModel([
            call("finish", summary="I think it is done."),
            call("edit_file", path="operations.py", old_string="    return 1\n\ndef other", new_string="    return 2\n\ndef other"),
            call("finish", summary="Fixed the return value."),
        ])
        s = self.make(model, env=env)
        s.run()
        repair = [m for m in model.seen_messages[1] if m["role"] == "user" and "Repair round" in m["content"]]
        self.assertEqual(len(repair), 1)
        self.assertIn("Repair round 1 of 2", repair[0]["content"])
        self.assertIn('<untrusted source="check:regression_core">', repair[0]["content"])
        self.assertIn("FAILED: sales total wrong", repair[0]["content"])
        self.assertEqual(s.data["counters"]["repair_rounds"], 1)
        self.assertEqual(s.session.state, "completed")
        self.assertIn("PASSED", s.data["status"])

    def test_repair_rounds_are_bounded_and_then_the_run_stops(self):
        failed = CheckResult("regression_core", [], "failed", 1, "FAILED")
        env = FakeEnv(results={"regression_core": failed})
        s = self.make(ScriptedModel([call("finish", summary="done")], repeat=True), env=env, max_repair_rounds=2)
        s.run()
        self.assertEqual(s.data["counters"]["repair_rounds"], 2)
        self.assertEqual(s.session.state, "stopped")
        self.assertTrue(s.data["status"].startswith("NOT VERIFIED"), s.data["status"])
        self.assertIn("after 2 repair rounds", s.data["status"])

    def test_checks_are_not_run_twice_when_nothing_changed(self):
        env = FakeEnv()
        s = self.make(ScriptedModel([call("finish", summary="done")]), env=env)
        s.run()
        self.assertEqual(env.ran.count("regression_core"), 1)

    def test_context_stays_within_the_limit_during_a_run(self):
        s = self.make(ScriptedModel([]), max_context_chars=6000, max_actions=40)
        (s.session.repo_dir / "big.py").write_text(
            "\n".join("value_{} = {}".format(i, i) for i in range(600)), encoding="utf-8")
        replies = [call("read_file", path="big.py", start_line=i * 20 + 1, end_line=i * 20 + 20) for i in range(15)]
        replies.append(call("finish", summary="done"))
        s.controller.model = ScriptedModel(replies)
        s.run()
        sizes = [d["chars"] for k, d in s.events if k == "context"]
        self.assertTrue(sizes)
        self.assertTrue(all(n <= 6000 for n in sizes), sizes)
        self.assertEqual(s.session.state, "completed")

    def test_selected_code_for_the_task_is_in_the_repository_section(self):
        s = self.make(ScriptedModel([]))
        s.task = Task(id="t", title="Sell", request="Change the sell function.", starting_commit=s.task.starting_commit,
                      allowed_write_paths=("operations.py",))
        s.controller.task = s.task
        text = s.controller._repo_text()
        self.assertIn("Selected code for the task", text)
        self.assertIn("--- operations.py:1 sell", text)
        self.assertIn("    1 | def sell():", text)

    def test_read_log_is_part_of_the_trusted_state(self):
        model = ScriptedModel([call("read_file", path="operations.py", start_line=1, end_line=2),
                               call("finish", summary="done")])
        s = self.make(model)
        s.run()
        second = model.seen_messages[1][0]["content"]
        self.assertIn("Already done (do not repeat these): operations.py 1-2", second)

    def test_reads_are_refused_after_the_navigation_budget_until_an_edit(self):
        replies = [call("list_files", path=".") for _ in range(13)]
        replies.append(call("edit_file", path="operations.py", old_string="    return 1\n\ndef other",
                            new_string="    return 2\n\ndef other"))
        replies.append(call("finish", summary="done"))
        model = ScriptedModel(replies)
        s = self.make(model, max_actions=40)
        s.run()
        self.assertEqual(s.data["counters"]["navigation_refusals"], 1)
        refused = [m["content"] for m in model.seen_messages[13] if m["role"] == "tool"][-1]
        self.assertIn("navigation budget", refused)
        self.assertEqual(s.session.state, "completed")

    def test_same_failing_request_three_times_stops_the_run(self):
        bad = call("edit_file", path="operations.py", old_string="no such text", new_string="x")
        model = ScriptedModel([bad, bad, bad, call("finish", summary="never reached")])
        s = self.make(model, max_actions=20)
        s.run()
        self.assertEqual(s.session.state, "stopped")
        self.assertIn("same failing edit_file request 3 times", s.data["stop_reason"])
        self.assertEqual(s.data["counters"]["invalid_requests"], 3)
        self.assertEqual(len(model.seen_messages), 3)          # the model was not asked a fourth time

    def test_different_failures_do_not_trigger_the_repeat_stop(self):
        replies = [call("edit_file", path="operations.py", old_string="no such {}".format(i), new_string="x")
                   for i in range(4)]
        replies.append(call("finish", summary="done"))
        s = self.make(ScriptedModel(replies), max_actions=20)
        s.run()
        self.assertEqual(s.data["counters"]["invalid_requests"], 4)
        self.assertNotIn("failing", s.data.get("stop_reason", ""))

    def test_extract_tool_calls_reads_structured_calls(self):
        calls, from_text = extract_tool_calls(call("search", query="x"))
        self.assertEqual(calls, [("search", {"query": "x"})])
        self.assertFalse(from_text)


if __name__ == "__main__":
    unittest.main()
