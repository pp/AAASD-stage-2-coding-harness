"""Safe resume tests. A session is interrupted, then resumed from disk after a restart.

Resume must preserve consumed limits, recheck permissions and the saved working-copy state (including
the harness's own edits), and pause on any mismatch. An interrupted action is never replayed automatically.
"""
import datetime
import subprocess
import unittest
from pathlib import Path

from harness.controller import AgentController, ResumeRefused
from harness.session import SessionError, SessionStore
from harness.tasks import Task
from harness.tools import RepositoryTools
from tests.helpers import FakeEnv, ScriptedModel, call, cleanup, settings
from tests.test_controller import Setup

EDIT = call("edit_file", path="operations.py", old_string="    return 1\n\ndef other",
            new_string="    return 2\n\ndef other")


def interrupt_once(name):
    """A check that is interrupted (as by Ctrl-C) the first time it runs, then passes."""
    state = {"left": 1}

    def run(n):
        if state["left"]:
            state["left"] -= 1
            raise KeyboardInterrupt()
        from harness.sandbox import CheckResult
        return CheckResult(n, [], "passed", 0, "ok")
    return run


def reopen(s, model, env, **over):
    """A fresh controller on the saved session, as after a restart."""
    store = SessionStore(s.runs)
    session = store.open(s.session.id)
    st = settings(**over) if over else s.settings
    tools = RepositoryTools(root=session.repo_dir, allowed_write_paths=s.task.allowed_write_paths,
                            denied_paths=st.denied_paths, max_read_chars=2000, max_list_entries=50,
                            max_search_hits=50, starting_commit=s.task.starting_commit)
    task = Task.from_dict(session.data["task"])
    ctrl = AgentController(model=model, tools=tools, env=env, settings=st, store=store, session=session,
                           task=task, on_event=lambda k, d: None)
    return ctrl, session


class SafeResumeTest(unittest.TestCase):
    def setUp(self):
        self.made = []

    def tearDown(self):
        for s in self.made:
            s.close()

    def interrupted_session(self, **over):
        """Edit the file, then get interrupted by Ctrl-C during the regression check."""
        env = FakeEnv(results={"regression_core": interrupt_once("regression_core")})
        model = ScriptedModel([EDIT, call("finish", summary="done")])
        s = Setup(self, model, env, **over)
        self.made.append(s)
        state = s.run()
        self.assertEqual(state, "cancelled")
        self.assertEqual(s.session.state, "cancelled")
        return s

    def test_interrupted_check_is_never_replayed_and_resume_pauses_first(self):
        s = self.interrupted_session()
        entries = SessionStore(s.runs).journal_read(s.session)
        self.assertEqual(SessionStore.uncertain(entries)[0]["tool"], "run_check")

        env2 = FakeEnv()
        model2 = ScriptedModel([call("finish", summary="done")])
        ctrl, session = reopen(s, model2, env2)
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=False)
        self.assertIn("not replayed", " ".join(ctx.exception.problems))
        self.assertEqual(env2.ran, [])               # nothing was run
        self.assertEqual(model2.calls, 0)            # the model was not asked
        self.assertEqual(session.state, "paused")

    def test_acknowledged_resume_continues_and_reports_the_interruption_to_the_model(self):
        s = self.interrupted_session()
        env2 = FakeEnv()
        model2 = ScriptedModel([call("finish", summary="done")])
        ctrl, session = reopen(s, model2, env2)
        state = ctrl.resume(acknowledge=True)
        self.assertEqual(state, "completed")
        tool_texts = [m["content"] for m in model2.seen_messages[0] if m["role"] == "tool"]
        self.assertTrue(any("interrupted before its result was recorded" in t for t in tool_texts))
        # the only check run after resume is the required final check, not the interrupted one replayed
        self.assertEqual(env2.ran.count("regression_core"), 1)
        self.assertEqual(SessionStore.uncertain(SessionStore(s.runs).journal_read(session)), [])

    def test_working_copy_changed_outside_the_harness_pauses_until_acknowledged(self):
        s = self.interrupted_session()
        (s.session.repo_dir / "operations.py").write_text("def sell():\n    return 9\n", encoding="utf-8")
        ctrl, session = reopen(s, ScriptedModel([call("finish", summary="done")]), FakeEnv())
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=False)
        self.assertIn("working copy changed", " ".join(ctx.exception.problems))

    def test_harness_edit_is_part_of_the_saved_state(self):
        s = self.interrupted_session()
        ctrl, session = reopen(s, ScriptedModel([call("finish", summary="done")]), FakeEnv())
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=False)
        # the harness's own edit is part of the saved state: the stored fingerprint still matches it,
        # so the only refusal is the interrupted check
        self.assertNotIn("working copy changed", " ".join(ctx.exception.problems))
        self.assertIn("not replayed", " ".join(ctx.exception.problems))

    def test_change_of_permissions_pauses_resume(self):
        s = self.interrupted_session()
        ctrl, _ = reopen(s, ScriptedModel([]), FakeEnv(), denied_paths=(".git", "data", "docs"))
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=True)
        self.assertIn("denied paths changed", " ".join(ctx.exception.problems))

    def test_change_of_check_configuration_pauses_resume(self):
        s = self.interrupted_session()
        ctrl, session = reopen(s, ScriptedModel([]), FakeEnv(checks={"acceptance": ["other"], "regression_core": ["y"]}))
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=True)
        self.assertIn("configured checks changed", " ".join(ctx.exception.problems))

    def test_file_outside_the_task_scope_pauses_resume(self):
        s = self.interrupted_session()
        (s.session.repo_dir / "db.py").write_text("class DB:\n    x = 1\n", encoding="utf-8")
        ctrl, _ = reopen(s, ScriptedModel([]), FakeEnv())
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=True)
        self.assertIn("outside the task scope", " ".join(ctx.exception.problems))

    def test_working_copy_on_another_commit_pauses_resume(self):
        s = self.interrupted_session()
        repo = s.session.repo_dir
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "other"],
                       cwd=repo, check=True, capture_output=True)
        ctrl, _ = reopen(s, ScriptedModel([]), FakeEnv())
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=True)
        self.assertIn("starting commit", " ".join(ctx.exception.problems))

    def test_consumed_action_limit_is_preserved_and_not_reset(self):
        s = self.interrupted_session()
        used = s.data["counters"]["actions"]
        self.assertGreater(used, 0)
        ctrl, _ = reopen(s, ScriptedModel([]), FakeEnv(), max_actions=used)
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=True)
        self.assertIn("action limit already reached", " ".join(ctx.exception.problems))
        # a higher limit lets the run continue, and the count carries on from what was used
        ctrl, session = reopen(s, ScriptedModel([call("finish", summary="done")]), FakeEnv(), max_actions=used + 5)
        ctrl.resume(acknowledge=True)
        self.assertGreater(session.data["counters"]["actions"], used)

    def test_stale_running_session_with_an_unfinished_action_pauses(self):
        s = self.interrupted_session()
        store = SessionStore(s.runs)
        session = store.open(s.session.id)
        session.data["state"] = "running"
        session.data["owner_pid"] = 999999
        session.data["heartbeat"] = (datetime.datetime.now() - datetime.timedelta(minutes=10)).isoformat()
        store.save(session)
        store.journal_append(session, {"seq": 99, "phase": "started", "tool": "edit_file",
                                       "args": {"path": "operations.py"}})   # a crash mid-edit
        ctrl, _ = reopen(s, ScriptedModel([call("finish", summary="done")]), FakeEnv())
        with self.assertRaises(ResumeRefused) as ctx:
            ctrl.resume(acknowledge=False)
        self.assertIn("edit_file", " ".join(ctx.exception.problems))

    def test_a_completed_session_cannot_be_resumed(self):
        s = Setup(self, ScriptedModel([call("finish", summary="done")]))
        self.made.append(s)
        s.run()
        ctrl, _ = reopen(s, ScriptedModel([]), FakeEnv())
        with self.assertRaises(SessionError):
            ctrl.resume()

    def test_fresh_process_can_open_and_list_the_interrupted_session(self):
        s = self.interrupted_session()
        store = SessionStore(s.runs)
        rows = store.list()
        self.assertEqual(rows[0]["state"], "cancelled")
        session = store.open(s.session.id)
        self.assertEqual(session.task["id"], "t")
        self.assertEqual(session.data["starting_commit"], s.task.starting_commit)


if __name__ == "__main__":
    unittest.main()
