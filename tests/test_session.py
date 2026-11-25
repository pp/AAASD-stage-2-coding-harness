"""Session storage tests: saved details survive a restart, the lock, and the uncertain-action rule."""
import datetime
import json
import unittest
from pathlib import Path

from harness.session import SessionBusy, SessionStore
from tests.helpers import cleanup


class SessionStorageTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__import__("tempfile").mkdtemp(prefix="harness-sessions-"))
        self.store = SessionStore(self.root)
        self.session = self.store.create(session_id="s1", data={
            "task": {"id": "t", "title": "Expiry"}, "starting_commit": "abc", "counters": {"actions": 0}})

    def tearDown(self):
        cleanup(self.root)

    def test_task_steps_messages_and_events_survive_a_restart(self):
        self.store.journal_append(self.session, {"seq": 1, "phase": "started", "tool": "edit_file",
                                                 "args": {"path": "db.py"}})
        self.store.journal_append(self.session, {"seq": 1, "phase": "finished", "tool": "edit_file",
                                                 "result": "Edited db.py"})
        self.store.messages_save(self.session, [{"role": "user", "content": "go"}])
        self.store.events_append(self.session, "tool", {"name": "edit_file"})
        self.store.release(self.session, "paused", stop_reason="test")

        reopened = SessionStore(self.root).open("s1")          # a new store object: as after a restart
        self.assertEqual(reopened.state, "paused")
        self.assertEqual(reopened.task["title"], "Expiry")
        self.assertEqual(reopened.data["stop_reason"], "test")
        entries = SessionStore(self.root).journal_read(reopened)
        self.assertEqual([e["phase"] for e in entries], ["started", "finished"])
        self.assertEqual(SessionStore(self.root).messages_load(reopened), [{"role": "user", "content": "go"}])
        self.assertEqual(SessionStore(self.root).events_read(reopened)[0]["event"], "tool")

    def test_listing_shows_saved_sessions(self):
        rows = self.store.list()
        self.assertEqual([r["id"] for r in rows], ["s1"])
        self.assertEqual(rows[0]["task"], "Expiry")

    def test_live_run_in_another_process_blocks_acquire(self):
        self.store.acquire(self.session, stale_s=30)
        other = SessionStore(self.root).open("s1")
        with self.assertRaises(SessionBusy):
            SessionStore(self.root).acquire(other, stale_s=30)

    def test_stale_heartbeat_allows_takeover(self):
        self.store.acquire(self.session, stale_s=30)
        old = datetime.datetime.now() - datetime.timedelta(minutes=5)
        self.session.data["heartbeat"] = old.isoformat(timespec="seconds")
        self.store.save(self.session)
        other = SessionStore(self.root).open("s1")
        SessionStore(self.root).acquire(other, stale_s=30)            # no exception
        self.assertEqual(other.state, "running")

    def test_uncertain_actions_are_started_or_interrupted_without_a_finish(self):
        entries = [
            {"seq": 1, "phase": "started", "tool": "edit_file"},
            {"seq": 1, "phase": "finished", "tool": "edit_file"},
            {"seq": 2, "phase": "started", "tool": "run_check"},            # no finish: uncertain
            {"seq": 3, "phase": "started", "tool": "run_check"},
            {"seq": 3, "phase": "interrupted", "tool": "run_check"},        # interrupted: uncertain
            {"seq": 4, "phase": "started", "tool": "run_check"},
            {"seq": 4, "phase": "interrupted", "tool": "run_check"},
            {"seq": 4, "phase": "acknowledged", "tool": "run_check"},       # acknowledged: settled
        ]
        pending = SessionStore.uncertain(entries)
        self.assertEqual([e["seq"] for e in pending], [2, 3])

    def test_sequence_numbers_follow_started_records(self):
        self.assertEqual(self.store.next_seq(self.session), 1)
        self.store.journal_append(self.session, {"seq": 1, "phase": "started", "tool": "edit_file"})
        self.assertEqual(self.store.next_seq(self.session), 2)


if __name__ == "__main__":
    unittest.main()
