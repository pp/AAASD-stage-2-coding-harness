"""Unit tests for repository tools: reads, searches, edits, new files, navigation, and path rules."""
import os
import unittest
from pathlib import Path

from harness.tools import DeniedAction, InvalidRequest, RepositoryTools
from tests.helpers import cleanup, head_of, make_repo


def can_symlink():
    probe = Path(make_repo({"a.txt": "x"})) / "probe"
    try:
        probe.symlink_to(probe.parent / "a.txt")
        return True
    except OSError:
        return False
    finally:
        cleanup(probe.parent)


class RepositoryToolsTest(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({
            "operations.py": "def sell():\n    return 1\n\ndef other():\n    return 1\n",
            "db.py": "class DB:\n    pass\n",
            "data/openstock.db": "secret",
        })
        self.outside = Path(self.root).parent / "outside-secret.txt"
        self.outside.write_text("nope")
        self.start = head_of(self.root)
        self.tools = RepositoryTools(
            root=self.root, allowed_write_paths=["operations.py", "migrations"],
            denied_paths=[".git", "data"], max_read_chars=200, max_list_entries=50, max_search_hits=50,
            starting_commit=self.start,
        )

    def tearDown(self):
        cleanup(self.root)
        self.outside.unlink(missing_ok=True)

    # ── Stage 1 behaviour ───────────────────────────────────
    def test_read_file_numbers_lines(self):
        out = self.tools.read_file("operations.py", 1, 2)
        self.assertIn("1 | def sell():", out)
        self.assertIn("2 |     return 1", out)
        self.assertNotIn("def other", out)

    def test_search_finds_lines_in_repo_only(self):
        out = self.tools.search("return 1")
        self.assertIn("operations.py:2:", out)
        self.assertNotIn("data/", out)
        self.assertNotIn(".git", out)

    def test_search_inside_a_single_file(self):
        out = self.tools.search("return 1", "operations.py")
        self.assertIn("operations.py:2:", out)
        self.assertNotIn("db.py", out)

    def test_empty_path_means_the_repository_root(self):
        self.assertIn("operations.py", self.tools.list_files(""))
        self.assertIn("operations.py:2:", self.tools.search("return 1", ""))

    def test_list_files_hides_denied_paths(self):
        out = self.tools.list_files(".")
        self.assertIn("operations.py", out)
        self.assertIn("db.py", out)
        self.assertNotIn("openstock.db", out)

    def test_edit_allowed_file(self):
        self.tools.edit_file("operations.py", "return 1\n\ndef other", "return 2\n\ndef other")
        self.assertIn("return 2", (Path(self.root) / "operations.py").read_text())

    def test_edit_rejects_file_outside_task_scope(self):
        with self.assertRaises(DeniedAction):
            self.tools.edit_file("db.py", "pass", "return 1")
        self.assertEqual((Path(self.root) / "db.py").read_text(), "class DB:\n    pass\n")

    def test_rejects_parent_directory_escape(self):
        with self.assertRaises(DeniedAction):
            self.tools.read_file("../outside-secret.txt")

    def test_rejects_absolute_path(self):
        with self.assertRaises(DeniedAction):
            self.tools.read_file(str(self.outside))

    @unittest.skipUnless(can_symlink(), "creating symlinks needs Windows developer mode or admin rights")
    def test_rejects_symlink_escape(self):
        link = Path(self.root) / "link.txt"
        link.symlink_to(self.outside)
        with self.assertRaises(DeniedAction):
            self.tools.read_file("link.txt")

    def test_rejects_denied_folders(self):
        with self.assertRaises(DeniedAction):
            self.tools.read_file("data/openstock.db")
        with self.assertRaises(DeniedAction):
            self.tools.read_file(".git/config")

    def test_edit_needs_exactly_one_match(self):
        with self.assertRaises(InvalidRequest):
            self.tools.edit_file("operations.py", "return 1", "return 2")
        with self.assertRaises(InvalidRequest):
            self.tools.edit_file("operations.py", "no such text", "x")

    def test_edit_refuses_invalid_python_and_leaves_file_unchanged(self):
        before = (Path(self.root) / "operations.py").read_text()
        with self.assertRaises(InvalidRequest) as ctx:
            self.tools.edit_file("operations.py", "    return 1\n\ndef other", "    if x <=:\n\ndef other")
        self.assertIn("would not be valid Python", str(ctx.exception))
        self.assertEqual((Path(self.root) / "operations.py").read_text(), before)

    def test_read_is_bounded_and_marked(self):
        Path(self.root, "big.py").write_text("\n".join("x = {}".format(i) for i in range(500)))
        out = self.tools.read_file("big.py")
        self.assertIn("file shortened", out)
        self.assertLessEqual(len(out), 200 + 200)

    # ── Stage 2: edits keep line endings ────────────────────
    def test_failed_match_shows_the_closest_lines_to_copy(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.tools.edit_file("operations.py", "def  sell():", "def sell2():")
        self.assertIn("1 | def sell():", str(ctx.exception))

    def test_replace_lines_by_number(self):
        self.tools.replace_lines("operations.py", 2, 2, "    return 7")
        text = (Path(self.root) / "operations.py").read_text()
        self.assertEqual(text, "def sell():\n    return 7\n\ndef other():\n    return 1\n")

    def test_replace_lines_refuses_bad_range_and_invalid_python(self):
        before = (Path(self.root) / "operations.py").read_text()
        with self.assertRaises(InvalidRequest):
            self.tools.replace_lines("operations.py", 3, 2, "x")
        with self.assertRaises(InvalidRequest):
            self.tools.replace_lines("operations.py", 1, 99, "x")
        with self.assertRaises(InvalidRequest) as ctx:
            self.tools.replace_lines("operations.py", 2, 2, "    return (")
        self.assertIn("would not be valid Python", str(ctx.exception))
        self.assertEqual((Path(self.root) / "operations.py").read_text(), before)

    def test_replace_lines_outside_scope_is_denied(self):
        with self.assertRaises(DeniedAction):
            self.tools.replace_lines("db.py", 1, 1, "x = 1")

    def test_multiline_edit_matches_crlf_file_and_keeps_crlf(self):
        path = Path(self.root) / "operations.py"
        path.write_bytes(b"def sell():\r\n    return 1\r\n\r\ndef other():\r\n    return 1\r\n")
        self.tools.edit_file("operations.py", "return 1\n\ndef other", "return 2\n\ndef other")
        data = path.read_bytes()
        self.assertIn(b"return 2\r\n\r\ndef other", data)
        self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))

    # ── Stage 2: new files ──────────────────────────────────
    def test_create_file_in_allowed_folder(self):
        self.tools.create_file("migrations/011_x.sql", "ALTER TABLE t ADD COLUMN c TEXT;\n")
        self.assertTrue((Path(self.root) / "migrations/011_x.sql").exists())

    def test_create_file_refuses_existing_file(self):
        with self.assertRaises(InvalidRequest):
            self.tools.create_file("operations.py", "x = 1\n")
        self.assertIn("def sell", (Path(self.root) / "operations.py").read_text())

    def test_create_file_outside_scope_is_denied(self):
        with self.assertRaises(DeniedAction):
            self.tools.create_file("evil.py", "x = 1\n")
        self.assertFalse((Path(self.root) / "evil.py").exists())

    def test_create_file_refuses_invalid_python(self):
        with self.assertRaises(InvalidRequest):
            self.tools.create_file("migrations/bad.py", "def (:\n")
        self.assertFalse((Path(self.root) / "migrations/bad.py").exists())

    # ── Stage 2: navigation ─────────────────────────────────
    def test_find_symbol_reports_file_line_and_kind(self):
        out = self.tools.find_symbol("sell")
        self.assertIn("operations.py:1 function sell", out)

    def test_dependencies_lists_imports_and_importers(self):
        Path(self.root, "app.py").write_text("import db\nfrom db import DB\n", encoding="utf-8")
        out = self.tools.dependencies("db.py")
        self.assertIn("imported by: app.py", out)
        out = self.tools.dependencies("app.py")
        self.assertIn("imports (repository): db.py", out)

    def test_git_state_reports_starting_commit(self):
        out = self.tools.git_state()
        self.assertIn("at starting commit: yes", out)
        self.assertIn("changed files: (none)", out)
        Path(self.root, "operations.py").write_text("def sell():\n    return 3\n", encoding="utf-8")
        out = self.tools.git_state()
        self.assertIn("changed files: operations.py", out)


if __name__ == "__main__":
    unittest.main()
