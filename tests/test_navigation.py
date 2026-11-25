"""Navigation tests: symbols, dependencies, file ranking, and the repository map."""
import os
import unittest
from pathlib import Path

from harness.navigation import SymbolIndex
from tests.helpers import cleanup, head_of, make_repo

FILES = {
    "db.py": (
        "import datetime\n\n"
        "class DB:\n"
        "    def register_advance(self, company_id, amount):\n"
        "        return amount\n\n"
        "    def use_advance(self, company_id, amount):\n"
        "        return amount\n\n"
        "def load_config():\n"
        "    return {}\n"
    ),
    "operations.py": (
        "from db import DB\n\n"
        "class Operations:\n"
        "    def sell(self, lines):\n"
        "        return lines\n"
    ),
    "app.py": (
        "import db\n"
        "from operations import Operations\n\n"
        "def api_advance():\n"
        "    return db.DB()\n"
    ),
    "broken.py": "def broken(:\n",
}


class NavigationTest(unittest.TestCase):
    def setUp(self):
        self.root = make_repo(FILES)
        self.index = SymbolIndex(self.root, denied=[".git", "data"])

    def tearDown(self):
        cleanup(self.root)

    def test_symbols_include_classes_functions_and_methods_with_lines(self):
        mods = self.index.modules()
        names = {(s.kind, s.name, s.line) for s in mods["db.py"].symbols}
        self.assertIn(("class", "DB", 3), names)
        self.assertIn(("method", "DB.register_advance", 4), names)
        self.assertIn(("function", "load_config", 10), names)

    def test_find_symbol_matches_fragments_case_insensitively(self):
        hits, more = self.index.find("ADVANCE", limit=10)
        self.assertFalse(more)
        self.assertTrue(any(h.startswith("db.py:4 method DB.register_advance") for h in hits))
        self.assertTrue(any(h.startswith("app.py:4 function api_advance") for h in hits))

    def test_dependencies_resolve_repository_modules_both_ways(self):
        deps = self.index.dependencies("operations.py")
        self.assertEqual(deps["imports"], ["db.py"])
        self.assertEqual(deps["imported_by"], ["app.py"])
        app = self.index.dependencies("app.py")
        self.assertEqual(sorted(app["imports"]), ["db.py", "operations.py"])

    def test_syntax_error_module_is_reported_not_fatal(self):
        mods = self.index.modules()
        self.assertEqual(mods["broken.py"].parse_error, "syntax error at line 1")

    def test_ranking_finds_the_modules_for_an_advance_request(self):
        ranked = [rel for rel, _ in self.index.rank("Add an optional expiry date to customer advances "
                                                   "and register_advance. use_advance must skip expired ones.")]
        self.assertEqual(ranked[0], "db.py")
        self.assertIn("app.py", ranked)

    def test_repo_map_is_bounded_and_puts_focus_first(self):
        full = self.index.repo_map(max_chars=5000, focus=["db.py"])
        self.assertGreater(len(full), 200)
        text = self.index.repo_map(max_chars=200, focus=["db.py"])
        self.assertIn("Likely relevant files:", text)
        self.assertLess(text.index("Likely relevant files:"), text.index("db.py: DB"))
        self.assertLessEqual(len(text), 200 + 80)   # the marker line is added after the cut
        self.assertIn("shortened", text)

    def test_repo_map_skips_denied_folders(self):
        Path(self.root, "data").mkdir()
        Path(self.root, "data", "secret.py").write_text("SECRET = 1\n")
        text = self.index.repo_map(max_chars=5000)
        self.assertNotIn("secret.py", text)


OPENSTOCK = os.environ.get("HARNESS_TARGET_REPO")


@unittest.skipUnless(OPENSTOCK and Path(OPENSTOCK).exists(), "set HARNESS_TARGET_REPO to run")
class OpenStockNavigationTest(unittest.TestCase):
    """The advance-expiry request must lead to db.py and app.py, and the git state must match the start."""

    def test_finds_advance_modules_and_dependencies(self):
        index = SymbolIndex(OPENSTOCK, denied=[".git", "data", "backups", "documents", "logs", "venv", ".venv"])
        ranked = [rel for rel, _ in index.rank("customer advances register_advance use_advance get_advances expiry")]
        self.assertIn("db.py", ranked[:2])
        self.assertIn("app.py", ranked)
        deps = index.dependencies("app.py")
        self.assertIn("db.py", deps["imports"])
        hits, _ = index.find("use_advance", 10)
        self.assertTrue(any(h.startswith("db.py:") for h in hits))
        self.assertEqual(head_of(Path(OPENSTOCK)), "abc87dccab96c78917b35add542284e61adc5f37")


if __name__ == "__main__":
    unittest.main()
