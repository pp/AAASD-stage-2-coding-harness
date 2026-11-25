"""Acceptance checks: each must fail on the starting commit and pass after a correct change.

They need the target repository (OpenStock at abc87dc, with data/openstock.db seeded for the regression
suites). Set HARNESS_TARGET_REPO to its path. Without it these tests are skipped.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent
TARGET = os.environ.get("HARNESS_TARGET_REPO")
ADVANCE_PATCH = HARNESS / "tests" / "fixtures" / "advance_expiry_reference.patch"

STAGE1_ANCHOR = (
    "    def _sell(self, *, customer_id, invoice_date, lines, operator,\n"
    "              stock_no, payment_method, invoice_no, allow_below_min=False):\n"
    "        vat_rate = self.db.get_vat_rate()\n"
)
STAGE1_GUARD = STAGE1_ANCHOR + (
    "        for e in lines:\n"
    "            if float(e[\"qty\"]) <= 0:\n"
    "                raise OperationError(\"Quantity must be greater than zero.\")\n"
)


def fresh_copy(parent: Path) -> Path:
    dst = parent / "repo"
    shutil.copytree(TARGET, dst, ignore=shutil.ignore_patterns(
        "venv", "vendor", "__pycache__", "backups", "_test*.db*"))
    return dst


def run_check(repo: Path, script: Path):
    env = dict(os.environ, REPO_DIR=str(repo))
    return subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300)


@unittest.skipUnless(TARGET and Path(TARGET).exists(), "set HARNESS_TARGET_REPO to run")
class Stage1AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="acc-stage1-"))
        self.repo = fresh_copy(self.tmp)
        self.check = HARNESS / "acceptance" / "check_sale_quantity.py"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fails_on_starting_code(self):
        proc = run_check(self.repo, self.check)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("ACCEPTANCE FAILED", proc.stdout)

    def test_passes_after_reference_fix(self):
        ops = self.repo / "operations.py"
        src = ops.read_text(encoding="utf-8")
        self.assertIn(STAGE1_ANCHOR, src)
        ops.write_text(src.replace(STAGE1_ANCHOR, STAGE1_GUARD, 1), encoding="utf-8")
        proc = run_check(self.repo, self.check)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("ACCEPTANCE PASSED", proc.stdout)


@unittest.skipUnless(TARGET and Path(TARGET).exists(), "set HARNESS_TARGET_REPO to run")
class AdvanceExpiryAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="acc-stage2-"))
        self.repo = fresh_copy(self.tmp)
        self.check = HARNESS / "acceptance" / "check_advance_expiry.py"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def apply_reference(self):
        proc = subprocess.run(["git", "apply", str(ADVANCE_PATCH)], cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_feature_check_fails_on_starting_code_for_the_expected_reason(self):
        proc = run_check(self.repo, self.check)
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, out)
        self.assertIn("ACCEPTANCE FAILED", out)
        self.assertIn("expires_on", out)      # the feature is missing, not a setup problem

    def test_feature_check_passes_after_the_reference_change(self):
        self.apply_reference()
        proc = run_check(self.repo, self.check)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("ACCEPTANCE PASSED", proc.stdout)

    def test_regression_suites_still_pass_after_the_reference_change(self):
        self.apply_reference()
        for script in ("tests/test_core.py", "tests/test_reversal_advances.py"):
            proc = subprocess.run([sys.executable, script], cwd=self.repo, capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, script + "\n" + proc.stdout[-2000:] + proc.stderr[-2000:])


class BuildCheckTest(unittest.TestCase):
    def run_build(self, files):
        repo = Path(tempfile.mkdtemp(prefix="build-check-"))
        try:
            for rel, content in files.items():
                p = repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content.encode("utf-8"))
            env = dict(os.environ, REPO_DIR=str(repo))
            return subprocess.run([sys.executable, str(HARNESS / "acceptance" / "check_build.py")],
                                  env=env, capture_output=True, text=True)
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_build_passes_when_all_python_compiles(self):
        proc = self.run_build({"a.py": "x = 1\n", "pkg/b.py": "def f():\n    return 2\n"})
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("BUILD PASSED (2 files", proc.stdout)

    def test_build_fails_on_a_syntax_error_and_names_the_file(self):
        proc = self.run_build({"a.py": "x = 1\n", "bad.py": "def (:\n"})
        self.assertEqual(proc.returncode, 1)
        self.assertIn("bad.py", proc.stdout)


if __name__ == "__main__":
    unittest.main()
