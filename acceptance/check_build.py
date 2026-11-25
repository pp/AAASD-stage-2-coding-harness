"""Build check: every Python file in the repository must compile.

Runs inside the sandbox on the working copy (REPO_DIR, default /work). It parses files with compile()
and writes no bytecode, so it does not change the workspace.
Exit code 0 = all files compile, 1 = at least one file does not.
"""
import os
import sys

REPO = os.environ.get("REPO_DIR", "/work")
SKIP_DIRS = {".git", "data", "backups", "documents", "logs", "venv", ".venv", "vendor", "__pycache__"}

failures = []
count = 0
for dirpath, dirnames, filenames in os.walk(REPO):
    dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
    for name in sorted(filenames):
        if not name.endswith(".py"):
            continue
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, REPO).replace(os.sep, "/")
        count += 1
        try:
            with open(path, "rb") as fh:
                source = fh.read()
            compile(source, rel, "exec")
        except (SyntaxError, ValueError) as exc:
            failures.append("{}: {}".format(rel, exc))

if failures:
    print("BUILD FAILED ({} of {} files)".format(len(failures), count))
    for f in failures:
        print(" - " + f)
    sys.exit(1)
print("BUILD PASSED ({} files compile)".format(count))
sys.exit(0)
