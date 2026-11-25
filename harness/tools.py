"""Repository tools. Every path is checked here, not by the model or the prompt."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

from . import workspace
from .navigation import SymbolIndex

TEXT_SUFFIXES = (".py", ".html", ".md", ".txt", ".json", ".css", ".js", ".sql", ".toml", ".cfg", ".ini")
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_NEW_FILE_CHARS = 20000


class ToolError(Exception):
    """A request that failed. The message is returned to the model."""


class InvalidRequest(ToolError):
    """Bad tool name or arguments. Nothing was executed."""


class DeniedAction(ToolError):
    """A path or permission rule refused the action."""


class RepositoryTools:
    def __init__(self, root: Path, allowed_write_paths: Iterable[str], denied_paths: Iterable[str],
                 max_read_chars: int, max_list_entries: int, max_search_hits: int,
                 starting_commit: Optional[str] = None, max_symbol_hits: int = 50):
        self.root = Path(root).resolve()
        self.allowed_write = tuple(p.strip("/") for p in allowed_write_paths)
        self.denied = frozenset(denied_paths)
        self.max_read_chars = max_read_chars
        self.max_list_entries = max_list_entries
        self.max_search_hits = max_search_hits
        self.max_symbol_hits = max_symbol_hits
        self.starting_commit = starting_commit
        self.index = SymbolIndex(self.root, self.denied)
        self.shortened = 0

    # ── path rules ───────────────────────────────────────────
    def resolve(self, rel: str, for_write: bool = False) -> Path:
        if rel is None or (isinstance(rel, str) and rel.strip() in ("", ".", "./")):
            if for_write:
                raise InvalidRequest("path must name a file")
            return self.root                       # an empty path means the repository root
        if not isinstance(rel, str):
            raise InvalidRequest("path must be a string")
        p = Path(rel.strip())
        if p.is_absolute() or ".." in p.parts:
            raise DeniedAction("path '{}' is outside the repository".format(rel))
        target = (self.root / p).resolve()
        try:
            inside = target.relative_to(self.root)
        except ValueError:
            raise DeniedAction("path '{}' resolves outside the repository".format(rel))
        if any(part in self.denied for part in inside.parts):
            raise DeniedAction("path '{}' is not accessible to the agent".format(rel))
        if for_write and not self._writable(inside.as_posix()):
            raise DeniedAction("'{}' is not in the files this task may change".format(inside.as_posix()))
        return target

    def _writable(self, rel_posix: str) -> bool:
        return any(rel_posix == a or rel_posix.startswith(a + "/") for a in self.allowed_write)

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # ── reading and searching ────────────────────────────────
    def list_files(self, path: str = ".") -> str:
        base = self.resolve(path)
        if not base.is_dir():
            raise InvalidRequest("'{}' is not a directory".format(path))
        entries: List[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in self.denied and not d.startswith("."))
            for name in sorted(filenames):
                full = Path(dirpath) / name
                if name.endswith(".pyc") or name in self.denied:
                    continue
                entries.append(self._rel(full))
                if len(entries) >= self.max_list_entries:
                    self.shortened += 1
                    return "\n".join(entries) + "\n[listing shortened at {} entries]".format(self.max_list_entries)
        return "\n".join(entries) if entries else "(no files)"

    def read_file(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise InvalidRequest("'{}' is not a file".format(path))
        if target.stat().st_size > MAX_FILE_BYTES:
            raise InvalidRequest("'{}' is larger than {} bytes".format(path, MAX_FILE_BYTES))
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        out = []
        size = 0
        for n in range(start, end + 1):
            row = "{:5d} | {}".format(n, lines[n - 1])
            size += len(row) + 1
            if size > self.max_read_chars:
                self.shortened += 1
                out.append("[file shortened: showing lines {}-{} of {}; use start_line to continue]".format(
                    start, n - 1, len(lines)))
                break
            out.append(row)
        return "\n".join(out) if out else "(empty range)"

    def search(self, query: str, path: str = ".") -> str:
        if not isinstance(query, str) or not query:
            raise InvalidRequest("query must be a non-empty string")
        base = self.resolve(path)
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        hits: List[str] = []
        for full in self._searchable_files(base):
            try:
                if full.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = full.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for n, line in enumerate(text, 1):
                if pattern.search(line):
                    hits.append("{}:{}: {}".format(self._rel(full), n, line.strip()[:200]))
                    if len(hits) >= self.max_search_hits:
                        self.shortened += 1
                        return "\n".join(hits) + "\n[search shortened at {} hits]".format(self.max_search_hits)
        return "\n".join(hits) if hits else "(no matches)"

    def _searchable_files(self, base: Path):
        """The file itself when a file path is given; otherwise every text file under the folder."""
        if base.is_file():
            if base.suffix in TEXT_SUFFIXES and base.name not in self.denied:
                yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in self.denied and not d.startswith("."))
            for name in sorted(filenames):
                if name.endswith(TEXT_SUFFIXES) and name not in self.denied:
                    yield Path(dirpath) / name

    # ── navigation ───────────────────────────────────────────
    def find_symbol(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise InvalidRequest("name must be a non-empty string")
        hits, more = self.index.find(name, self.max_symbol_hits)
        if not hits:
            return "(no definitions match '{}')".format(name)
        text = "\n".join(hits)
        if more:
            self.shortened += 1
            text += "\n[symbol search shortened at {} hits]".format(self.max_symbol_hits)
        return text

    def dependencies(self, path: str) -> str:
        target = self.resolve(path)
        rel = self._rel(target)
        if not rel.endswith(".py"):
            raise InvalidRequest("dependencies works on Python modules (.py)")
        try:
            deps = self.index.dependencies(rel)
        except KeyError:
            raise InvalidRequest("'{}' is not a Python module in the repository".format(path))
        return "\n".join([
            "module: {}".format(rel),
            "imports (repository): {}".format(", ".join(deps["imports"]) or "(none)"),
            "imported by: {}".format(", ".join(deps["imported_by"]) or "(none)"),
            "imports (external): {}".format(", ".join(deps["external"]) or "(none)"),
        ])

    def git_state(self) -> str:
        if not self.starting_commit:
            raise InvalidRequest("this task has no starting commit recorded")
        state = workspace.git_state(self.root, self.starting_commit)
        changed = state["changed_files"]
        return "\n".join([
            "branch: {}".format(state["branch"]),
            "HEAD: {}".format(state["head"]),
            "starting commit: {}".format(state["starting_commit"]),
            "at starting commit: {}".format("yes" if state["at_starting_commit"] else "NO"),
            "changed files: {}".format(", ".join(changed) if changed else "(none)"),
        ])

    # ── editing ──────────────────────────────────────────────
    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        target = self.resolve(path, for_write=True)
        if not target.is_file():
            raise InvalidRequest("'{}' is not a file".format(path))
        if not isinstance(old_string, str) or old_string == "":
            raise InvalidRequest("old_string must be a non-empty string")
        if not isinstance(new_string, str):
            raise InvalidRequest("new_string must be a string")
        with target.open("r", encoding="utf-8", newline="") as fh:
            raw = fh.read()
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        old = old_string.replace("\r\n", "\n")
        new = new_string.replace("\r\n", "\n")
        count = text.count(old)
        if count == 0:
            raise InvalidRequest("old_string was not found in '{}'. Closest lines in the file (copy them exactly, "
                                 "with their indentation):\n{}".format(path, self._closest_lines(text, old)))
        if count > 1:
            raise InvalidRequest("old_string matches {} times in '{}'; include more surrounding lines".format(count, path))
        updated = text.replace(old, new, 1)
        if target.suffix == ".py":
            try:
                ast.parse(updated, filename=path)
            except SyntaxError as exc:
                raise InvalidRequest(
                    "edit refused: the result would not be valid Python ({} at line {}). '{}' was not changed.".format(
                        exc.msg, exc.lineno, path))
        if crlf:
            updated = updated.replace("\n", "\r\n")
        with target.open("w", encoding="utf-8", newline="") as fh:
            fh.write(updated)
        return "Edited {}: replaced 1 occurrence".format(path)

    @staticmethod
    def _closest_lines(text: str, old: str, count: int = 3) -> str:
        """The lines of the file that look most like the first line of old_string, with line numbers."""
        import difflib
        lines = text.split("\n")
        probe = next((ln.strip() for ln in old.split("\n") if ln.strip()), old.strip())
        scored = []
        for n, line in enumerate(lines, 1):
            ratio = difflib.SequenceMatcher(None, probe, line.strip()).ratio() if probe else 0.0
            if ratio > 0.5:
                scored.append((ratio, n))
        best = sorted(scored, reverse=True)[:count]
        if not best:
            return "(no line in the file is similar to the first line of old_string)"
        return "\n".join("{:5d} | {}".format(n, lines[n - 1]) for n in sorted(n for _, n in best))

    def replace_lines(self, path: str, start_line: int, end_line: int, new_text: str) -> str:
        """Replace lines start_line..end_line (inclusive, as numbered by read_file) with new_text."""
        target = self.resolve(path, for_write=True)
        if not target.is_file():
            raise InvalidRequest("'{}' is not a file".format(path))
        if not isinstance(start_line, int) or not isinstance(end_line, int) or start_line < 1 or end_line < start_line:
            raise InvalidRequest("start_line and end_line must be integers with 1 <= start_line <= end_line")
        if not isinstance(new_text, str):
            raise InvalidRequest("new_text must be a string")
        with target.open("r", encoding="utf-8", newline="") as fh:
            raw = fh.read()
        crlf = "\r\n" in raw
        lines = raw.replace("\r\n", "\n").split("\n")
        if lines and lines[-1] == "":
            lines.pop()                       # the final newline is not a line
        if end_line > len(lines):
            raise InvalidRequest("'{}' has {} lines; end_line {} is past the end".format(path, len(lines), end_line))
        replacement = new_text.replace("\r\n", "\n")
        if replacement.endswith("\n"):
            replacement = replacement[:-1]
        new_lines = replacement.split("\n") if replacement != "" else []
        updated_lines = lines[:start_line - 1] + new_lines + lines[end_line:]
        updated = "\n".join(updated_lines) + "\n"
        if target.suffix == ".py":
            try:
                ast.parse(updated, filename=path)
            except SyntaxError as exc:
                raise InvalidRequest(
                    "edit refused: the result would not be valid Python ({} at line {}). '{}' was not changed.".format(
                        exc.msg, exc.lineno, path))
        if crlf:
            updated = updated.replace("\n", "\r\n")
        with target.open("w", encoding="utf-8", newline="") as fh:
            fh.write(updated)
        return "Replaced lines {}-{} of {} ({} lines) with {} line(s)".format(
            start_line, end_line, path, end_line - start_line + 1, len(new_lines))

    def create_file(self, path: str, content: str) -> str:
        target = self.resolve(path, for_write=True)
        if target.exists():
            raise InvalidRequest("'{}' already exists; use edit_file to change it".format(path))
        if not isinstance(content, str):
            raise InvalidRequest("content must be a string")
        if len(content) > MAX_NEW_FILE_CHARS:
            raise InvalidRequest("content is longer than {} characters".format(MAX_NEW_FILE_CHARS))
        if target.suffix == ".py":
            try:
                ast.parse(content, filename=path)
            except SyntaxError as exc:
                raise InvalidRequest("new file refused: not valid Python ({} at line {})".format(exc.msg, exc.lineno))
        # The target is already inside an allowed path, so creating its parent folders stays in scope.
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="") as fh:
            fh.write(content)
        return "Created {} ({} characters)".format(path, len(content))
