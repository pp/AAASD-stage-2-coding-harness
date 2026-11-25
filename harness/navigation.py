"""Repository navigation: a Python symbol index, import dependencies, and file ranking for a task.

Simple and language-aware: it reads Python modules with ast. It never executes repository code.
All paths are relative to the repository root. The denied folders are skipped, as in the file tools.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

STOPWORDS = frozenset("""
the and with that this must when from into each should before after keep existing call calls
value values valid works working behavior behaviour only other their there these those which
while where then than also make used using does done have will would could about over under
rather without within feature task request change changes code file files repository
""".split())


@dataclass
class Symbol:
    kind: str       # class | function | method
    name: str       # qualified name, e.g. DB.register_advance
    line: int
    end: int = 0    # last line of the definition


@dataclass
class ModuleInfo:
    rel: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)   # dotted module names as written
    parse_error: Optional[str] = None


def _stem(token: str) -> str:
    """'advances' -> 'advance', so a plural in the request still matches a singular name in the code."""
    return token[:-1] if len(token) > 4 and token.endswith("s") else token


def module_name(rel: str) -> str:
    parts = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse_module(rel: str, text: str) -> ModuleInfo:
    info = ModuleInfo(rel=rel)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        info.parse_error = "syntax error at line {}".format(exc.lineno)
        return info
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            info.symbols.append(Symbol("class", node.name, node.lineno, node.end_lineno or node.lineno))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info.symbols.append(Symbol("method", "{}.{}".format(node.name, sub.name), sub.lineno,
                                               sub.end_lineno or sub.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.symbols.append(Symbol("function", node.name, node.lineno, node.end_lineno or node.lineno))
        elif isinstance(node, ast.Import):
            info.imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                info.imports.append(node.module)
    return info


class SymbolIndex:
    """Index of the Python modules in one repository copy. Re-parses a file only when it changed."""

    def __init__(self, root: Path, denied: Iterable[str], max_files: int = 2000):
        self.root = Path(root).resolve()
        self.denied = frozenset(denied)
        self.max_files = max_files
        self._cache: Dict[str, Tuple[Tuple[int, int], ModuleInfo]] = {}

    def _py_files(self) -> List[str]:
        found: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in self.denied and not d.startswith("."))
            for name in sorted(filenames):
                if name.endswith(".py") and name not in self.denied:
                    rel = (Path(dirpath) / name).relative_to(self.root).as_posix()
                    found.append(rel)
                    if len(found) >= self.max_files:
                        return found
        return found

    def modules(self) -> Dict[str, ModuleInfo]:
        out: Dict[str, ModuleInfo] = {}
        for rel in self._py_files():
            path = self.root / rel
            try:
                st = path.stat()
            except OSError:
                continue
            key = (st.st_mtime_ns, st.st_size)
            cached = self._cache.get(rel)
            if cached and cached[0] == key:
                out[rel] = cached[1]
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            info = parse_module(rel, text)
            self._cache[rel] = (key, info)
            out[rel] = info
        return out

    # ── lookups ─────────────────────────────────────────────
    def find(self, name: str, limit: int) -> Tuple[List[str], bool]:
        needle = name.strip().lower()
        hits: List[str] = []
        for rel, info in self.modules().items():
            for sym in info.symbols:
                if needle in sym.name.lower():
                    hits.append("{}:{} {} {}".format(rel, sym.line, sym.kind, sym.name))
                    if len(hits) >= limit:
                        return hits, True
        return hits, False

    def _resolve(self, dotted: str, known: Dict[str, ModuleInfo], by_module: Dict[str, str]) -> Optional[str]:
        # Longest prefix that is a repository module: "db.DB" -> "db"; "tests.helpers" -> "tests/helpers.py"
        parts = dotted.split(".")
        for n in range(len(parts), 0, -1):
            candidate = ".".join(parts[:n])
            if candidate in by_module:
                return by_module[candidate]
        return None

    def dependencies(self, rel: str) -> Dict[str, List[str]]:
        mods = self.modules()
        if rel not in mods:
            raise KeyError(rel)
        by_module = {module_name(r): r for r in mods}
        imports = sorted({self._resolve(m, mods, by_module) for m in mods[rel].imports} - {None, rel})
        imported_by = sorted(
            other for other, info in mods.items()
            if other != rel and any(self._resolve(m, mods, by_module) == rel for m in info.imports)
        )
        external = sorted({m.split(".")[0] for m in mods[rel].imports
                           if self._resolve(m, mods, by_module) is None})
        return {"imports": imports, "imported_by": imported_by, "external": external}

    # ── ranking and the repository map ─────────────────────
    @staticmethod
    def tokens(text: str) -> List[str]:
        seen: List[str] = []
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text.lower()):
            if tok not in STOPWORDS and tok not in seen:
                seen.append(tok)
        return seen[:40]

    def rank(self, text: str, limit: int = 5) -> List[Tuple[str, int]]:
        toks = self.tokens(text)
        mods = self.modules()
        scores: Dict[str, int] = {}
        for rel, info in mods.items():
            score = 0
            low_rel = rel.lower()
            body = None
            for tok in toks:
                stem = _stem(tok)
                if stem in low_rel:
                    score += 3
                for sym in info.symbols:
                    low = sym.name.lower()
                    if stem in low:
                        score += 2
                        if low.split(".")[-1] == stem:
                            score += 2
                if body is None:
                    body = (self.root / rel).read_text(encoding="utf-8", errors="replace").lower()
                if stem in body:
                    score += 1
            if score:
                scores[rel] = score
        top = sorted(scores, key=lambda r: (-scores[r], r))[:limit]
        # Neighbours of the best matches get a small boost, so the caller sees the modules they use.
        by_module = {module_name(r): r for r in mods}
        for rel in top[:3]:
            for m in mods[rel].imports:
                other = self._resolve(m, mods, by_module)
                if other and other not in top:
                    scores[other] = scores.get(other, 0) + 1
        ranked = sorted(scores, key=lambda r: (-scores[r], r))[:limit]
        return [(r, scores[r]) for r in ranked]

    def top_symbols(self, text: str, limit: int = 6) -> List[Tuple[str, Symbol, int]]:
        """Definitions whose names match the task text, best first. Methods are preferred over classes."""
        toks = self.tokens(text)
        scored: List[Tuple[str, Symbol, int]] = []
        for rel, info in self.modules().items():
            for sym in info.symbols:
                if sym.kind == "class":
                    continue
                low = sym.name.lower()
                last = low.split(".")[-1]
                score = 0
                for tok in toks:
                    stem = _stem(tok)
                    if stem in last:
                        score += 2
                    if tok == last:
                        score += 4
                if score:
                    scored.append((rel, sym, score))
        scored.sort(key=lambda t: (-t[2], t[0], t[1].line))
        return scored[:limit]

    def snippet(self, rel: str, sym: Symbol, max_lines: int = 80) -> str:
        """The source of one definition, with line numbers, as the file has it now."""
        lines = (self.root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(sym.end or sym.line, sym.line + max_lines - 1, len(lines))
        body = ["{:5d} | {}".format(n, lines[n - 1]) for n in range(sym.line, end + 1)]
        if (sym.end or sym.line) > end:
            body.append("[shortened: showing lines {}-{}]".format(sym.line, end))
        return "\n".join(body)

    def repo_map(self, max_chars: int, focus: Iterable[str] = ()) -> str:
        """Compact map: focus modules with line numbers first, then the other modules by name."""
        mods = self.modules()
        focus = [f for f in focus if f in mods]
        lines = ["Repository map (Python modules; [L n] = line number):"]
        if focus:
            lines.append("Likely relevant files:")
            for rel in focus:
                info = mods[rel]
                syms = ", ".join("{} [L{}]".format(s.name, s.line) for s in info.symbols[:60]) or "(no top-level symbols)"
                lines.append("- {}: {}".format(rel, syms))
        lines.append("Other modules:")
        for rel in sorted(mods):
            if rel in focus:
                continue
            info = mods[rel]
            names = ", ".join(s.name for s in info.symbols if s.kind != "method")[:300]
            lines.append("- {}: {}".format(rel, names or "(no top-level classes or functions)"))
        other_dirs = sorted({p.split("/")[0] for p in self._all_files() if "/" in p} - {"tests"})
        if other_dirs:
            lines.append("Folders: " + ", ".join(d + "/" for d in other_dirs[:30]))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[repository map shortened at {} characters]".format(max_chars)
        return text

    def _all_files(self) -> List[str]:
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in self.denied and not d.startswith("."))
            for name in filenames:
                if name not in self.denied:
                    out.append((Path(dirpath) / name).relative_to(self.root).as_posix())
            if len(out) > 5000:
                break
        return out
