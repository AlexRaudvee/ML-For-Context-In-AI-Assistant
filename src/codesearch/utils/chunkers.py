# src/codesearch/utils/chunkers.py
from __future__ import annotations
import ast
from pathlib import Path
from typing import Dict, List, Optional

LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".kt": "kotlin", ".cpp": "cpp", ".c": "c",
    ".rs": "rust", ".go": "go"
}

def detect_lang(path: str) -> str:
    return LANG_MAP.get(Path(path).suffix.lower(), "text")

def _leading_comments(lines: List[str], start_idx: int, max_lines: int = 5) -> List[str]:
    """Collect up to `max_lines` of consecutive comment lines above the start, stopping at blank/non-comment."""
    out: List[str] = []
    i = start_idx - 1
    while i >= 0 and len(out) < max_lines:
        s = lines[i].strip()
        if not s:
            break
        if s.startswith("#"):
            out.append(lines[i])
            i -= 1
            continue
        break
    return list(reversed(out))

def chunk_by_lines(
    text: str,
    path: str,
    lang: str,
    window: int = 120,
    overlap: int = 30,
    min_chars: int = 50,
) -> List[Dict]:
    """Generic sliding window chunker by lines with overlap."""
    lines = text.splitlines()
    if window <= 0: window = 120
    if overlap < 0: overlap = 0
    step = max(1, window - overlap)

    chunks: List[Dict] = []
    n_parts = (len(lines) + step - 1) // step
    part = 0
    for start in range(0, len(lines), step):
        end = min(start + window, len(lines))
        blob = "\n".join(lines[start:end]).strip()
        if len(blob) < min_chars:
            continue
        part += 1
        chunks.append({
            "text": blob,
            "path": str(path),
            "lang": lang,
            "kind": "window",
            "symbol": None,
            "start_line": start + 1,
            "end_line": end,
            "part": part,
            "n_parts": n_parts,
        })
    return chunks

def chunk_python_source(
    text: str,
    path: str,
    max_func_lines: int = 200,
    window: int = 120,
    overlap: int = 30,
    include_comments: bool = True,
    min_chars: int = 50,
) -> List[Dict]:
    """Python-aware chunking: split into functions/classes, then window long ones; add leading comments."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_by_lines(text, path, "python", window, overlap, min_chars)

    lines = text.splitlines()
    n = len(lines)
    covered = [False] * n
    chunks: List[Dict] = []

    def add_chunk(start: int, end: int, name: Optional[str], kind: str,
                  part: Optional[int] = None, n_parts: Optional[int] = None):
        raw = lines[start:end]
        if include_comments:
            raw = _leading_comments(lines, start) + raw
        blob = "\n".join(raw).strip()
        if len(blob) < min_chars:
            return
        chunk = {
            "text": blob,
            "path": str(path),
            "lang": "python",
            "kind": kind,                  # "function" | "class" | "module"
            "symbol": name,                # function/class name
            "start_line": start + 1,
            "end_line": end,
        }
        if part is not None and n_parts is not None:
            chunk["part"] = part
            chunk["n_parts"] = n_parts
        chunks.append(chunk)
        for i in range(start, end):
            covered[i] = True

    # Functions/classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = max(0, node.lineno - 1)
            end = getattr(node, "end_lineno", None)
            if end is None:
                continue
            name = getattr(node, "name", None)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if (end - start) <= max_func_lines:
                add_chunk(start, end, name, kind)
            else:
                # Sliding windows for long defs
                win = max(10, window)
                step = max(1, win - overlap)
                total = 0
                s = start
                spans = []
                while s < end:
                    e = min(s + win, end)
                    spans.append((s, e))
                    if e >= end:
                        break
                    s += step
                total = len(spans)
                for idx, (s, e) in enumerate(spans, start=1):
                    add_chunk(s, e, name, kind, part=idx, n_parts=total)

    # Remaining module-level code blocks (imports, globals, scripts)
    i = 0
    while i < n:
        if not covered[i] and lines[i].strip():
            j = i + 1
            while j < n and not covered[j]:
                j += 1
            # Skip tiny/mostly blank blocks
            block = [ln for ln in lines[i:j] if ln.strip()]
            if len(block) >= 3:
                add_chunk(i, j, name=None, kind="module")
            i = j
        else:
            i += 1

    # Fallback if nothing parsed meaningfully
    if not chunks:
        return chunk_by_lines(text, path, "python", window, overlap, min_chars)
    return chunks

def chunk_file_auto(
    path: str,
    max_func_lines: int = 200,
    window: int = 120,
    overlap: int = 30,
    include_comments: bool = True,
    min_chars: int = 50,
) -> List[Dict]:
    """Dispatch by language; Python uses AST, others use line windows."""
    lang = detect_lang(path)
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if lang == "python":
        return chunk_python_source(
            text=text,
            path=path,
            max_func_lines=max_func_lines,
            window=window,
            overlap=overlap,
            include_comments=include_comments,
            min_chars=min_chars,
        )
    # Generic fallback for other languages
    return chunk_by_lines(text, path, lang, window, overlap, min_chars)
