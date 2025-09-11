"""Pytest guard that forbids `>>> ak.connect(...)` inside NumPy-style Examples sections.

This test scans all Python files (excluding a small ignore set), parses triple-quoted
docstrings, finds their "Examples" sections, and flags any doctest prompt lines that
call ``ak.connect``. Failures report file, line number, and a small context block.
"""

from pathlib import Path
import re
from typing import Generator, List, Tuple

# --- Heuristics that mirror the stripper script ---
TRIPLE_QUOTE_RE = re.compile(r'(?P<q>"""|\'\'\')(?P<body>.*?)(?P=q)', re.DOTALL)
SECTION_HEADER_RE = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9 _-]*)[ \t]*\n[ \t]*[-=~]{3,}[ \t]*\n", re.MULTILINE
)
DOCTEST_CONNECT_RE = re.compile(r"^\s*>{2,}\s*ak\.connect\(", re.MULTILINE)

# Optional: ignore some paths (examples, build outputs, vendored code, etc.)
IGNORES = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "site-packages",
    "__pycache__",
    "docs/_build",
    "client.py",
}


def should_skip(path: Path) -> bool:
    """
    Determine whether a given file path should be skipped based on ignore rules.

    Parameters
    ----------
    path : Path
        Path to the file, **relative** to the repository root.

    Returns
    -------
    bool
        ``True`` if the file lives under any of the ignored directories defined
        in :data:`IGNORES`, otherwise ``False``.
    """
    parts = set(path.parts)
    return bool(parts & IGNORES)


def examples_spans_in_doc(doc: str) -> Generator[tuple[int, int], None, None]:
    """
    Yield (start, end) index pairs for "Examples" sections within a docstring body.

    This inspects the given docstring text (without surrounding triple quotes)
    for NumPy-style section headers. For each section named "Examples",
    the start and end offsets of the section's content within the docstring
    are yielded.

    Parameters
    ----------
    doc : str
        The full text of a docstring body (without triple quotes).

    Yields
    ------
    int
        The start and end character indices (relative to ``doc``) of each
        Examples section found.

    Returns
    -------
    None
        The generator returns ``None`` upon completion.
    """
    sections = list(SECTION_HEADER_RE.finditer(doc))
    if not sections:
        return None
    for i, m in enumerate(sections):
        name = m.group(1).strip().lower()
        start = m.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(doc)
        if name == "examples":
            yield start, end


def find_offenses_with_lines(text: str) -> List[Tuple[int, str, str]]:
    """
    Find forbidden doctest prompt lines inside Examples sections of docstrings.

    Parameters
    ----------
    text : str
        The full text of a Python source file.

    Returns
    -------
    list[tuple[int, str, str]]
        A list of ``(line_number, matched_line, context_block)`` tuples for each
        offending line inside an Examples section. ``context_block`` contains the
        offending line plus one line of context above and below.
    """
    offenses: List[Tuple[int, str, str]] = []
    for m in TRIPLE_QUOTE_RE.finditer(text):
        doc_start = m.start()
        body = m.group("body")
        for s, e in examples_spans_in_doc(body):
            # Search within this Examples slice
            slice_ = body[s:e]
            for bad in DOCTEST_CONNECT_RE.finditer(slice_):
                # Compute absolute index in file text
                absolute_idx = doc_start + s + bad.start()
                # Determine line bounds for matched line
                line_start = text.rfind("\n", 0, absolute_idx) + 1
                line_end = text.find("\n", absolute_idx)
                if line_end == -1:
                    line_end = len(text)
                # Compute line number
                line_no = text.count("\n", 0, line_start) + 1
                matched_line = text[line_start:line_end].rstrip()

                # Build a small context block: previous, current, next lines
                # Previous line
                prev_start = text.rfind("\n", 0, max(0, line_start - 1)) + 1
                prev_end = line_start - 1 if line_start > 0 else 0
                prev_line = text[prev_start:prev_end].rstrip() if prev_end > prev_start else ""

                # Next line
                next_start = line_end + 1 if line_end < len(text) else len(text)
                next_end = text.find("\n", next_start)
                if next_end == -1:
                    next_end = len(text)
                next_line = text[next_start:next_end].rstrip() if next_end > next_start else ""

                context = "\n".join(
                    [
                        f"{line_no - 1:>6}: {prev_line}" if prev_line else "",
                        f"{line_no:>6}: {matched_line}",
                        f"{line_no + 1:>6}: {next_line}" if next_line else "",
                    ]
                ).strip("\n")

                offenses.append((line_no, matched_line, context))
    return offenses


def test_no_ak_connect_doctest():
    """
    Assert no ``>>> ak.connect(...)`` doctest prompts exist inside Examples sections.

    The test walks through all Python files in the repository (excluding ignored
    paths), parses triple-quoted docstrings, and checks their "Examples" sections
    for doctest prompt lines calling ``ak.connect``. If any such lines are found,
    the test fails and reports the file, line number, and a small context block.

    Raises
    ------
    AssertionError
        If one or more offending lines are found inside Examples sections.
    """
    repo_root = Path(__file__).resolve().parents[1]
    offenders: List[Tuple[Path, int, str, str]] = []  # (path, line, matched_line, context)

    for py in repo_root.rglob("*.py"):
        rel = py.relative_to(repo_root)
        if should_skip(rel):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        hits = find_offenses_with_lines(text)
        for line_no, matched_line, context in hits:
            offenders.append((rel, line_no, matched_line, context))

    assert not offenders, (
        "Forbidden doctest prompts found INSIDE docstring 'Examples' sections "
        "(remove lines like `>>> ak.connect(...)`).\n\n"
        + "\n\n".join(
            f"""{path}:{line}
{context}"""
            for path, line, matched_line, context in offenders
        )
    )
