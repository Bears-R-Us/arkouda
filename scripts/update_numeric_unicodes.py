#!/usr/bin/env python3
"""
Regenerate src/NumericUnicodes.chpl to match the current Python interpreter's
definition of isnumeric(). Run as:

    $CHPL_HOME/util/config/run-in-venv-with-python-bindings.bash \
        python3 scripts/update_numeric_unicodes.py --in-place src/NumericUnicodes.chpl
"""

from chapel import Variable, each_matching
from chapel.replace import run


def _build_set_literal() -> str:
    """Return a Chapel set literal containing every codepoint chr(i).isnumeric() accepts."""
    codepoints = sorted(i for i in range(0x110000) if chr(i).isnumeric())
    lines = ["const allNumericUnicodes = {"]
    for i in range(0, len(codepoints), 10):
        chunk = codepoints[i : i + 10]
        lines.append("        " + ", ".join(f"0x{c:x}" for c in chunk) + " ,")
    lines.append("      }")
    return "\n".join(lines)


def replace_numeric_unicodes(rc, root):
    for var, _ in each_matching(root, Variable):
        if var.name() == "allNumericUnicodes":
            yield (var, _build_set_literal())


run(replace_numeric_unicodes)
