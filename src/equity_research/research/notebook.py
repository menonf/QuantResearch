"""What several notebooks would otherwise each define: a file name for a strategy, a table style, a printable path."""

from __future__ import annotations

from pathlib import Path

from research_core.settings import REPO_ROOT


def slug(name: str) -> str:
    """A strategy's name as a file name."""
    return "".join(c if c.isalnum() else "_" for c in name)


def red_negatives(v: object) -> str:
    """Negative numbers in red (a ``Styler.map`` function), as in S&P's dashboard."""
    return "color: #e34948" if isinstance(v, float) and v < 0 else ""


def shown(path: Path) -> str:
    """``path`` as a notebook prints it: relative to the repository where it is inside it, so no machine's folders."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
