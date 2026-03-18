"""
Path-pattern matching utilities for Seekr's ignore logic.

Lives in the domain layer because the matching rules are pure Python (stdlib
only) with no external dependencies, and both the application layer and the
infrastructure layer need them — placing them here avoids a circular import
where the application layer would otherwise have to reach into infrastructure.

Pattern types (applied to individual file or directory *names*, not full paths):
  - Extension match: ``.csv``       — name has that suffix (e.g. ``data.csv``)
  - Glob match:      ``*.log``      — fnmatch-style wildcard on the name
  - Exact match:     ``node_modules``— name equals the pattern exactly
"""

from __future__ import annotations

import fnmatch
from pathlib import Path


def matches_pattern(name: str, pattern: str) -> bool:
    """
    Return True if *name* (a single file or directory name) matches *pattern*.

    Three pattern types are recognised, checked in this order:

    1. **Extension match** — pattern starts with ``.`` and contains no wildcards
       (e.g. ``.csv``, ``.log``).  Matches any name whose suffix equals the pattern.
       Example: ``.csv`` matches ``data.csv`` but not ``backup.csv.gz``.

    2. **Glob match** — pattern contains ``*`` or ``?``
       (e.g. ``*.log``, ``tmp_*``, ``report_202?``).  Uses :func:`fnmatch.fnmatch`.

    3. **Exact match** — plain name with no wildcards (e.g. ``node_modules``).
       Matches only when the name equals the pattern exactly.
    """
    if pattern.startswith(".") and "*" not in pattern and "?" not in pattern:
        return Path(name).suffix == pattern
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(name, pattern)
    return name == pattern


def is_ignored(path: Path, patterns: set[str]) -> bool:
    """
    Return True if *path* should be ignored.

    Checks every component of the path against every pattern.  A path is
    ignored as soon as any component matches any pattern.

    See :func:`matches_pattern` for the supported pattern types.
    """
    return any(
        matches_pattern(part, pattern)
        for part in path.parts
        for pattern in patterns
    )
