"""
Path exclusion for indexing — default ignore list and .seekrignore loading.

Keeps the index small by skipping build artifacts, caches, dependencies,
and other irrelevant directories. Users can customize via:
  - Global: <data-dir>/.seekrignore (e.g. ~/.seekr/.seekrignore)
  - Local:  <index-root>/.seekrignore (per-project)
  - CLI:    --exclude "pattern1,pattern2"

Pattern matching rules (applied to file and directory *names*, not full paths):
  - Exact match:     ``node_modules`` — name equals the pattern exactly
  - Extension match: ``.csv``         — name has that suffix (e.g. ``data.csv``)
  - Glob match:      ``*.log``, ``tmp_*`` — fnmatch-style wildcard on the name
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from seekr.domain.patterns import is_ignored, matches_pattern  # noqa: F401  — re-exported

logger = logging.getLogger(__name__)

# Directory and file names to ignore when they appear as a path component.
# Keeps index relevant and avoids explosion from node_modules, .git, etc.
DEFAULT_IGNORE_PATTERNS = frozenset({
    # VCS and metadata
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    # Python
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "egg-info",
    ".eggs",
    # Node / frontend
    "node_modules",
    ".next",
    ".nuxt",
    ".output",
    "dist",
    ".parcel-cache",
    ".cache",
    ".turbo",
    # Rust / Go / C
    "target",      # Rust
    "vendor",      # Go
    "build",
    "out",
    # IDE / editor
    ".idea",
    ".vscode",
    ".vs",
    # OS / misc
    ".DS_Store",
    "Thumbs.db",
    # Common build/cache
    "coverage",
    ".coverage",
    "htmlcov",
    ".nx",
    ".direnv",
    "tmp",
    "temp",
    ".tmp",
    ".temp",
})


def _parse_ignore_file(path: Path) -> set[str]:
    """Read a .seekrignore file: one pattern per line, # for comments."""
    out: set[str] = set()
    if not path.exists() or not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def load_ignore_patterns(
    data_dir: Path,
    index_root: Optional[Path] = None,
    extra: Optional[Iterable[str]] = None,
) -> set[str]:
    """
    Build the set of path-component patterns to ignore when indexing.

    Loading order (each source adds to the set; later sources never remove):

    1. Built-in defaults (``DEFAULT_IGNORE_PATTERNS``).
    2. Global file: ``<data_dir>/.seekrignore`` (e.g. ``~/.seekr/.seekrignore``).
    3. Ancestor files: every ``.seekrignore`` found by walking **up** from
       ``index_root`` to the user's home directory, outermost-first.  This
       mirrors how ``.gitignore`` discovery works — a project root file and
       any parent-directory file are both honoured.
    4. Extra patterns from the ``--exclude`` CLI flag.

    Args:
        data_dir:   Seekr data directory (e.g. ``~/.seekr``).
        index_root: Directory being indexed or watched.  When provided, all
                    ``.seekrignore`` files on the path from ``~`` down to
                    ``index_root`` are loaded.  Pass ``None`` for global-only
                    patterns (unusual; prefer passing the root).
        extra:      Additional patterns, e.g. from ``--exclude``.

    Returns:
        Set of pattern strings ready to pass to ``is_ignored()``.
    """
    patterns: set[str] = set(DEFAULT_IGNORE_PATTERNS)

    # 1. Global file
    global_path = data_dir / ".seekrignore"
    from_file = _parse_ignore_file(global_path)
    if from_file:
        patterns |= from_file
        logger.debug("Loaded %d patterns from %s", len(from_file), global_path)

    # 2. Ancestor files (home → index_root, outermost first)
    if index_root is not None and index_root.is_dir():
        home = Path.home()
        # Collect directories from home down to index_root so that a
        # parent-level .seekrignore is applied before the project-level one.
        try:
            rel = index_root.resolve().relative_to(home)
            ancestors: list[Path] = []
            current = home
            for part in rel.parts:
                current = current / part
                ancestors.append(current)
        except ValueError:
            # index_root is outside the home directory — just use the root itself.
            ancestors = [index_root.resolve()]

        for ancestor in ancestors:
            if not ancestor.is_dir():
                continue
            local_path = ancestor / ".seekrignore"
            from_file = _parse_ignore_file(local_path)
            if from_file:
                patterns |= from_file
                logger.debug("Loaded %d patterns from %s", len(from_file), local_path)

    if extra:
        for p in extra:
            p = p.strip()
            if p:
                patterns.add(p)
    return patterns


