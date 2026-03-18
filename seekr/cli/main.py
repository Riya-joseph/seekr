"""
Seekr CLI — the outermost layer.

Commands:
  seekr index <path>      — Index a file or directory
  seekr search "<query>"  — Semantic search
  seekr watch <path>      — Continuous incremental indexing
  seekr watch-stop        — Stop the background watcher daemon
  seekr status            — Show index statistics
  seekr prune <path>      — Remove a subtree from the index
  seekr reset             — Wipe the entire index

Architecture: This layer imports ONLY from seekr.application and
seekr.infrastructure.container.  No FAISS, sentence-transformers, or
other infrastructure types leak into this file.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.request import pathname2url

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from seekr.utils.logging import configure_logging

# ------------------------------------------------------------------
# Shared console and Typer app
# ------------------------------------------------------------------

console = Console()

app = typer.Typer(
    name="seekr",
    help="[bold cyan]Seekr[/] — offline semantic file search engine.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

logger = logging.getLogger("seekr.cli")

# ------------------------------------------------------------------
# Shared option descriptions
# ------------------------------------------------------------------

_VERBOSE_HELP = "Enable debug logging."
_DATA_DIR_HELP = "Override the Seekr data directory (default: ~/.seekr)."
_EXCLUDE_HELP = (
    "Comma-separated path components to exclude (e.g. node_modules,.git). "
    "Merged with defaults and .seekrignore (see README)."
)


def _setup_logging(verbose: bool) -> None:
    """Configure seekr logging.  Idempotent — safe to call once per command."""
    configure_logging(verbose=verbose)


# ------------------------------------------------------------------
# index command
# ------------------------------------------------------------------

_PATH_INDEX_HELP = (
    "Directory (or file) to index. Defaults to the [bold]current directory[/]. "
    "Avoid indexing ~ or / directly — use .seekrignore or --exclude to skip dirs."
)
_DRY_RUN_HELP = (
    "Only walk the tree and report how many files would be indexed "
    "(and estimated chunks). No indexing."
)


@app.command(help="[bold]Index[/] a file or directory for semantic search.")
def index(
    path: Path | None = typer.Option(None, "--path", "-p", help=_PATH_INDEX_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    data_dir: Path | None = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
    exclude: str | None = typer.Option(None, "--exclude", "-e", help=_EXCLUDE_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """
    [bold]Index[/] a file or directory for semantic search.

    Re-indexes only files that have changed since the last run.
    Defaults to the current directory when [bold]--path[/] is omitted.
    [dim]Use --dry-run to preview what would be indexed.[/]

    Examples:
      seekr index
      seekr index --path ~/Projects/myapp
      seekr index --path ~/Downloads --dry-run
    """
    _setup_logging(verbose)
    from seekr.config.settings import NUM_INDEX_WORKERS
    from seekr.infrastructure.container import Container
    from seekr.infrastructure.ignore import load_ignore_patterns

    # Default to current working directory when no path is given
    resolved_path: Path = (path or Path.cwd()).expanduser().resolve()

    data_dir_path = data_dir or Path.home() / ".seekr"
    container = Container(data_dir=data_dir_path)

    index_root = resolved_path if resolved_path.is_dir() else resolved_path.parent
    extra = [p.strip() for p in exclude.split(",")] if exclude else None
    ignore_patterns = load_ignore_patterns(
        data_dir_path,
        index_root=index_root,
        extra=extra,
    )

    if dry_run:
        svc = container.dry_run_service()
        try:
            result = svc.dry_run(resolved_path, ignore_patterns=ignore_patterns)
        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(code=1) from exc

        to_index: list[str] = result.get("to_index", [])  # type: ignore[assignment]
        already_indexed: list[str] = result.get("already_indexed", [])  # type: ignore[assignment]
        max_show = 100

        def _format_paths(paths: list[str]) -> str:
            lines = "\n".join(f"  • {_shorten_path(p)}" for p in paths[:max_show])
            if len(paths) > max_show:
                lines += f"\n  [dim]… and {len(paths) - max_show} more[/]"
            return lines

        summary = f"[bold]Path scanned:[/] [dim]{resolved_path}[/]\n\n"

        if to_index:
            summary += (
                f"[bold green]New / changed — would be indexed:[/] {len(to_index)}\n"
                f"[bold]Estimated new chunks:[/] {result['estimated_chunks']}\n\n"
                f"{_format_paths(to_index)}\n\n"
            )
        else:
            summary += "[bold green]No new or changed files — index is up-to-date.[/]\n\n"

        if already_indexed:
            summary += (
                f"[bold dim]Already indexed (unchanged) — would be skipped:[/] "
                f"{len(already_indexed)}\n"
            )

        summary += (
            "\n[dim]Run without --dry-run to index. "
            "Adjust --exclude or .seekrignore if the count is too high.[/]"
        )
        console.print(Panel(summary, title="[bold]Dry run[/] (no indexing)", border_style="cyan"))
        return

    svc = container.index_service(background=True, progress_callback=None)
    try:
        counts = svc.index_path(resolved_path, ignore_patterns=ignore_patterns)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    queued = counts.get("queued", 0)
    skipped = counts.get("skipped", 0)

    if queued == 0:
        console.print(
            Panel(
                f"[bold]Scanning files…[/]\n"
                f"Discovered [bold]{queued + skipped}[/] files\n"
                f"[dim]Nothing to index (all up to date or skipped).[/]",
                title="[bold]Index[/]",
                border_style="cyan",
            )
        )
        return

    console.print(
        Panel(
            f"[bold]Scanning files…[/]\n"
            f"Discovered [bold]{queued + skipped}[/] files\n"
            f"Queued [bold cyan]{queued}[/] indexing tasks",
            title="[bold]Index[/]",
            border_style="cyan",
        )
    )

    def _run_workers() -> None:
        container.start_index_workers(
            num_workers=NUM_INDEX_WORKERS,
            daemon=False,
            ignore_patterns=ignore_patterns,
        )

    worker_thread = threading.Thread(target=_run_workers)
    worker_thread.start()

    queue = container.index_queue
    initial_stats = queue.get_stats()
    initial_done = initial_stats.completed + initial_stats.failed

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Indexing…", total=queued, completed=0)
        while worker_thread.is_alive():
            stats = queue.get_stats()
            done_this_run = (stats.completed + stats.failed) - initial_done
            progress.update(task_id, completed=min(done_this_run, queued))
            time.sleep(0.25)
        stats = queue.get_stats()
        done_this_run = (stats.completed + stats.failed) - initial_done
        progress.update(task_id, completed=min(done_this_run, queued))

    worker_thread.join()
    console.print(
        Panel("[green]Indexing complete.[/]", title="[bold]Index[/]", border_style="green")
    )


# ------------------------------------------------------------------
# search command
# ------------------------------------------------------------------

_PATH_SEARCH_HELP = (
    "Only show results under this path (file or directory). "
    "Uses the same index; filters results by path."
)

_TYPE_ICONS: dict[str, str] = {
    "TEXT": "📄",
    "CODE": "💻",
    "IMAGE": "🖼️",
    "DOCUMENT": "📑",
    "UNKNOWN": "❓",
}


@app.command(help="[bold]Search[/] the index with a natural-language query.")
def search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    top_k: int = typer.Option(10, "--top", "-n", help="Number of results to return."),
    file_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by type: text, image, code, document.",
    ),
    path: Path | None = typer.Option(None, "--path", "-p", help=_PATH_SEARCH_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    data_dir: Path | None = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
) -> None:
    """
    [bold]Search[/] the index with a natural-language query.

    Examples:
      seekr search "authentication middleware"
      seekr search "sunset over mountains" --type image
      seekr search "database connection pool" --top 5
      seekr search "config" --path ~/Downloads/docs
    """
    _setup_logging(verbose)
    from seekr.domain.entities import FileType
    from seekr.infrastructure.container import Container

    container = Container(data_dir=data_dir or Path.home() / ".seekr")

    type_filter = None
    if file_type:
        try:
            type_filter = FileType[file_type.upper()]
        except KeyError as exc:
            console.print(
                f"[red]Unknown file type:[/] {file_type}. "
                "Choose from: text, image, code, document."
            )
            raise typer.Exit(code=1) from exc

    path_prefix = path.expanduser().resolve() if path else None
    svc = container.search_service()

    with console.status("[cyan]Searching…[/]", spinner="dots"):
        try:
            results = svc.search(
                query,
                top_k=top_k,
                file_type_filter=type_filter,
                path_prefix=path_prefix,
            )
        except Exception as exc:
            console.print(f"[bold red]Search failed:[/] {exc}")
            raise typer.Exit(code=1) from exc

    if not results:
        console.print(
            Panel(
                "[dim]No results found.[/]\n\nTry broadening your query or running "
                "[bold]seekr index[/] first.",
                title="[yellow]No Results[/]",
                border_style="yellow",
            )
        )
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Score", width=7, justify="right")
    table.add_column("Type", width=8)
    table.add_column("File", style="bold")
    table.add_column("Snippet", ratio=1)

    for i, result in enumerate(results, 1):
        score_pct = f"{result.score * 100:.1f}%"
        score_style = "green" if result.score > 0.7 else "yellow" if result.score > 0.45 else "red"
        icon = _TYPE_ICONS.get(result.file_type.name, "❓")
        abs_path = str(Path(result.file_path).expanduser().resolve())
        file_url = "file://" + pathname2url(abs_path)
        short_path = _shorten_path(result.file_path)
        file_cell = Text()
        file_cell.append(short_path, style=f"bold link {file_url}")
        file_cell.append(f"\n{abs_path}", style=f"dim link {file_url}")
        snippet = result.snippet[:200].replace("\n", " ")

        table.add_row(
            str(i),
            Text(score_pct, style=score_style),
            f"{icon} {result.file_type.name.capitalize()}",
            file_cell,
            Text(snippet, style="dim"),
        )

    console.print(
        Panel(
            table,
            title=f"[bold]Results for:[/] [italic cyan]{query}[/]",
            subtitle="[dim]Ctrl+Click a file path to open[/]",
            border_style="cyan",
        )
    )


# ------------------------------------------------------------------
# watch command
# ------------------------------------------------------------------


@app.command(help="[bold]Watch[/] directories and auto-index new/modified files.")
def watch(
    paths: list[Path] = typer.Argument(..., help="Directories to watch."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    data_dir: Path | None = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
    daemon: bool = typer.Option(
        False,
        "--daemon",
        "-d",
        help="Run in the background. Logs go to ~/.seekr/watch.log. " "Stop with seekr watch-stop.",
    ),
    _worker: bool = typer.Option(
        False,
        "--_worker",
        hidden=True,
        help="Internal: skip daemonize and run directly as the daemon worker "
        "(used by --daemon on Windows).",
    ),
) -> None:
    """
    [bold]Watch[/] directories and auto-index new/modified files.

    By default blocks until Ctrl+C. Use [bold]--daemon[/] to run in the background.

    Example:
      seekr watch ~/documents ~/projects
      seekr watch ~/projects --daemon      # background mode
      seekr watch-stop                     # stop background watcher
    """
    _setup_logging(verbose)
    from seekr.config.settings import NUM_INDEX_WORKERS
    from seekr.infrastructure.container import Container
    from seekr.infrastructure.ignore import load_ignore_patterns

    data_dir_path = data_dir or Path.home() / ".seekr"

    # ------------------------------------------------------------------
    # Daemon setup — two paths depending on OS and role
    # ------------------------------------------------------------------
    # is_background: True when this process should behave as a silent daemon
    # (no Rich output, log to file).  On POSIX this is the forked child;
    # on Windows it's the re-spawned child launched with --_worker.
    is_background = False

    if daemon and not _worker:
        # ---- Parent path (both POSIX and Windows) ----
        from seekr.infrastructure.daemon import daemonize, is_running

        running, existing_pid = is_running(data_dir_path)
        if running:
            console.print(
                f"[yellow]A watcher daemon is already running (pid={existing_pid}). "
                f"Run [bold]seekr watch-stop[/] first.[/]"
            )
            raise typer.Exit(1)

        resolved = [str(p.resolve()) for p in paths]
        console.print(
            Panel(
                "\n".join(f"  📁 [cyan]{p}[/]" for p in resolved)
                + f"\n\n[dim]Logs:[/] {data_dir_path / 'watch.log'}\n"
                + "[dim]Stop with:[/] [bold]seekr watch-stop[/]",
                title="[bold]Starting daemon watcher[/]",
                border_style="cyan",
            )
        )
        # POSIX: parent exits inside daemonize(); child continues here.
        # Windows: parent spawns child with --_worker then exits inside
        #          daemonize(); this line is never reached by the parent.
        daemonize(data_dir_path, resolved, verbose=verbose)

        # --- Only the POSIX child reaches this point ---
        is_background = True
        configure_logging(verbose=verbose, log_file=data_dir_path / "watch.log")

    elif _worker:
        # ---- Windows daemon child path ----
        # Re-spawned by _daemonize_windows with --_worker; run as background
        # worker without any Rich terminal output.
        is_background = True
        configure_logging(verbose=verbose, log_file=data_dir_path / "watch.log")

    # ------------------------------------------------------------------
    # Set up container, workers, and watcher (runs for all paths)
    # ------------------------------------------------------------------
    import threading as _threading

    container = Container(data_dir=data_dir_path)
    # Load global patterns first, then merge in the local .seekrignore from
    # each watched directory so per-project exclusions are honoured in watch mode.
    ignore_patterns = load_ignore_patterns(data_dir_path, index_root=None)
    for watch_path in paths:
        rp = watch_path.expanduser().resolve()
        local_root = rp if rp.is_dir() else rp.parent
        ignore_patterns |= load_ignore_patterns(data_dir_path, index_root=local_root)

    worker_stop = _threading.Event()
    container.start_index_workers(
        num_workers=NUM_INDEX_WORKERS,
        daemon=True,
        stop_event=worker_stop,
        ignore_patterns=ignore_patterns,
    )

    def _on_event(event_type: str, event_path: Path) -> None:
        icons = {"created": "➕", "modified": "✏️", "deleted": "🗑️"}  # noqa: RUF001
        icon = icons.get(event_type, "•")
        ts = datetime.now().strftime("%H:%M:%S")
        if is_background:
            logger.info("%s %s: %s", icon, event_type, event_path)
        else:
            label = (
                f"[dim]{ts}[/] {icon} [bold]{event_type}[/]: " f"{_shorten_path(str(event_path))}"
            )
            console.print(label)

    svc = container.watcher_service(
        on_event=_on_event,
        use_queue=True,
        ignore_patterns=ignore_patterns,
    )

    if not is_background:
        console.print(
            Panel(
                "\n".join(f"  📁 [cyan]{p.resolve()}[/]" for p in paths)
                + "\n\n[dim]Press Ctrl+C to stop.[/]",
                title="[bold]Seekr Watch Mode[/]",
                border_style="cyan",
            )
        )

    try:
        svc.start(paths, blocking=True)
    except Exception as exc:
        if not is_background:
            console.print(f"[bold red]Watcher error:[/] {exc}")
        logger.error("Watcher error: %s", exc, exc_info=True)
        raise typer.Exit(code=1) from exc
    finally:
        worker_stop.set()

    if not is_background:
        console.print("[dim]Watcher stopped.[/]")


# ------------------------------------------------------------------
# watch-stop command
# ------------------------------------------------------------------


@app.command(name="watch-stop", help="Stop the background [bold]seekr watch --daemon[/] process.")
def watch_stop(
    data_dir: Path | None = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
) -> None:
    """
    Stop the background [bold]seekr watch --daemon[/] process.
    """
    from seekr.infrastructure.daemon import read_watched_paths, stop_daemon

    data_dir_path = data_dir or Path.home() / ".seekr"
    paths = read_watched_paths(data_dir_path)
    ok, msg = stop_daemon(data_dir_path)
    if ok:
        details = "\n".join(f"  📁 [dim]{p}[/]" for p in paths) + "\n\n" if paths else ""
        console.print(
            Panel(
                details + f"[green]{msg}[/]",
                title="[bold]Watcher stopped[/]",
                border_style="cyan",
            )
        )
    else:
        console.print(f"[yellow]{msg}[/]")


# ------------------------------------------------------------------
# status command
# ------------------------------------------------------------------


@app.command(help="Show index [bold]status[/] and statistics.")
def status(
    data_dir: Path | None = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
) -> None:
    """
    Show index [bold]status[/] and statistics (including background queue progress).
    """
    _setup_logging(verbose)
    from seekr.infrastructure.container import Container

    container = Container(data_dir=data_dir or Path.home() / ".seekr")
    meta = container.metadata_store

    try:
        stats = meta.stats()
    except Exception as exc:
        console.print(f"[bold red]Could not read index stats:[/] {exc}")
        raise typer.Exit(code=1) from exc

    text_vec = container.text_vector_store.total_vectors
    clip_vec = container.clip_vector_store.total_vectors

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", justify="right")
    grid.add_column(style="bold")

    def _row(label: str, value: str) -> None:
        grid.add_row(label, value)

    _row("Data directory", str(container.data_dir))
    _row("Total files", str(stats.total_files))
    _row("Text files", str(stats.text_files))
    _row("Image files", str(stats.image_files))
    _row("Total chunks", str(stats.total_chunks))
    _row("Text vectors", str(text_vec))
    _row("CLIP vectors", str(clip_vec))
    _row("Index size", _human_bytes(stats.index_size_bytes))
    _row(
        "Last updated",
        stats.last_updated.strftime("%Y-%m-%d %H:%M UTC") if stats.last_updated else "never",
    )

    console.print(Panel(grid, title="[bold]Seekr Index Status[/]", border_style="cyan"))

    try:
        queue = container.index_queue
        qstats = queue.get_stats()
        if qstats.total > 0:
            queue_grid = Table.grid(padding=(0, 3))
            queue_grid.add_column(style="dim", justify="right")
            queue_grid.add_column(style="bold")
            queue_grid.add_row("Total tasks", str(qstats.total))
            queue_grid.add_row("Completed", str(qstats.completed))
            queue_grid.add_row("Processing", str(qstats.processing))
            queue_grid.add_row("Pending", str(qstats.pending))
            queue_grid.add_row("Failed", str(qstats.failed))
            queue_grid.add_row("Progress", f"{qstats.progress_pct:.0f}%")
            console.print(Panel(queue_grid, title="[bold]Index Queue[/]", border_style="cyan"))
    except Exception as exc:
        logger.debug("Could not read queue stats: %s", exc)


# ------------------------------------------------------------------
# prune command
# ------------------------------------------------------------------


@app.command(help="[bold]Prune[/] a path from the index.")
def prune(
    path: Path | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Directory (or file) to remove from the index. "
        "Defaults to the current directory. "
        "All indexed files under this path are pruned.",
    ),
    data_dir: Path | None = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
) -> None:
    """
    [bold]Prune[/] a path from the index — remove all indexed files under the given directory.

    Use this to un-index a subtree (e.g. a project you no longer want in search)
    without re-indexing everything else. Does not delete files on disk.

    Examples:
      seekr prune --path ~/Projects/old-repo
      seekr prune                               # prunes current directory
    """
    _setup_logging(verbose)
    from seekr.infrastructure.container import Container

    data_dir_path = data_dir or Path.home() / ".seekr"
    container = Container(data_dir=data_dir_path)
    svc = container.dry_run_service()  # prune only needs parsers + metadata, not models

    resolved_prune_path = (path or Path.cwd()).expanduser().resolve()
    try:
        result = svc.prune_path(resolved_prune_path)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[bold]Files removed from index:[/] {result['removed']}\n\n"
            f"[dim]Path pruned:[/] {resolved_prune_path}",
            title="[bold]Prune complete[/]",
            border_style="cyan",
        )
    )


# ------------------------------------------------------------------
# reset command
# ------------------------------------------------------------------


@app.command(help="[bold]Reset[/] the index — wipe all indexed data and start over.")
def reset(
    data_dir: Path | None = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
) -> None:
    """
    [bold]Reset[/] the index — wipe all indexed data and start over.

    Deletes the vector indices and metadata DB. Cached models in ~/.seekr/models/
    are kept. Run [bold]seekr index[/] after this to rebuild.
    """
    _setup_logging(verbose)
    import shutil

    data_dir_path = data_dir or Path.home() / ".seekr"
    meta_file = data_dir_path / "metadata.db"
    text_index_dir = data_dir_path / "text_index"
    clip_index_dir = data_dir_path / "clip_index"

    if not force:
        console.print(
            "[yellow]This will delete all index data (vectors + metadata). "
            "Cached models will be kept.[/]"
        )
        try:
            confirm = typer.confirm("Continue?", default=False)
        except Exception:
            confirm = False
        if not confirm:
            console.print("[dim]Reset cancelled.[/]")
            raise typer.Exit(0)

    removed = []
    if meta_file.exists():
        meta_file.unlink()
        removed.append(str(meta_file))
    if text_index_dir.exists():
        shutil.rmtree(text_index_dir)
        removed.append(str(text_index_dir))
    if clip_index_dir.exists():
        shutil.rmtree(clip_index_dir)
        removed.append(str(clip_index_dir))

    if removed:
        console.print(
            Panel(
                "[green]Index data removed.[/]\n\n"
                + "\n".join(f"  [dim]•[/] {p}" for p in removed)
                + "\n\n[dim]Run [bold]seekr index <path>[/] to rebuild.[/]",
                title="[bold]Reset complete[/]",
                border_style="cyan",
            )
        )
    else:
        console.print("[dim]No index data found; nothing to reset.[/]")


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------


def _shorten_path(path: str, max_len: int = 60) -> str:
    """Abbreviate a long absolute path using ~ for the home directory."""
    home = str(Path.home())
    if path.startswith(home):
        path = "~" + path[len(home) :]
    if len(path) > max_len:
        path = "…" + path[-(max_len - 1) :]
    return path


def _human_bytes(n: float) -> str:
    """Convert a byte count to a human-readable string (B / KB / MB / GB / TB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
