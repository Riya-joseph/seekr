"""
Daemon utilities for seekr watch --daemon.

On POSIX (Linux / macOS):
  Implements a double-fork to detach the watcher from the terminal.
  The parent process exits; the child becomes the daemon.

On Windows:
  Uses subprocess.Popen with DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
  to spawn a new seekr process as the daemon worker.  The parent then exits.
  The child is re-launched with the hidden --_worker flag so it skips the
  daemonize step and runs the watcher directly.

PID and watched paths are written to ~/.seekr/watch.pid.
Logs go to ~/.seekr/watch.log.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pid_file(data_dir: Path) -> Path:
    return data_dir / "watch.pid"


def _log_file(data_dir: Path) -> Path:
    return data_dir / "watch.log"


def _process_exists(pid: int) -> bool:
    """Cross-platform check: is the process with this PID alive?"""
    if IS_WINDOWS:
        import ctypes  # noqa: PLC0415
        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # process exists but owned by another user


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def is_running(data_dir: Path) -> tuple[bool, Optional[int]]:
    """
    Return (running, pid).

    Reads the PID file and checks if the process is still alive.
    """
    pid_path = _pid_file(data_dir)
    if not pid_path.exists():
        return False, None
    try:
        info = json.loads(pid_path.read_text())
        pid = int(info["pid"])
    except Exception:
        return False, None

    if _process_exists(pid):
        return True, pid

    # Stale PID file
    pid_path.unlink(missing_ok=True)
    return False, None


def read_watched_paths(data_dir: Path) -> list[str]:
    """Return the list of paths the running daemon is watching, or []."""
    pid_path = _pid_file(data_dir)
    if not pid_path.exists():
        return []
    try:
        info = json.loads(pid_path.read_text())
        return info.get("paths", [])
    except Exception:
        return []


def stop_daemon(data_dir: Path) -> tuple[bool, str]:
    """
    Terminate the running daemon process.

    On POSIX sends SIGTERM; on Windows calls TerminateProcess via os.kill.
    Returns (success, message).
    """
    running, pid = is_running(data_dir)
    if not running or pid is None:
        return False, "No running seekr watcher found."
    try:
        if IS_WINDOWS:
            import subprocess  # noqa: PLC0415
            subprocess.call(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, signal.SIGTERM)
        _pid_file(data_dir).unlink(missing_ok=True)
        return True, f"Stopped watcher (pid={pid})."
    except ProcessLookupError:
        _pid_file(data_dir).unlink(missing_ok=True)
        return False, f"Process {pid} not found (already stopped)."
    except PermissionError as exc:
        return False, f"Permission denied stopping pid {pid}: {exc}"
    except Exception as exc:
        return False, f"Failed to stop pid {pid}: {exc}"


def daemonize(
    data_dir: Path,
    watched_paths: list[str],
    verbose: bool = False,
) -> None:
    """
    Detach the watcher into the background.

    POSIX: double-forks; the parent exits inside this call and the child
    continues executing after the call returns.

    Windows: spawns a fresh detached seekr process with --_worker, writes
    the PID file, then exits the parent inside this call.

    In both cases, after this function returns you are always in the child
    (daemon) process.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if IS_WINDOWS:
        _daemonize_windows(data_dir, watched_paths, verbose=verbose)
    else:
        _daemonize_posix(data_dir, watched_paths)


# ------------------------------------------------------------------
# Platform implementations
# ------------------------------------------------------------------

def _daemonize_posix(data_dir: Path, watched_paths: list[str]) -> None:
    log_path = _log_file(data_dir)
    pid_path = _pid_file(data_dir)

    # ---- First fork ----
    try:
        pid = os.fork()
    except OSError as exc:
        raise RuntimeError(f"First fork failed: {exc}") from exc
    if pid > 0:
        sys.exit(0)  # parent exits

    os.setsid()

    # ---- Second fork — prevent reacquiring a controlling terminal ----
    try:
        pid = os.fork()
    except OSError as exc:
        raise RuntimeError(f"Second fork failed: {exc}") from exc
    if pid > 0:
        sys.exit(0)

    # ---- We are now the daemon ----
    os.chdir("/")
    os.umask(0)

    with open(os.devnull, "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    log_fd = open(log_path, "a", buffering=1)  # line-buffered
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())
    log_fd.close()

    pid_info = {"pid": os.getpid(), "paths": watched_paths}
    pid_path.write_text(json.dumps(pid_info))

    import atexit  # noqa: PLC0415
    atexit.register(lambda: pid_path.unlink(missing_ok=True))


def _daemonize_windows(
    data_dir: Path,
    watched_paths: list[str],
    verbose: bool = False,
) -> None:
    """
    Spawn a detached child seekr process on Windows and write its PID file.

    The parent calls sys.exit(0) after spawning; the child is launched with
    the hidden --_worker flag so it skips this daemonize block entirely.
    """
    import subprocess  # noqa: PLC0415

    log_path = _log_file(data_dir)
    pid_path = _pid_file(data_dir)

    # Reconstruct the seekr watch command for the child process
    cmd = [sys.executable, "-m", "seekr", "watch"]
    cmd += watched_paths
    cmd += ["--data-dir", str(data_dir), "--_worker"]
    if verbose:
        cmd.append("--verbose")

    DETACHED_PROCESS = 0x00000008        # noqa: N806
    CREATE_NEW_PROCESS_GROUP = 0x00000200  # noqa: N806
    CREATE_NO_WINDOW = 0x08000000          # noqa: N806

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            creationflags=(
                DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            ),
            close_fds=True,
        )
    finally:
        log_file.close()

    pid_info = {"pid": proc.pid, "paths": watched_paths}
    pid_path.write_text(json.dumps(pid_info))

    logger.info("Daemon worker spawned (pid=%d)", proc.pid)
    sys.exit(0)  # parent exits; child runs independently
