#!/usr/bin/env python3
"""Minimal process manager for the AlphaBench-EA Qlib backend."""

from __future__ import annotations

import pathlib
import subprocess
import time
from typing import Optional

import click
import psutil
import requests
from rich.console import Console
from rich.table import Table

console = Console()


def _get_cfg():
    from ffo.config import get_config

    return get_config()


def _ffo_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.resolve()


def _pid_file() -> pathlib.Path:
    path = _get_cfg().pid_dir
    path.mkdir(parents=True, exist_ok=True)
    return path / "backend.pid"


def _log_file() -> pathlib.Path:
    path = _get_cfg().log_dir
    path.mkdir(parents=True, exist_ok=True)
    return path / "backend.log"


def _read_pid() -> Optional[int]:
    try:
        return int(_pid_file().read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _is_running() -> bool:
    pid = _read_pid()
    if pid is None:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        _pid_file().unlink(missing_ok=True)
        return False


def _wait_for_health(port: int, retries: int = 15) -> bool:
    for _ in range(retries):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _start_backend(port: Optional[int], workers: Optional[int], no_wait: bool) -> None:
    if _is_running():
        console.print("[yellow]Backend is already running.[/yellow]")
        return

    cfg = _get_cfg()
    selected_port = port or cfg.get("server.backend.port", 19777)
    selected_workers = workers or cfg.get("server.backend.workers", 1)
    threads = cfg.get("server.backend.threads", 4)
    timeout = cfg.get("server.backend.timeout", 900)
    log_path = _log_file()
    cmd = [
        "gunicorn",
        "backend_app:app",
        "--bind",
        f"0.0.0.0:{selected_port}",
        "--workers",
        str(selected_workers),
        "--threads",
        str(threads),
        "--timeout",
        str(timeout),
        "--chdir",
        str(_ffo_root()),
        "--access-logfile",
        str(log_path),
        "--error-logfile",
        str(log_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _pid_file().write_text(str(process.pid))
    console.print(
        f"[green]Backend starting[/green] (PID {process.pid}, port {selected_port})"
    )
    console.print(f"  Logs → {log_path}")
    if not no_wait:
        status = "[green]OK[/green]" if _wait_for_health(selected_port) else "[yellow]timeout[/yellow]"
        console.print(f"  Health → {status}")


def _stop_backend() -> bool:
    pid = _read_pid()
    if pid is None:
        return False
    try:
        process = psutil.Process(pid)
        process.terminate()
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            process.kill()
    except psutil.NoSuchProcess:
        pass
    _pid_file().unlink(missing_ok=True)
    return True


@click.group()
@click.version_option(package_name="ppo")
def cli() -> None:
    """Manage the AlphaBench-EA Qlib factor backend."""


@cli.group()
def start() -> None:
    """Start a service."""


@start.command("backend")
@click.option("--port", type=int, default=None)
@click.option("--workers", type=int, default=None)
@click.option("--no-wait", is_flag=True)
def start_backend(port: Optional[int], workers: Optional[int], no_wait: bool) -> None:
    """Start the Qlib factor-evaluation backend."""

    _start_backend(port, workers, no_wait)


@cli.group()
def stop() -> None:
    """Stop a service."""


@stop.command("backend")
def stop_backend() -> None:
    """Stop the Qlib factor-evaluation backend."""

    if _stop_backend():
        console.print("[green]Stopped[/green] backend")
    else:
        console.print("[dim]Backend was not running[/dim]")


@cli.command()
@click.option("--port", type=int, default=None)
@click.option("--workers", type=int, default=None)
def restart(port: Optional[int], workers: Optional[int]) -> None:
    """Restart the Qlib factor-evaluation backend."""

    _stop_backend()
    time.sleep(1)
    _start_backend(port, workers, no_wait=False)


@cli.command()
def status() -> None:
    """Show backend process status."""

    table = Table("Service", "Status", "PID", "Log")
    pid = _read_pid()
    table.add_row(
        "backend",
        "[green]running[/green]" if _is_running() else "[dim]stopped[/dim]",
        str(pid or "—"),
        str(_log_file()),
    )
    console.print(table)


@cli.command()
@click.option("--lines", default=100, type=int)
def logs(lines: int) -> None:
    """Print the latest backend log lines."""

    path = _log_file()
    if not path.exists():
        console.print("[dim]No backend log exists yet.[/dim]")
        return
    content = path.read_text(errors="replace").splitlines()
    click.echo("\n".join(content[-max(1, lines) :]))


if __name__ == "__main__":
    cli()
