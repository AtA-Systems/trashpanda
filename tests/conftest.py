"""Shared test options and deterministic CPU defaults."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import pytest


REQUIRED_FEATURE_MODULES = ("torch", "transformers", "tokenizers", "safetensors")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-feature-deps",
        action="store_true",
        help="fail instead of silently skipping feature tests when optional dependencies are unavailable",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--require-feature-deps"):
        missing = []
        for module in REQUIRED_FEATURE_MODULES:
            try:
                importlib.import_module(module)
            except ImportError:
                missing.append(module)
        if missing:
            pytest.exit("required feature dependencies are unavailable: " + ", ".join(missing), returncode=2)


@pytest.fixture(autouse=True)
def _deterministic_cpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU acceptance runs reproducible and prevent accidental Hub access."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")


def run_command(command: list[str], *, cwd: Path | None = None, timeout: float = 120.0, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a child command with a timeout and captured diagnostics."""
    child_env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    child_env["PYTHONPATH"] = source_root + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env.update(env or {})
    if env and "PYTHONPATH" in env:
        child_env["PYTHONPATH"] = str(env["PYTHONPATH"]) + os.pathsep + source_root
    process = subprocess.Popen(command, cwd=cwd, env=child_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        finally:
            stdout, stderr = process.communicate()
        pytest.fail(f"timed out after {timeout}s: {' '.join(command)}\nstdout={exc.stdout or stdout}\nstderr={exc.stderr or stderr}")
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and result.returncode:
        pytest.fail(f"command failed ({result.returncode}): {' '.join(command)}\nstdout={result.stdout}\nstderr={result.stderr}")
    return result


def run_cli(*args: str | Path, cwd: Path | None = None, timeout: float = 120.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the source CLI with bounded capture and useful failures."""
    return run_command([sys.executable, "-m", "trashpanda", *(str(arg) for arg in args)], cwd=cwd, timeout=timeout, env=env)


@pytest.fixture
def cli_runner() -> Any:
    return run_cli
