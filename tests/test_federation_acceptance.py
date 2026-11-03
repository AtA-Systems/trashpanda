from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from trashpanda.federation.bundles import load_round_bundle
from trashpanda.federation.provenance import export_federation
from trashpanda.federation.runner import make_demo_config, run_federation
from trashpanda.provenance import inspect_bundle, verify_bundle

from .conftest import run_cli


pytestmark = pytest.mark.acceptance


def _state_bytes(path: Path) -> dict[str, bytes]:
    bundle = load_round_bundle(path)
    return {name: value.detach().cpu().numpy().tobytes() for name, value in bundle.state.items()}


def test_cli_demo_has_real_processes_and_exact_accounting(tmp_path: Path):
    result = run_cli("federate", "demo", "--output", tmp_path / "demo")
    summary = json.loads(result.stdout)
    assert summary["current_round"] == 3
    assert summary["target_count"] == 1488
    events = [json.loads(line) for line in (tmp_path / "demo" / "run" / "processes.jsonl").read_text().splitlines()]
    assert len({event["pid"] for event in events if event["event"] == "worker_started"}) == 6
    assert {commit["target_count"] for commit in summary["commits"]} == {496}


def test_failed_worker_resume_matches_uninterrupted_and_export_is_private_portable(tmp_path: Path):
    good_config, _ = make_demo_config(tmp_path / "good-fixture")
    good_summary = run_federation(good_config, tmp_path / "good-run")
    failed_config, _ = make_demo_config(tmp_path / "failed-fixture")
    with pytest.raises(RuntimeError, match="incomplete"):
        run_federation(failed_config, tmp_path / "failed-run", fail_participant="alice", fail_after_steps=1)
    resumed = run_federation(failed_config, tmp_path / "failed-run", resume=True)
    assert resumed["target_count"] == good_summary["target_count"] == 1488
    assert _state_bytes(tmp_path / "good-run" / "rounds" / "round-000003") == _state_bytes(tmp_path / "failed-run" / "rounds" / "round-000003")

    export = export_federation(tmp_path / "failed-run", tmp_path / "export")
    relocated = tmp_path / "relocated"
    shutil.copytree(export, relocated)
    shutil.rmtree(tmp_path / "failed-fixture")
    shutil.rmtree(tmp_path / "failed-run")
    assert verify_bundle(relocated).ok
    manifest = json.loads((relocated / "provenance-manifest.json").read_text(encoding="utf-8"))
    records = [json.loads((relocated / "provenance" / "records" / f"{item['id']}.json").read_text(encoding="utf-8")) for item in manifest["records"]]
    assert any(record["type"] == "federation" and "withheld" in record["lineage"] for record in records)
    public_text = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    assert "private-state" not in public_text
    assert "prepared" not in public_text
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(relocated, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(relocated, local_files_only=True, trust_remote_code=False)
    assert model.config.model_type == "llama"
    assert tokenizer.vocab_size == 32


def test_runtime_network_guard_is_inherited_by_workers(tmp_path: Path):
    guard = tmp_path / "network-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket\n"
        "def _blocked(*args, **kwargs): raise RuntimeError('network disabled by acceptance test')\n"
        "socket.socket.connect = _blocked\n"
        "socket.socket.connect_ex = _blocked\n",
        encoding="utf-8",
    )
    config, _ = make_demo_config(tmp_path / "fixture")
    env = {"PYTHONPATH": str(guard)}
    # The real coordinator/worker path completes while every descendant has
    # connect/connect_ex replaced. A separate negative control proves the guard.
    run_cli("federate", "demo", "--output", tmp_path / "guarded", env=env)
    negative = subprocess.run(
        [sys.executable, "-c", "import socket; socket.socket().connect(('127.0.0.1', 9))"],
        env={**os.environ, "PYTHONPATH": str(guard)},
        text=True,
        capture_output=True,
    )
    assert negative.returncode != 0
    assert "network disabled" in negative.stderr
