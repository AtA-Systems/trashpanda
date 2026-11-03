"""Local multi-process federation runner and synthetic demo fixture."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import replace
from typing import Any

from ..data import SequenceReader
from ..evaluate import evaluate
from ..prepare import prepare_splits
from ..tokenizer import TinyWordLevelTokenizer
from .bundles import RoundBundle, load_round_bundle
from .config import FederationConfig, ParticipantConfig
from .coordinator import FederationCoordinator
from .participant import train_local_round


def _fixture_jsonl(path: Path, *, offset: int) -> None:
    path.write_text("".join(json.dumps({"text": " ".join(f"w{(i + offset + row) % 29}" for i in range(256))}) + "\n" for row in range(8)), encoding="utf-8")


def make_demo_config(root: str | Path) -> tuple[FederationConfig, Path]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"demo fixture output already exists: {root}")
    root.mkdir(parents=True)
    private = root / "private"
    private.mkdir()
    participants: list[ParticipantConfig] = []
    for participant_id, offset in (("alice", 0), ("bob", 7)):
        participant_root = private / participant_id
        participant_root.mkdir()
        train = participant_root / "train.jsonl"
        validation = participant_root / "validation.jsonl"
        _fixture_jsonl(train, offset=offset)
        _fixture_jsonl(validation, offset=offset + 13)
        prepared = participant_root / "prepared"
        prepare_splits(prepared, train_path=train, validation_path=validation, tokenizer=TinyWordLevelTokenizer(), context_length=32)
        participants.append(ParticipantConfig(participant_id=participant_id, data_dir=str(prepared)))
    config = FederationConfig(participants=tuple(participants))
    config_path = root / "federation.json"
    config_path.write_bytes(json.dumps(config.to_dict(), sort_keys=True, indent=2).encode("utf-8") + b"\n")
    return config, config_path


def run_federation(config: FederationConfig, output_dir: str | Path, *, resume: bool = False, fail_participant: str | None = None, fail_after_steps: int | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    coordinator = FederationCoordinator(output, config, resume=resume)
    logs = output / "processes.jsonl"
    try:
        with logs.open("a", encoding="utf-8") as log:
            while not coordinator.complete:
                round_number = coordinator.current_round_number
                round_path = coordinator.rounds_dir / f"round-{round_number:06d}"
                round_exchange = output / "exchange" / f"round-{round_number:06d}"
                round_exchange.mkdir(parents=True, exist_ok=True)
                ledger = coordinator.summary()
                processes: list[tuple[str, subprocess.Popen[str], Path]] = []
                for participant in config.participants:
                    key = f"{round_number}:{participant.participant_id}"
                    if key in ledger.get("accepted", {}):
                        continue
                    private = output / "private-state" / participant.participant_id
                    bundle = round_exchange / participant.participant_id
                    if bundle.exists():
                        coordinator.accept_update(bundle)
                        continue
                    command = [sys.executable, "-m", "trashpanda.federation.worker", "--config", str(output / "config.json"), "--round", str(round_path), "--participant", participant.participant_id, "--private", str(private), "--output", str(bundle)]
                    if fail_participant == participant.participant_id and fail_after_steps is not None:
                        command.extend(["--fail-after-steps", str(fail_after_steps)])
                    environment = dict(os.environ)
                    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONHASHSEED": "0"})
                    source_root = str(Path(__file__).resolve().parents[2])
                    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
                    processes.append((participant.participant_id, process, bundle))
                    log.write(json.dumps({"event": "worker_started", "participant": participant.participant_id, "round": round_number, "pid": process.pid}) + "\n")
                    log.flush()
                failed: list[str] = []
                for participant_id, process, bundle in processes:
                    try:
                        stdout, stderr = process.communicate(timeout=config.timeout_seconds)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                        failed.append(participant_id)
                        continue
                    log.write(json.dumps({"event": "worker_finished", "participant": participant_id, "round": round_number, "pid": process.pid, "returncode": process.returncode, "stderr": stderr[-1000:]}) + "\n")
                    log.flush()
                    if process.returncode != 0 or not bundle.exists():
                        failed.append(participant_id)
                        continue
                    coordinator.accept_update(bundle)
                if failed:
                    raise RuntimeError(f"federation round {round_number} incomplete; failed participants: {', '.join(failed)}")
                coordinator.commit_current_round()
        summary = coordinator.summary()
    finally:
        coordinator.close()
    return summary


def compare_baselines(run_dir: str | Path) -> dict[str, Any]:
    """Train small local-only and pooled synthetic baselines from round zero."""
    root = Path(run_dir)
    config = FederationConfig.from_file(root / "config.json")
    coordinator = FederationCoordinator(root, config, resume=True)
    initial = load_round_bundle(coordinator.rounds_dir / "round-000000")
    baseline_root = root / "baselines"
    baseline_root.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    local_config = replace(config, rounds=1, local_steps=config.rounds * config.local_steps)
    local_round = replace(initial, round_number=0, local_steps=local_config.local_steps, target_count_per_step=config.batch_size * (config.model_config.model.context_length - 1))
    for participant in config.participants:
        private = baseline_root / f"local-{participant.participant_id}-private"
        update_path = baseline_root / f"local-{participant.participant_id}-update"
        train_local_round(local_round, local_config, participant, private, update_path)
        update = __import__("trashpanda.federation.bundles", fromlist=["load_update_bundle"]).load_update_bundle(update_path)
        validation = SequenceReader(Path(participant.data_dir), split="validation")
        model = __import__("trashpanda.model", fromlist=["build_model"]).build_model(config.model_config.model, seed=config.seed)
        model.load_state_dict(update.state)
        result = evaluate(model, validation, max_batches=2)
        rows.append({"name": f"local-only-{participant.participant_id}", "targets": update.target_count, "validation_tokens": result.tokens, "loss": result.loss, "perplexity": result.perplexity})
    source_paths = [Path(item.data_dir).parent / "train.jsonl" for item in config.participants]
    validation_paths = [Path(item.data_dir).parent / "validation.jsonl" for item in config.participants]
    pooled_dir = baseline_root / "pooled-prepared"
    if not pooled_dir.exists():
        prepare_splits(pooled_dir, train_path=source_paths, validation_path=validation_paths, tokenizer=TinyWordLevelTokenizer(), context_length=32)
    pooled_participant = ParticipantConfig("pooled", str(pooled_dir))
    pooled_steps = config.rounds * config.local_steps * len(config.participants)
    pooled_round = replace(initial, participant_ids=("pooled",), local_steps=pooled_steps, target_count_per_step=config.batch_size * 31)
    pooled_update_path = baseline_root / "pooled-update"
    pooled_config = replace(config, rounds=1, local_steps=pooled_steps)
    train_local_round(pooled_round, pooled_config, pooled_participant, baseline_root / "pooled-private", pooled_update_path)
    pooled_update = __import__("trashpanda.federation.bundles", fromlist=["load_update_bundle"]).load_update_bundle(pooled_update_path)
    pooled_model = __import__("trashpanda.model", fromlist=["build_model"]).build_model(config.model_config.model, seed=config.seed)
    pooled_model.load_state_dict(pooled_update.state)
    pooled_validation = SequenceReader(pooled_dir, split="validation")
    pooled_result = evaluate(pooled_model, pooled_validation, max_batches=4)
    rows.append({"name": "pooled-synthetic", "targets": pooled_update.target_count, "validation_tokens": pooled_result.tokens, "loss": pooled_result.loss, "perplexity": pooled_result.perplexity})
    report = {"initial_round": 0, "algorithm": "local AdamW from identical initial tensors", "rows": rows}
    (root / "baseline-comparison.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    coordinator.close()
    return report


__all__ = ["compare_baselines", "make_demo_config", "run_federation"]
