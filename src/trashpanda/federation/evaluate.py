"""Held-out evaluation that runs inside participant processes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..data import SequenceReader
from ..evaluate import evaluate
from ..model import build_model
from .bundles import load_round_bundle
from .config import FederationConfig
from .coordinator import FederationCoordinator


def evaluate_federation(run_dir: str | Path, *, max_batches: int | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    config = FederationConfig.from_file(root / "config.json")
    coordinator = FederationCoordinator(root, config, resume=True)
    if not coordinator.complete:
        raise ValueError("cannot evaluate an incomplete federation")
    round_path = coordinator.rounds_dir / f"round-{coordinator.current_round_number:06d}"
    coordinator.close()
    results: dict[str, dict[str, Any]] = {}
    for participant in config.participants:
        command = [sys.executable, "-m", "trashpanda.federation.eval_worker", "--round", str(round_path), "--data", participant.data_dir, "--config", str(root / "config.json")]
        if max_batches is not None:
            command.extend(["--max-batches", str(max_batches)])
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env={**__import__("os").environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        results[participant.participant_id] = json.loads(completed.stdout)
    total_tokens = sum(int(item["tokens"]) for item in results.values())
    weighted_loss = sum(float(item["loss"]) * int(item["tokens"]) for item in results.values()) / total_tokens
    return {"round": coordinator.current_round_number, "participants": results, "tokens": total_tokens, "loss": weighted_loss, "perplexity": __import__("math").exp(min(weighted_loss, 50.0))}


__all__ = ["evaluate_federation"]
