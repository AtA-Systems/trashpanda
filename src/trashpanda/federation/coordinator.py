"""Crash-safe local federation coordinator and ledger."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import fcntl
from pathlib import Path
import shutil
from typing import Any

import torch

from ..model import build_model
from ..provenance import canonical_bytes
from .aggregate import aggregate_updates
from .bundles import RoundBundle, UpdateBundle, load_round_bundle, load_update_bundle, write_round_bundle
from .config import FederationConfig


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _round_digest(path: Path) -> str:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return hashlib.sha256(canonical_bytes(metadata) + (path / "tensors.safetensors").read_bytes()).hexdigest()


class FederationCoordinator:
    def __init__(self, output_dir: str | Path, config: FederationConfig, *, resume: bool = False) -> None:
        self.root = Path(output_dir)
        self.config = config
        self.rounds_dir = self.root / "rounds"
        self.updates_dir = self.root / "updates"
        self.ledger_path = self.root / "ledger.json"
        if resume:
            if not self.root.is_dir() or not self.ledger_path.is_file():
                raise FileNotFoundError(f"federation run is not resumable: {self.root}")
            saved = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if saved.get("config_digest") != config.digest(include_paths=False):
                raise ValueError("resume federation config does not match the run")
            self._acquire_lock()
            self._recover_published_rounds()
        else:
            if self.root.exists():
                raise FileExistsError(f"federation output already exists: {self.root}")
            self.root.mkdir(parents=True)
            self._acquire_lock()
            self.rounds_dir.mkdir()
            self.updates_dir.mkdir()
            (self.root / "config.json").write_bytes(canonical_bytes(config.to_dict(include_paths=True)) + b"\n")
            model = build_model(config.model_config.model, seed=config.seed)
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            initial = RoundBundle(
                federation_id=self.federation_id,
                round_number=0,
                parent_digest="",
                config_digest=config.digest(include_paths=False),
                model_digest=hashlib.sha256(canonical_bytes(config.model_config.model.__dict__)).hexdigest(),
                tokenizer_identity=config.tokenizer_identity,
                participant_ids=tuple(item.participant_id for item in config.participants),
                local_steps=config.local_steps,
                target_count_per_step=config.batch_size * (config.model_config.model.context_length - 1),
                state=state,
            )
            write_round_bundle(self.rounds_dir / "round-000000", initial)
            self.ledger_path.write_bytes(canonical_bytes({
                "protocol_version": config.protocol_version,
                "federation_id": self.federation_id,
                "config_digest": config.digest(include_paths=False),
                "current_round": 0,
                "accepted": {},
                "commits": [],
                "target_count": 0,
            }) + b"\n")

    def _acquire_lock(self) -> None:
        self._lock_handle = (self.root / "COORDINATOR.lock").open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_handle.close()
            raise RuntimeError("another federation coordinator is already active") from exc

    def _recover_published_rounds(self) -> None:
        ledger = self._ledger()
        while True:
            current = int(ledger["current_round"])
            next_path = self.rounds_dir / f"round-{current + 1:06d}"
            if not next_path.exists():
                break
            current_bundle = load_round_bundle(self.rounds_dir / f"round-{current:06d}")
            entries = [ledger["accepted"].get(f"{current}:{pid}") for pid in current_bundle.participant_ids]
            if any(entry is None for entry in entries):
                raise ValueError("published round has no complete accepted ledger")
            updates = [load_update_bundle(entry["path"]) for entry in entries if entry is not None]
            _state, summary = aggregate_updates(current_bundle, updates)
            next_bundle = load_round_bundle(next_path)
            expected_parent = _round_digest(self.rounds_dir / f"round-{current:06d}")
            if next_bundle.parent_digest != expected_parent:
                raise ValueError("published federation round has broken ancestry")
            ledger["commits"].append({"round": current, "parent_digest": current_bundle.parent_digest, "output_digest": _round_digest(next_path), **summary})
            ledger["current_round"] = current + 1
            ledger["target_count"] = int(ledger.get("target_count", 0)) + int(summary["target_count"])
        self._save_ledger(ledger)

    @property
    def federation_id(self) -> str:
        return hashlib.sha256(canonical_bytes(self.config.to_dict(include_paths=False))).hexdigest()[:32]

    def _ledger(self) -> dict[str, Any]:
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def _save_ledger(self, ledger: dict[str, Any]) -> None:
        temporary = self.ledger_path.with_suffix(".json.tmp")
        temporary.write_bytes(canonical_bytes(ledger) + b"\n")
        temporary.replace(self.ledger_path)

    def close(self) -> None:
        handle = getattr(self, "_lock_handle", None)
        if handle is not None and not handle.closed:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def __enter__(self) -> "FederationCoordinator":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    @property
    def current_round_number(self) -> int:
        return int(self._ledger()["current_round"])

    @property
    def complete(self) -> bool:
        return self.current_round_number >= self.config.rounds

    def current_round(self) -> RoundBundle:
        return load_round_bundle(self.rounds_dir / f"round-{self.current_round_number:06d}")

    def current_round_digest(self) -> str:
        return _round_digest(self.rounds_dir / f"round-{self.current_round_number:06d}")

    def accept_update(self, update_path: str | Path) -> str:
        update = load_update_bundle(update_path)
        round_bundle = self.current_round()
        if update.round_number != round_bundle.round_number or update.federation_id != self.federation_id:
            raise ValueError("update does not belong to the current federation round")
        if update.parent_digest != round_bundle.parent_digest:
            raise ValueError("stale or incompatible update parent")
        if update.participant_id not in round_bundle.participant_ids:
            raise ValueError("unknown federation participant")
        ledger = self._ledger()
        key = f"{update.round_number}:{update.participant_id}"
        existing = ledger["accepted"].get(key)
        if existing:
            if existing["digest"] == update.digest:
                return str(existing["path"])
            raise ValueError("conflicting update for participant and round")
        destination = self.updates_dir / f"round-{update.round_number:06d}" / update.participant_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            recovered = load_update_bundle(destination)
            if recovered.digest != update.digest:
                raise ValueError("conflicting update payload already exists")
        else:
            shutil.copytree(Path(update_path), destination)
        ledger["accepted"][key] = {"participant_id": update.participant_id, "round": update.round_number, "digest": update.digest, "path": str(destination)}
        self._save_ledger(ledger)
        return str(destination)

    def commit_current_round(self) -> dict[str, Any]:
        if self.complete:
            raise ValueError("federation is already complete")
        round_bundle = self.current_round()
        ledger = self._ledger()
        updates: list[UpdateBundle] = []
        for participant_id in round_bundle.participant_ids:
            entry = ledger["accepted"].get(f"{round_bundle.round_number}:{participant_id}")
            if not entry:
                raise ValueError(f"round is missing participant update: {participant_id}")
            updates.append(load_update_bundle(entry["path"]))
        state, summary = aggregate_updates(round_bundle, updates)
        next_round = round_bundle.round_number + 1
        next_bundle = RoundBundle(
            federation_id=self.federation_id,
            round_number=next_round,
            parent_digest=_round_digest(self.rounds_dir / f"round-{round_bundle.round_number:06d}"),
            config_digest=round_bundle.config_digest,
            model_digest=round_bundle.model_digest,
            tokenizer_identity=round_bundle.tokenizer_identity,
            participant_ids=round_bundle.participant_ids,
            local_steps=round_bundle.local_steps,
            target_count_per_step=round_bundle.target_count_per_step,
            state=state,
        )
        write_round_bundle(self.rounds_dir / f"round-{next_round:06d}", next_bundle)
        ledger["commits"].append({"round": round_bundle.round_number, "parent_digest": round_bundle.parent_digest, "output_digest": _round_digest(self.rounds_dir / f"round-{next_round:06d}"), **summary})
        ledger["current_round"] = next_round
        ledger["target_count"] = int(ledger.get("target_count", 0)) + int(summary["target_count"])
        self._save_ledger(ledger)
        return summary

    def summary(self) -> dict[str, Any]:
        return self._ledger()


__all__ = ["FederationCoordinator"]
