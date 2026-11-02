"""Atomic, resumable checkpoints containing only tensor/primitive state."""
from __future__ import annotations

from dataclasses import dataclass
import os
import random
from pathlib import Path
import re
import hashlib
from typing import Any, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


CHECKPOINT_VERSION = 1


@dataclass
class TrainingState:
    update_step: int = 0
    consumed_tokens: int = 0
    micro_steps: int = 0

    @property
    def step(self) -> int:
        return self.update_step

    @property
    def tokens_seen(self) -> int:
        return self.consumed_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "update_step": int(self.update_step),
            "consumed_tokens": int(self.consumed_tokens),
            "micro_steps": int(self.micro_steps),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingState":
        return cls(
            update_step=int(value.get("update_step", 0)),
            consumed_tokens=int(value.get("consumed_tokens", 0)),
            micro_steps=int(value.get("micro_steps", 0)),
        )


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("checkpoints require PyTorch")


def _safe_rng_value(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, tuple):
        return [_safe_rng_value(item) for item in value]
    if isinstance(value, list):
        return [_safe_rng_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_rng_value(item) for key, item in value.items()}
    return value


def _restore_rng_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_restore_rng_value(item) for item in value)
    if isinstance(value, dict) and set(value) == {"__ndarray__", "dtype", "shape"} and np is not None:
        return np.asarray(value["__ndarray__"], dtype=value["dtype"]).reshape(value["shape"])
    if isinstance(value, dict):
        return {key: _restore_rng_value(item) for key, item in value.items()}
    return value


def capture_rng_state() -> dict[str, Any]:
    _require_torch()
    return {
        "python": _safe_rng_value(random.getstate()),
        "numpy": _safe_rng_value(np.random.get_state()) if np is not None else None,
        "torch": torch.get_rng_state().cpu(),
        "cuda": [item.cpu() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    _require_torch()
    random.setstate(_restore_rng_value(value["python"]))
    if np is not None and value.get("numpy") is not None:
        np.random.set_state(_restore_rng_value(value["numpy"]))
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value.get("cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def _cpu_state(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def checkpoint_path(directory: str | Path, update_step: int) -> Path:
    return Path(directory) / f"checkpoint-{int(update_step):08d}.pt"


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    state: TrainingState,
    config: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    tokenizer_snapshot: Mapping[str, Any],
    tokenizer_files: Mapping[str, bytes] | None = None,
    data_identity: str | None = None,
    scheduler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a complete checkpoint by fsyncing a temporary file then replacing."""
    _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "trashpanda.checkpoint.v1",
        "version": CHECKPOINT_VERSION,
        "model": _cpu_state(model.state_dict()),
        "optimizer": _cpu_state(optimizer.state_dict()),
        "state": state.to_dict(),
        "config": dict(config),
        "sampler": _cpu_state(dict(sampler_state)),
        "tokenizer": dict(tokenizer_snapshot),
        "tokenizer_files": dict(tokenizer_files or {}),
        "data_identity": data_identity,
        "scheduler": _cpu_state(scheduler.state_dict()) if scheduler is not None else None,
        "rng": capture_rng_state(),
    }
    if extra:
        payload["extra"] = _cpu_state(dict(extra))
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle, _use_new_zipfile_serialization=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # Ensure the directory entry is durable where supported.
        try:
            directory_handle = os.open(destination.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_handle)
            finally:
                os.close(directory_handle)
        except (AttributeError, OSError):
            pass
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def save_training_state(
    path: str | Path,
    *,
    optimizer: Any,
    state: TrainingState,
    config: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    tokenizer_snapshot: Mapping[str, Any],
    model_file: str,
    model_sha256: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save resumable optimizer/RNG state referencing external model weights.

    This bundle format lets a native HF ``model.safetensors`` file serve as
    the single copy of model weights while retaining optimizer, sampler, and
    RNG state for resume.
    """
    _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "trashpanda.checkpoint.bundle.v1",
        "version": CHECKPOINT_VERSION,
        "model_file": model_file,
        "optimizer": _cpu_state(optimizer.state_dict()),
        "state": state.to_dict(),
        "config": dict(config),
        "sampler": _cpu_state(dict(sampler_state)),
        "tokenizer": dict(tokenizer_snapshot),
        "rng": capture_rng_state(),
    }
    if model_sha256:
        payload["model_sha256"] = model_sha256
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle, _use_new_zipfile_serialization=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(path: str | Path, *, model: Any, optimizer: Any | None = None, map_location: str = "cpu") -> dict[str, Any]:
    _require_torch()
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") not in {"trashpanda.checkpoint.v1", "trashpanda.checkpoint.bundle.v1"}:
        raise ValueError("not a trashpanda checkpoint")
    if payload.get("format") == "trashpanda.checkpoint.bundle.v1":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("bundle checkpoints require safetensors") from exc
        model_name = str(payload["model_file"])
        model_path = (checkpoint.parent / model_name).resolve()
        if model_path.parent != checkpoint.parent.resolve() or not model_path.is_file():
            raise ValueError("external model_file must be a regular sibling file")
        expected_hash = payload.get("model_sha256")
        if expected_hash:
            digest = hashlib.sha256()
            with model_path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError("external model hash mismatch; refusing to load model bytes")
        model.load_state_dict(load_file(str(model_path), device="cpu"))
    else:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def list_checkpoints(directory: str | Path) -> list[Path]:
    paths = list(Path(directory).glob("checkpoint-*.pt"))
    return sorted(paths, key=lambda path: int(re.search(r"checkpoint-(\d+)", path.name).group(1)))


__all__ = [
    "CHECKPOINT_VERSION",
    "TrainingState",
    "checkpoint_path",
    "list_checkpoints",
    "load_checkpoint",
    "save_checkpoint",
    "save_training_state",
]
