"""Hash-bound, relocatable round and update bundles.

Only model tensors and bounded protocol metadata cross the participant boundary.
Optimizer, sampler, RNG and dataset state are deliberately absent from these
bundles and remain in participant-private recovery files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ..provenance import canonical_bytes, safe_relative_path, sha256_file
from .config import PROTOCOL_VERSION


MAX_BUNDLE_BYTES = 128 * 1024 * 1024


def _state_for_disk(state: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    pointers: dict[int, str] = {}
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        pointer = tensor.data_ptr()
        if pointer in pointers:
            aliases[name] = pointers[pointer]
        else:
            pointers[pointer] = name
            tensors[name] = tensor
    return tensors, aliases


def _state_from_disk(tensors: Mapping[str, torch.Tensor], aliases: Mapping[str, str]) -> dict[str, torch.Tensor]:
    result = {str(key): value for key, value in tensors.items()}
    for alias, target in aliases.items():
        if target not in result:
            raise ValueError(f"tensor alias target is missing: {target}")
        result[str(alias)] = result[target]
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def _publish(directory: Path, writer: Any) -> Path:
    if directory.exists():
        raise FileExistsError(f"bundle destination already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        writer(temporary)
        _write_json(temporary / "COMPLETE.json", {"format": PROTOCOL_VERSION, "files": sorted(path.name for path in temporary.iterdir())})
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory


def _validate_tree(directory: Path) -> None:
    if not directory.is_dir() or not (directory / "COMPLETE.json").is_file():
        raise ValueError(f"incomplete federation bundle: {directory}")
    total = 0
    try:
        completion = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid federation completion marker") from exc
    if completion.get("format") != PROTOCOL_VERSION:
        raise ValueError("unsupported federation completion marker")
    declared = set(completion.get("files", []))
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if declared | {"COMPLETE.json"} != actual_names:
        raise ValueError("federation completion marker does not match payload")
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        safe_relative_path(relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe federation bundle payload: {relative}")
        total += path.stat().st_size
    if total > MAX_BUNDLE_BYTES:
        raise ValueError("federation bundle exceeds size limit")


def _metadata(directory: Path) -> dict[str, Any]:
    _validate_tree(directory)
    try:
        value = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid federation bundle metadata: {directory}") from exc
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported federation bundle protocol")
    return value


@dataclass(frozen=True)
class RoundBundle:
    federation_id: str
    round_number: int
    parent_digest: str
    config_digest: str
    model_digest: str
    tokenizer_identity: str
    participant_ids: tuple[str, ...]
    local_steps: int
    target_count_per_step: int
    state: dict[str, torch.Tensor]


@dataclass(frozen=True)
class UpdateBundle:
    federation_id: str
    round_number: int
    participant_id: str
    parent_digest: str
    config_digest: str
    model_digest: str
    tokenizer_identity: str
    completed_steps: int
    target_count: int
    state: dict[str, torch.Tensor]
    digest: str = ""


def _write_state(path: Path, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    tensors, aliases = _state_for_disk(state)
    if not tensors:
        raise ValueError("bundle state cannot be empty")
    for name, tensor in tensors.items():
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"non-finite tensor: {name}")
        if tensor.dtype not in {torch.float32, torch.float64, torch.int64, torch.int32, torch.bool}:
            raise ValueError(f"unsupported tensor dtype: {name}: {tensor.dtype}")
    save_file(tensors, str(path))
    return {"aliases": aliases, "sha256": sha256_file(path), "keys": sorted(tensors), "bytes": path.stat().st_size}


def _read_state(directory: Path, metadata: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload = metadata.get("tensor_payload")
    if not isinstance(payload, Mapping) or payload.get("path") != "tensors.safetensors":
        raise ValueError("invalid tensor payload declaration")
    path = directory / "tensors.safetensors"
    if not path.is_file() or sha256_file(path) != payload.get("sha256"):
        raise ValueError("tensor payload hash mismatch")
    state = load_file(str(path), device="cpu")
    result = _state_from_disk(state, payload.get("aliases", {}))
    expected = set(payload.get("full_keys", []))
    if expected and set(result) != expected:
        raise ValueError("tensor key set mismatch")
    for name, tensor in result.items():
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"non-finite tensor: {name}")
    return result


def write_round_bundle(directory: str | Path, bundle: RoundBundle) -> Path:
    destination = Path(directory)
    if bundle.round_number < 0 or not bundle.participant_ids or tuple(sorted(bundle.participant_ids)) != bundle.participant_ids:
        raise ValueError("invalid round participant metadata")

    def writer(root: Path) -> None:
        payload = _write_state(root / "tensors.safetensors", bundle.state)
        payload["path"] = "tensors.safetensors"
        payload["full_keys"] = sorted(bundle.state)
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "bundle_type": "round",
            "federation_id": bundle.federation_id,
            "round_number": bundle.round_number,
            "parent_digest": bundle.parent_digest,
            "config_digest": bundle.config_digest,
            "model_digest": bundle.model_digest,
            "tokenizer_identity": bundle.tokenizer_identity,
            "participant_ids": list(bundle.participant_ids),
            "local_steps": bundle.local_steps,
            "target_count_per_step": bundle.target_count_per_step,
            "tensor_payload": payload,
        }
        _write_json(root / "metadata.json", metadata)

    return _publish(destination, writer)


def write_update_bundle(directory: str | Path, bundle: UpdateBundle) -> Path:
    destination = Path(directory)
    if not bundle.participant_id or bundle.completed_steps <= 0 or bundle.target_count <= 0:
        raise ValueError("invalid update counters")

    def writer(root: Path) -> None:
        payload = _write_state(root / "tensors.safetensors", bundle.state)
        payload["path"] = "tensors.safetensors"
        payload["full_keys"] = sorted(bundle.state)
        unsigned = {
            "protocol_version": PROTOCOL_VERSION,
            "bundle_type": "update",
            "federation_id": bundle.federation_id,
            "round_number": bundle.round_number,
            "participant_id": bundle.participant_id,
            "parent_digest": bundle.parent_digest,
            "config_digest": bundle.config_digest,
            "model_digest": bundle.model_digest,
            "tokenizer_identity": bundle.tokenizer_identity,
            "completed_steps": bundle.completed_steps,
            "target_count": bundle.target_count,
            "tensor_payload": payload,
        }
        digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
        _write_json(root / "metadata.json", {**unsigned, "digest": digest})

    return _publish(destination, writer)


def load_round_bundle(directory: str | Path) -> RoundBundle:
    root = Path(directory)
    metadata = _metadata(root)
    if metadata.get("bundle_type") != "round":
        raise ValueError("not a round bundle")
    return RoundBundle(
        federation_id=str(metadata["federation_id"]), round_number=int(metadata["round_number"]),
        parent_digest=metadata.get("parent_digest"), config_digest=str(metadata["config_digest"]),
        model_digest=str(metadata["model_digest"]), tokenizer_identity=str(metadata["tokenizer_identity"]),
        participant_ids=tuple(str(item) for item in metadata["participant_ids"]), local_steps=int(metadata["local_steps"]),
        target_count_per_step=int(metadata["target_count_per_step"]), state=_read_state(root, metadata),
    )


def load_update_bundle(directory: str | Path) -> UpdateBundle:
    root = Path(directory)
    metadata = _metadata(root)
    if metadata.get("bundle_type") != "update":
        raise ValueError("not an update bundle")
    unsigned = {key: value for key, value in metadata.items() if key != "digest"}
    expected_digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if metadata.get("digest") != expected_digest:
        raise ValueError("update bundle digest mismatch")
    return UpdateBundle(
        federation_id=str(metadata["federation_id"]), round_number=int(metadata["round_number"]),
        participant_id=str(metadata["participant_id"]), parent_digest=str(metadata["parent_digest"]),
        config_digest=str(metadata["config_digest"]), model_digest=str(metadata["model_digest"]),
        tokenizer_identity=str(metadata["tokenizer_identity"]), completed_steps=int(metadata["completed_steps"]),
        target_count=int(metadata["target_count"]), state=_read_state(root, metadata), digest=str(metadata["digest"]),
    )


__all__ = ["RoundBundle", "UpdateBundle", "load_round_bundle", "load_update_bundle", "write_round_bundle", "write_update_bundle"]
