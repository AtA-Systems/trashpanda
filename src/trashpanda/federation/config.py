"""Frozen configuration for the local federation protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..config import Config, config_from_dict, model_recipe
from ..provenance import canonical_bytes


PROTOCOL_VERSION = "trashpanda.federation.v1"


@dataclass(frozen=True)
class ParticipantConfig:
    participant_id: str
    data_dir: str
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if not self.participant_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in self.participant_id):
            raise ValueError("participant_id must contain only letters, digits, '_' or '-'")
        path = Path(self.data_dir)
        if path.is_absolute() and not path.exists():
            # Absolute paths are valid for a local run; missing paths are
            # rejected by the coordinator after config parsing.
            pass
        if self.seed_offset < 0:
            raise ValueError("seed_offset must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FederationConfig:
    participants: tuple[ParticipantConfig, ...]
    rounds: int = 3
    local_steps: int = 2
    batch_size: int = 4
    microbatch_size: int = 2
    timeout_seconds: float = 120.0
    seed: int = 42
    architecture: str = "llama"
    recipe: str = "tiny"
    tokenizer_identity: str = "trashpanda/tiny-wordlevel@project"
    protocol_version: str = PROTOCOL_VERSION
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported federation protocol {self.protocol_version!r}")
        if len(self.participants) < 2:
            raise ValueError("federation requires at least two participants")
        ids = [item.participant_id for item in self.participants]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("participants must be unique and sorted by participant_id")
        for name in ("rounds", "local_steps", "batch_size", "microbatch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size % self.microbatch_size:
            raise ValueError("batch_size must divide evenly by microbatch_size")
        if self.timeout_seconds <= 0 or self.learning_rate <= 0 or self.seed < 0:
            raise ValueError("invalid federation limits or optimizer settings")
        if self.recipe != "tiny" or self.architecture != "llama":
            raise ValueError("the local prototype supports only the tiny Llama recipe")

    @property
    def model_config(self) -> Config:
        return model_recipe(self.recipe, self.architecture)

    def to_dict(self, *, include_paths: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["participants"] = [item.to_dict() for item in self.participants]
        if not include_paths:
            for item in value["participants"]:
                item.pop("data_dir", None)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FederationConfig":
        allowed = {"participants", "rounds", "local_steps", "batch_size", "microbatch_size", "timeout_seconds", "seed", "architecture", "recipe", "tokenizer_identity", "protocol_version", "learning_rate"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown federation config fields: {sorted(unknown)}")
        participants = tuple(ParticipantConfig(**dict(item)) for item in value.get("participants", ()))
        return cls(participants=participants, **{key: value[key] for key in allowed - {"participants"} if key in value})

    @classmethod
    def from_file(cls, path: str | Path) -> "FederationConfig":
        source = Path(path).resolve()
        value = json.loads(source.read_text(encoding="utf-8"))
        participants = []
        for item in value.get("participants", []):
            current = dict(item)
            data_dir = Path(str(current["data_dir"]))
            if not data_dir.is_absolute():
                current["data_dir"] = str((source.parent / data_dir).resolve())
            participants.append(current)
        value["participants"] = participants
        return cls.from_dict(value)

    def digest(self, *, include_paths: bool = False) -> str:
        return hashlib.sha256(canonical_bytes(self.to_dict(include_paths=include_paths))).hexdigest()


def default_federation_config(participants: tuple[ParticipantConfig, ...]) -> FederationConfig:
    return FederationConfig(participants=tuple(sorted(participants, key=lambda item: item.participant_id)))


__all__ = ["FederationConfig", "ParticipantConfig", "PROTOCOL_VERSION", "default_federation_config"]
