"""Participant-local bounded training and crash-resumable private state."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..checkpoint import capture_rng_state, restore_rng_state
from ..data import PCG64Sampler, SequenceReader
from ..model import build_model, causal_loss
from ..train import optimizer_param_groups, seed_everything
from .bundles import RoundBundle, UpdateBundle, load_update_bundle, write_update_bundle
from .config import FederationConfig, ParticipantConfig


def _private_path(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "participant-state.pt"


def train_local_round(
    round_bundle: RoundBundle,
    federation: FederationConfig,
    participant: ParticipantConfig,
    private_state_dir: str | Path,
    output_bundle: str | Path,
    *,
    fail_after_steps: int | None = None,
) -> Path:
    """Train exactly the round's local budget and emit an allowlisted update.

    The private state file may contain optimizer, RNG, sampler and model state;
    none of it is placed in the exchange bundle.
    """
    if participant.participant_id not in round_bundle.participant_ids:
        raise ValueError(f"participant is not configured for this round: {participant.participant_id}")
    data = SequenceReader(participant.data_dir, split="train")
    model_config = federation.model_config
    if data.context_length != model_config.model.context_length or data.vocab_size != model_config.model.vocab_size:
        raise ValueError("participant data does not match federation model")
    state_path = _private_path(Path(private_state_dir))
    saved: dict[str, Any] | None = None
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
    if saved and int(saved.get("round_number", -1)) == round_bundle.round_number and saved.get("status") == "complete":
        existing = Path(saved["update_bundle"])
        if existing.exists():
            return existing
    seed_everything(federation.seed + participant.seed_offset, deterministic=True)
    model = build_model(model_config.model, seed=federation.seed + participant.seed_offset)
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, weight_decay=model_config.training.weight_decay),
        lr=federation.learning_rate,
        betas=(model_config.training.beta1, model_config.training.beta2),
    )
    sampler = PCG64Sampler(len(data), federation.seed + participant.seed_offset, repeat_data=True)
    completed = 0
    if saved and int(saved.get("round_number", -1)) == round_bundle.round_number and saved.get("status") == "running":
        if saved.get("parent_digest") != round_bundle.parent_digest:
            raise ValueError("private interrupted state has a different round parent")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load_state_dict(saved["sampler"])
        restore_rng_state(saved["rng"])
        completed = int(saved.get("completed_steps", 0))
    else:
        model.load_state_dict(round_bundle.state)
        if saved and saved.get("sampler"):
            sampler.load_state_dict(saved["sampler"])
            restore_rng_state(saved["rng"])
    model.train()
    targets_per_step = federation.batch_size * (data.context_length - 1)
    while completed < federation.local_steps:
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(federation.batch_size // federation.microbatch_size):
            indices = sampler.next_indices(federation.microbatch_size)
            inputs, targets = data.batch(indices)
            loss = causal_loss(model, inputs, targets) / (federation.batch_size // federation.microbatch_size)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite participant loss")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.training.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        completed += 1
        torch.save({
            "round_number": round_bundle.round_number,
            "parent_digest": round_bundle.parent_digest,
            "status": "running" if completed < federation.local_steps else "complete",
            "completed_steps": completed,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "rng": capture_rng_state(),
            "update_bundle": str(Path(output_bundle)),
        }, state_path)
        if fail_after_steps is not None and completed >= fail_after_steps:
            raise RuntimeError("injected participant failure")
    update = UpdateBundle(
        federation_id=round_bundle.federation_id,
        round_number=round_bundle.round_number,
        participant_id=participant.participant_id,
        parent_digest=round_bundle.parent_digest or "",
        config_digest=round_bundle.config_digest,
        model_digest=round_bundle.model_digest,
        tokenizer_identity=round_bundle.tokenizer_identity,
        completed_steps=completed,
        target_count=completed * targets_per_step,
        state={key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    path = write_update_bundle(output_bundle, update)
    torch.save({
        "round_number": round_bundle.round_number,
        "parent_digest": round_bundle.parent_digest,
        "status": "complete",
        "completed_steps": completed,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": capture_rng_state(),
        "update_bundle": str(path),
    }, state_path)
    return path


def load_completed_update(path: str | Path) -> UpdateBundle:
    return load_update_bundle(path)


__all__ = ["load_completed_update", "train_local_round"]
