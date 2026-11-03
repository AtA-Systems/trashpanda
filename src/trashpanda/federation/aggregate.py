"""Pure deterministic FedAvg-style aggregation."""
from __future__ import annotations

from typing import Any, Iterable

import torch

from .bundles import RoundBundle, UpdateBundle


def aggregate_updates(round_bundle: RoundBundle, updates: Iterable[UpdateBundle]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    ordered = sorted(list(updates), key=lambda item: item.participant_id)
    expected = set(round_bundle.participant_ids)
    if {item.participant_id for item in ordered} != expected or len(ordered) != len(expected):
        raise ValueError("aggregation requires exactly one update from every configured participant")
    total = sum(int(item.target_count) for item in ordered)
    if total <= 0:
        raise ValueError("aggregation target count must be positive")
    for item in ordered:
        if item.federation_id != round_bundle.federation_id or item.round_number != round_bundle.round_number:
            raise ValueError("update federation or round does not match round bundle")
        if item.parent_digest != (round_bundle.parent_digest or ""):
            raise ValueError("update parent digest does not match round bundle")
        if item.config_digest != round_bundle.config_digest or item.model_digest != round_bundle.model_digest or item.tokenizer_identity != round_bundle.tokenizer_identity:
            raise ValueError("update identity does not match round bundle")
        if item.completed_steps != round_bundle.local_steps or item.target_count <= 0 or item.target_count > round_bundle.target_count_per_step * round_bundle.local_steps:
            raise ValueError("update work counters do not match authorized local work")
        if set(item.state) != set(round_bundle.state):
            raise ValueError("update tensor keys do not match parent model")
        for name, parent in round_bundle.state.items():
            value = item.state[name]
            if value.shape != parent.shape or value.dtype != parent.dtype:
                raise ValueError(f"update tensor shape or dtype mismatch: {name}")
            if value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"non-finite update tensor: {name}")
    result: dict[str, torch.Tensor] = {}
    parent_aliases: dict[int, str] = {}
    for name, parent in round_bundle.state.items():
        pointer = parent.data_ptr()
        if pointer in parent_aliases:
            result[name] = result[parent_aliases[pointer]]
            continue
        parent_aliases[pointer] = name
        if parent.dtype.is_floating_point:
            value = torch.zeros_like(parent, dtype=torch.float32)
            for item in ordered:
                value.add_(item.state[name].float(), alpha=float(item.target_count) / total)
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"non-finite aggregate tensor: {name}")
            result[name] = value.to(dtype=parent.dtype)
        else:
            if any(not torch.equal(item.state[name], parent) for item in ordered):
                raise ValueError(f"non-floating tensor changed during aggregation: {name}")
            result[name] = parent.detach().cpu().clone()
    return result, {
        "participants": [item.participant_id for item in ordered],
        "target_count": total,
        "weights": {item.participant_id: int(item.target_count) for item in ordered},
        "update_digests": {item.participant_id: item.digest for item in ordered},
    }


__all__ = ["aggregate_updates"]
