import copy

import pytest
import torch

from trashpanda.federation.aggregate import aggregate_updates
from trashpanda.federation.bundles import RoundBundle, UpdateBundle, load_round_bundle, write_round_bundle


pytestmark = pytest.mark.unit


def _bundles() -> tuple[RoundBundle, UpdateBundle, UpdateBundle]:
    parent = {"weight": torch.tensor([[1.0, 2.0]]), "counter": torch.tensor([1], dtype=torch.int64)}
    round_bundle = RoundBundle("fed", 0, "parent", "cfg", "model", "tok", ("alice", "bob"), 1, 8, parent)
    first = UpdateBundle("fed", 0, "alice", "parent", "cfg", "model", "tok", 1, 2, {"weight": torch.tensor([[2.0, 2.0]]), "counter": parent["counter"]})
    second = UpdateBundle("fed", 0, "bob", "parent", "cfg", "model", "tok", 1, 6, {"weight": torch.tensor([[6.0, 6.0]]), "counter": parent["counter"]})
    return round_bundle, first, second


def test_weighting_is_target_weighted_and_order_independent():
    round_bundle, first, second = _bundles()
    original = {key: value.clone() for key, value in round_bundle.state.items()}
    result, summary = aggregate_updates(round_bundle, [second, first])
    assert torch.equal(result["weight"], torch.tensor([[5.0, 5.0]]))
    assert summary["target_count"] == 8
    assert summary["weights"] == {"alice": 2, "bob": 6}
    assert all(torch.equal(round_bundle.state[key], value) for key, value in original.items())
    reverse, _ = aggregate_updates(round_bundle, [first, second])
    assert torch.equal(reverse["weight"], result["weight"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("federation_id", "other", "federation or round"),
        ("round_number", 1, "federation or round"),
        ("parent_digest", "stale", "parent digest"),
        ("config_digest", "other", "identity"),
        ("model_digest", "other", "identity"),
        ("tokenizer_identity", "other", "identity"),
        ("completed_steps", 2, "work counters"),
        ("target_count", 0, "work counters"),
    ],
)
def test_invalid_update_metadata_is_rejected(field, value, message):
    round_bundle, first, second = _bundles()
    changed = {field: value}
    update = UpdateBundle(**{**first.__dict__, **changed})
    with pytest.raises(ValueError, match=message):
        aggregate_updates(round_bundle, [update, second])


@pytest.mark.parametrize(
    "state",
    [
        {"weight": torch.tensor([[2.0]]), "counter": torch.tensor([1], dtype=torch.int64)},
        {"weight": torch.tensor([[float("nan"), 2.0]]), "counter": torch.tensor([1], dtype=torch.int64)},
        {"weight": torch.tensor([[float("inf"), 2.0]]), "counter": torch.tensor([1], dtype=torch.int64)},
        {"weight": torch.tensor([[2.0, 2.0]]), "counter": torch.tensor([2], dtype=torch.int64)},
    ],
)
def test_invalid_tensor_state_is_rejected(state):
    round_bundle, first, second = _bundles()
    update = UpdateBundle(**{**first.__dict__, "state": state})
    with pytest.raises(ValueError):
        aggregate_updates(round_bundle, [update, second])


def test_missing_or_unknown_participant_is_rejected():
    round_bundle, first, second = _bundles()
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_updates(round_bundle, [first])
    unknown = UpdateBundle(**{**second.__dict__, "participant_id": "carol"})
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_updates(round_bundle, [first, unknown])


def test_corrupt_bundle_payload_is_rejected(tmp_path):
    round_bundle, _, _ = _bundles()
    path = write_round_bundle(tmp_path / "round", round_bundle)
    payload = path / "tensors.safetensors"
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_round_bundle(path)
