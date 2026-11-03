from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trashpanda.federation import coordinator as coordinator_module
from trashpanda.federation.bundles import UpdateBundle, write_update_bundle
from trashpanda.federation.coordinator import FederationCoordinator
from trashpanda.federation.runner import make_demo_config


pytestmark = pytest.mark.unit


def _updates(coordinator: FederationCoordinator, root: Path) -> list[Path]:
    current = coordinator.current_round()
    result = []
    for participant_id in current.participant_ids:
        state = {name: value.clone() + (0.1 if value.dtype.is_floating_point else 0) for name, value in current.state.items()}
        update = UpdateBundle(
            federation_id=coordinator.federation_id,
            round_number=current.round_number,
            participant_id=participant_id,
            parent_digest=current.parent_digest,
            config_digest=current.config_digest,
            model_digest=current.model_digest,
            tokenizer_identity=current.tokenizer_identity,
            completed_steps=current.local_steps,
            target_count=current.target_count_per_step * current.local_steps,
            state=state,
        )
        result.append(write_update_bundle(root / participant_id, update))
    return result


def test_retries_are_idempotent_and_missing_participant_does_not_commit(tmp_path: Path):
    config, _ = make_demo_config(tmp_path / "fixture")
    with FederationCoordinator(tmp_path / "run", config) as coordinator:
        paths = _updates(coordinator, tmp_path / "updates")
        first_receipt = coordinator.accept_update(paths[0])
        assert coordinator.accept_update(paths[0]) == first_receipt
        before = coordinator.summary()
        with pytest.raises(ValueError, match="missing participant"):
            coordinator.commit_current_round()
        after = coordinator.summary()
        assert after["current_round"] == before["current_round"] == 0
        assert after["target_count"] == before["target_count"] == 0

        conflicting = paths[0].parent / "conflicting"
        original = coordinator.current_round()
        bad = UpdateBundle(
            federation_id=coordinator.federation_id,
            round_number=0,
            participant_id="alice",
            parent_digest=original.parent_digest,
            config_digest=original.config_digest,
            model_digest=original.model_digest,
            tokenizer_identity=original.tokenizer_identity,
            completed_steps=original.local_steps,
            target_count=original.target_count_per_step * original.local_steps,
            state={name: value.clone() + (0.2 if value.dtype.is_floating_point else 0) for name, value in original.state.items()},
        )
        write_update_bundle(conflicting, bad)
        with pytest.raises(ValueError, match="conflicting update"):
            coordinator.accept_update(conflicting)


def test_published_round_is_recovered_after_ledger_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, _ = make_demo_config(tmp_path / "fixture")
    run = tmp_path / "run"
    with FederationCoordinator(run, config) as coordinator:
        for path in _updates(coordinator, tmp_path / "updates"):
            coordinator.accept_update(path)
        original = coordinator_module.write_round_bundle
        failed = {"value": False}

        def write_then_fail(path, bundle):
            result = original(path, bundle)
            if not failed["value"]:
                failed["value"] = True
                raise RuntimeError("test crash after publication")
            return result

        monkeypatch.setattr(coordinator_module, "write_round_bundle", write_then_fail)
        with pytest.raises(RuntimeError, match="after publication"):
            coordinator.commit_current_round()
    monkeypatch.setattr(coordinator_module, "write_round_bundle", original)
    with FederationCoordinator(run, config, resume=True) as recovered:
        assert recovered.current_round_number == 1
        assert recovered.summary()["target_count"] == 496
        assert len(recovered.summary()["commits"]) == 1


def test_second_active_coordinator_is_rejected(tmp_path: Path):
    config, _ = make_demo_config(tmp_path / "fixture")
    first = FederationCoordinator(tmp_path / "run", config)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            FederationCoordinator(tmp_path / "run", config, resume=True)
    finally:
        first.close()
