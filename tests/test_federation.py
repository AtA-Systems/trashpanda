import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from trashpanda.federation.aggregate import aggregate_updates
from trashpanda.federation.bundles import RoundBundle, UpdateBundle, load_round_bundle, load_update_bundle, write_round_bundle, write_update_bundle
from trashpanda.federation.config import FederationConfig, ParticipantConfig
from trashpanda.federation.runner import make_demo_config, run_federation
from trashpanda.federation.provenance import export_federation
from trashpanda.provenance import verify_bundle


def _round():
    state = {"weight": torch.tensor([[1.0, 2.0]]), "counter": torch.tensor([1], dtype=torch.int64)}
    return RoundBundle("fed", 0, "", "cfg", "model", "tok", ("a", "b"), 1, 6, state)


def test_bundles_round_trip_and_aggregate(tmp_path):
    round_bundle = _round()
    round_path = write_round_bundle(tmp_path / "round", round_bundle)
    assert torch.equal(load_round_bundle(round_path).state["weight"], round_bundle.state["weight"])
    first = UpdateBundle("fed", 0, "a", "", "cfg", "model", "tok", 1, 2, {"weight": torch.tensor([[2.0, 2.0]]), "counter": torch.tensor([1])})
    second = UpdateBundle("fed", 0, "b", "", "cfg", "model", "tok", 1, 6, {"weight": torch.tensor([[6.0, 6.0]]), "counter": torch.tensor([1])})
    first_path = write_update_bundle(tmp_path / "a", first)
    second_path = write_update_bundle(tmp_path / "b", second)
    aggregate, summary = aggregate_updates(round_bundle, [load_update_bundle(first_path), load_update_bundle(second_path)])
    assert torch.equal(aggregate["weight"], torch.tensor([[5.0, 5.0]]))
    assert summary["target_count"] == 8


def test_local_federation_demo_has_round_accounting(tmp_path):
    config, _ = make_demo_config(tmp_path / "fixture")
    summary = run_federation(config, tmp_path / "run")
    assert summary["current_round"] == 3
    assert summary["target_count"] == 1488
    assert len(summary["commits"]) == 3
    assert {item["participants"] == ["alice", "bob"] for item in summary["commits"]} == {True}
    events = [json.loads(line) for line in (tmp_path / "run" / "processes.jsonl").read_text().splitlines()]
    assert len({event["pid"] for event in events if event["event"] == "worker_started"}) == 6
    exported = export_federation(tmp_path / "run", tmp_path / "export")
    assert verify_bundle(exported).ok


def test_failed_participant_round_resumes_without_double_counting(tmp_path):
    config, _ = make_demo_config(tmp_path / "fixture")
    with pytest.raises(RuntimeError, match="incomplete"):
        run_federation(config, tmp_path / "run", fail_participant="alice", fail_after_steps=1)
    resumed = run_federation(config, tmp_path / "run", resume=True)
    assert resumed["current_round"] == 3
    assert resumed["target_count"] == 1488
    assert len(resumed["commits"]) == 3

