"""Public federation evidence and native HF export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..export import export_hf
from ..model import build_model
from ..provenance import build_bundle_manifest, make_record, write_record
from .bundles import load_round_bundle
from .config import FederationConfig
from .coordinator import FederationCoordinator


def federation_records(run_dir: str | Path, config: FederationConfig) -> list[dict[str, Any]]:
    root = Path(run_dir)
    ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    federation = make_record("federation", {
        "federation_id": ledger["federation_id"],
        "protocol_version": config.protocol_version,
        "participants": [item.participant_id for item in config.participants],
        "rounds": int(config.rounds),
        "algorithm": "weighted-full-model-fedavg",
        "privacy": "individual updates visible to coordinator; secure aggregation and differential privacy are not implemented",
        "lineage": "participant source records withheld and not independently inspected by coordinator",
        "evidence_state": "observed",
    })
    records: list[dict[str, Any]] = [federation]
    for commit in ledger.get("commits", []):
        round_record = make_record("federation_round", {
            "federation_id": ledger["federation_id"],
            "round_number": int(commit["round"]),
            "status": "committed",
            "parent_digest": commit["parent_digest"],
            "output_digest": commit["output_digest"],
            "target_count": int(commit["target_count"]),
            "participant_ids": list(commit["participants"]),
            "evidence_state": "observed",
        })
        records.append(round_record)
        for participant_id, digest in commit["update_digests"].items():
            records.append(make_record("federation_contribution", {
                "federation_id": ledger["federation_id"],
                "round_number": int(commit["round"]),
                "participant_id": participant_id,
                "update_digest": digest,
                "target_count": int(commit["weights"][participant_id]),
                "source_lineage": "withheld",
                "evidence_state": "participant-attested/coordinator-observed",
            }))
    return records


def export_federation(run_dir: str | Path, output_dir: str | Path) -> Path:
    root = Path(run_dir)
    config = FederationConfig.from_file(root / "config.json")
    coordinator = FederationCoordinator(root, config, resume=True)
    if not coordinator.complete:
        raise ValueError("cannot export an incomplete federation")
    bundle = load_round_bundle(coordinator.rounds_dir / f"round-{coordinator.current_round_number:06d}")
    coordinator.close()
    model = build_model(config.model_config.model, seed=config.seed)
    model.load_state_dict(bundle.state)
    exported = export_hf(model, output_dir, config=config.model_config, write_manifest=False)
    records = federation_records(root, config)
    for record in records:
        write_record(exported, record)
    (exported / "trashpanda-federation.json").write_text(json.dumps({
        "format": "trashpanda.federation.export.v1",
        "federation_id": coordinator.federation_id,
        "rounds": coordinator.current_round_number,
        "accepted_target_count": coordinator.summary()["target_count"],
        "lineage": "participant source records withheld",
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    artifact = make_record("artifact", {
        "artifact_type": "federated-hf-export",
        "federation_id": coordinator.federation_id,
        "round": coordinator.current_round_number,
        "evidence_state": "observed",
    })
    records.append(artifact)
    write_record(exported, artifact)
    build_bundle_manifest(exported, records=records, artifact_type="federated-hf-export")
    return exported


__all__ = ["export_federation", "federation_records"]
