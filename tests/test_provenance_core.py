import json

import pytest

from trashpanda.provenance import (
    DuplicateKeyError,
    build_bundle_manifest,
    canonical_bytes,
    inspect_bundle,
    make_record,
    parse_json,
    verify_bundle,
)
from trashpanda.provenance.exchange import to_sourcelume
from trashpanda.provenance.policy import evaluate_policy, example_policy
from trashpanda.provenance.signing import sign_manifest, verify_detached_signature
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer
from trashpanda.data import SequenceReader
from trashpanda.cli import main

pytestmark = pytest.mark.unit


def _records():
    source = make_record(
        "input",
        {
            "source_id": "synthetic:A",
            "content_sha256": "a" * 64,
            "source": "project-owned-fixture",
            "source_revision": "fixture-1",
            "selection": {"kind": "all-records"},
            "evidence_state": "observed",
        },
    )
    preparation = make_record(
        "preparation",
        {
            "split": "train",
            "tokenizer": {"name": "synthetic", "sha256": "b" * 64},
            "settings": {"context_length": 4},
            "sources": [source["id"]],
        },
    )
    run = make_record(
        "run",
        {
            "run_id": "run-synthetic",
            "status": "completed",
            "segments": [],
            "parents": [],
        },
    )
    return source, preparation, run


def test_canonical_json_is_key_order_independent_and_strict():
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(DuplicateKeyError):
        parse_json('{"a": 1, "a": 2}')
    with pytest.raises(ValueError):
        parse_json('{"a": NaN}')


def test_bundle_detects_payload_and_nested_record_tampering(tmp_path):
    source, preparation, run = _records()
    (tmp_path / "config.json").write_text('{"model":"synthetic"}\n', encoding="utf-8")
    build_bundle_manifest(tmp_path, records=[source, preparation, run], artifact_type="synthetic-model")
    result = verify_bundle(tmp_path)
    assert result.ok
    assert result.signatures["status"] == "unsigned"
    assert inspect_bundle(tmp_path)["artifact_type"] == "synthetic-model"
    (tmp_path / "config.json").write_text('{"model":"tampered"}\n', encoding="utf-8")
    assert verify_bundle(tmp_path).integrity["status"] == "invalid"


def test_policy_and_sourcelume_mapping_are_explicit(tmp_path):
    source, preparation, run = _records()
    policy = example_policy(strict=False)
    decision = evaluate_policy([source, preparation, run], policy)
    assert decision["decision"] == "allow"
    strict = example_policy(strict=True)
    assert evaluate_policy([source, preparation, run], strict)["decision"] == "deny"
    mapped = to_sourcelume([source, preparation, run])
    assert mapped.payload["@type"] == "ProvenanceRecord"
    assert "run" in mapped.unsupported_fields


def test_cli_inspect_and_verify_are_runtime_independent(tmp_path, capsys):
    source, preparation, run = _records()
    (tmp_path / "payload.bin").write_bytes(b"synthetic")
    build_bundle_manifest(tmp_path, records=[source, preparation, run], artifact_type="fixture")
    assert main(["provenance", "inspect", str(tmp_path), "--json"]) == 0
    assert main(["provenance", "verify", str(tmp_path), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"integrity"' in output
    assert '"status": "unsigned"' in output


def test_detached_signature_requires_explicit_trust(tmp_path):
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    source, preparation, run = _records()
    (tmp_path / "payload.bin").write_bytes(b"signed")
    build_bundle_manifest(tmp_path, records=[source, preparation, run], artifact_type="fixture")
    private = Ed25519PrivateKey.generate()
    sign_manifest(tmp_path, private, key_id="synthetic")
    public = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    assert verify_detached_signature(tmp_path, trust={})["status"] == "untrusted"
    assert verify_detached_signature(tmp_path, trust={"synthetic": public})["status"] == "valid"


def test_preparation_records_mixed_source_target_attribution(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text(json.dumps({"text": "w0 w1 w2 w3 w4"}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"text": "w5 w6 w7 w8 w9"}) + "\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepare_splits(
        prepared,
        train_path=[first, second],
        tokenizer=TinyWordLevelTokenizer(),
        context_length=4,
        source_descriptors=[{"source_id": "A"}, {"source_id": "B"}],
    )
    reader = SequenceReader(prepared)
    counts = reader.target_source_counts([0, 1, 1])
    assert counts == {"A": 5, "B": 4}
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    assert {record["type"] for record in manifest["provenance"]["records"]} == {"input", "preparation"}
    assert manifest["splits"]["train"]["source_target_counts"] == {"A": 4, "B": 5}
