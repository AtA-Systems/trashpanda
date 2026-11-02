"""Optional, explicitly invoked exchange/publication adapters.

The local Trashpanda contract remains authoritative.  The adapter targets the
versioned Apache Sourcelume 0.0.1 dataset record shape where fields map cleanly
and preserves all other local evidence under an extension object.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.request import Request, urlopen

from .core import SCHEMA_VERSION, canonical_bytes, make_record, record_identity


SOURCELUME_VERSION = "0.0.1"
SOURCELUME_CONTEXT = "https://sourcelume.apache.org/context/0.0.1/sourcelume.jsonld"


@dataclass(frozen=True)
class MappingResult:
    payload: dict[str, Any]
    lossy_fields: tuple[str, ...]
    unsupported_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"payload": self.payload, "lossy_fields": list(self.lossy_fields), "unsupported_fields": list(self.unsupported_fields)}


def to_sourcelume(records: Iterable[Mapping[str, Any]]) -> MappingResult:
    values = list(records)
    inputs = [record for record in values if record.get("type") == "input"]
    if not inputs:
        raise ValueError("Sourcelume dataset mapping requires at least one input record")
    first = inputs[0]
    creator = [{"@type": "schema:Organization", "name": str(first.get("issuer", "unknown producer")), "role": "curator"}]
    payload: dict[str, Any] = {
        "@context": SOURCELUME_CONTEXT,
        "@type": "ProvenanceRecord",
        "@id": f"urn:trashpanda:{record_identity(first)}",
        "creator": creator,
        "origin": first.get("source_id", "unknown"),
        "added": first.get("acquired_at"),
        "distribution": [{"@type": "schema:DataDownload", "contentUrl": "withheld", "sha256": first.get("content_sha256")}],
        "trashpanda": {
            "mapping_version": SOURCELUME_VERSION,
            "local_record_ids": [record.get("id") for record in values if record.get("id")],
            "records": values,
        },
    }
    lossy = ("source_revision", "selection", "rights", "private_evidence")
    unsupported = ("run", "execution_segment", "policy_decision", "signature_trust")
    return MappingResult(payload=payload, lossy_fields=lossy, unsupported_fields=unsupported)


def preview_public(records: Iterable[Mapping[str, Any]], allowlist: Iterable[str]) -> MappingResult:
    allowed = set(allowlist)
    projected = [{key: value for key, value in record.items() if key in allowed} for record in records]
    return to_sourcelume(projected)


def write_exchange(records: Iterable[Mapping[str, Any]], output: str | Path) -> MappingResult:
    result = to_sourcelume(records)
    Path(output).write_text(json.dumps(result.payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def publish(
    exchange_path: str | Path,
    *,
    endpoint: str,
    token: str | None = None,
    dry_run: bool = True,
    transport: Callable[[str, Mapping[str, str], bytes], Any] | None = None,
) -> dict[str, Any]:
    """Publish only when explicitly called; dry-run is the safe default."""
    payload = Path(exchange_path).read_bytes()
    identity = record_identity(json.loads(payload))
    result = {"status": "preview" if dry_run else "pending", "endpoint": endpoint, "identity": identity, "payload_sha256": hashlib.sha256(payload).hexdigest()}
    if dry_run:
        result["public_payload"] = json.loads(payload)
        return result
    headers = {"Content-Type": "application/json", "Idempotency-Key": identity}
    if token:
        headers["Authorization"] = "Bearer [redacted]"
    sender = transport
    if sender is None:
        request = Request(endpoint, data=payload, headers={**headers, "Authorization": f"Bearer {token}"} if token else headers, method="POST")
        with urlopen(request, timeout=30) as response:  # nosec B310 - endpoint is adopter-supplied and explicit
            body = response.read()
        return {**result, "status": "published", "response": json.loads(body) if body else None}
    response = sender(endpoint, headers, payload)
    return {**result, "status": "published", "response": response}


def exchange_record(result: MappingResult) -> dict[str, Any]:
    return make_record(
        "exchange",
        {"mapping": "sourcelume", "version": SOURCELUME_VERSION, "lossy_fields": list(result.lossy_fields), "unsupported_fields": list(result.unsupported_fields)},
    )


__all__ = [
    "MappingResult",
    "SOURCELUME_CONTEXT",
    "SOURCELUME_VERSION",
    "exchange_record",
    "preview_public",
    "publish",
    "to_sourcelume",
    "write_exchange",
]
