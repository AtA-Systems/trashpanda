"""Portable provenance records, canonical identities, and offline bundles.

The provenance layer deliberately has no dependency on torch, transformers, or
network access.  It is a small, JSON based contract that can be used by a
recipient who only has an exported directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.0"
RECORD_FORMAT = "trashpanda.provenance.record"
BUNDLE_FORMAT = "trashpanda.provenance.bundle"
MANIFEST_NAME = "provenance-manifest.json"
COMPAT_MANIFEST_NAME = "trashpanda-provenance.json"
SIGNATURE_NAME = "provenance-signature.json"

RECORD_TYPES = frozenset(
    {
        "input",
        "preparation",
        "run",
        "execution_segment",
        "artifact",
        "evaluation",
        "claim",
        "policy_decision",
        "exchange",
    }
)


class ProvenanceError(ValueError):
    """Raised for malformed or unsupported provenance data."""


class UnsupportedVersionError(ProvenanceError):
    """Raised when a reader cannot safely interpret a record or bundle."""


class DuplicateKeyError(ProvenanceError):
    """Raised when JSON contains duplicate object keys."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> Any:
    raise ProvenanceError(f"non-finite JSON number is not allowed: {value}")


def parse_json(value: str | bytes) -> Any:
    """Parse JSON using the strict rules required by the record contract."""
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"invalid JSON at line {exc.lineno}, column {exc.colno}") from exc


def _normalise(value: Any, path: str = "$", *, allow_float: bool = True) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53:
            raise ProvenanceError(f"integer outside JSON interoperability range at {path}")
        return value
    if isinstance(value, float):
        if not allow_float:
            raise ProvenanceError(f"floating-point values are not allowed at {path}")
        if not (value == value and abs(value) != float("inf")):
            raise ProvenanceError(f"non-finite number at {path}")
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProvenanceError(f"object key at {path} is not a string")
            if key in result:
                raise DuplicateKeyError(f"duplicate object key at {path}: {key}")
            result[key] = _normalise(item, f"{path}.{key}", allow_float=allow_float)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalise(item, f"{path}[{index}]", allow_float=allow_float) for index, item in enumerate(value)]
    raise ProvenanceError(f"unsupported JSON value at {path}: {type(value).__name__}")


def canonical_bytes(value: Any, *, exclude: Iterable[str] = ()) -> bytes:
    """Return deterministic UTF-8 JSON bytes for an identity or signature."""
    excluded = set(exclude)
    if excluded and isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key not in excluded}
    normalised = _normalise(value)
    return json.dumps(
        normalised,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file without reading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def record_identity(record: Mapping[str, Any]) -> str:
    """Hash record content without its self-referential identity/signature."""
    return sha256_bytes(canonical_bytes(record, exclude=("id", "record_id", "signature", "signatures")))


def make_record(record_type: str, fields: Mapping[str, Any], *, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    if record_type not in RECORD_TYPES:
        raise ProvenanceError(f"unsupported record type: {record_type}")
    reserved = {"format", "schema_version", "type", "id", "record_id", "signature", "signatures"}
    if reserved & set(fields):
        raise ProvenanceError("record format, type, identity, and signature fields are serializer-owned")
    record = {
        "format": RECORD_FORMAT,
        "schema_version": schema_version,
        "type": record_type,
        **dict(fields),
    }
    record["id"] = record_identity(record)
    validate_record(record)
    return record


def validate_record(record: Mapping[str, Any], *, expected_type: str | None = None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ProvenanceError("record must be a JSON object")
    if record.get("format") != RECORD_FORMAT:
        raise ProvenanceError("unsupported provenance record format")
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnsupportedVersionError(f"unsupported provenance schema version: {version!r}")
    record_type = record.get("type")
    if record_type not in RECORD_TYPES:
        raise ProvenanceError(f"unsupported provenance record type: {record_type!r}")
    if expected_type and record_type != expected_type:
        raise ProvenanceError(f"expected {expected_type} record, got {record_type}")
    identity = record.get("id")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ProvenanceError("record id must be a lowercase SHA-256 digest")
    if record_identity(record) != identity:
        raise ProvenanceError(f"record identity mismatch: {identity}")
    required = {
        "input": ("source_id", "content_sha256"),
        "preparation": ("split", "tokenizer", "settings"),
        "run": ("run_id", "status", "segments"),
        "execution_segment": ("session_id", "status", "start_counters", "end_counters"),
        "artifact": ("artifact_type",),
        "evaluation": ("split", "settings", "counts"),
        "claim": ("subject", "issuer", "purpose"),
        "policy_decision": ("policy_id", "decision", "reasons"),
        "exchange": ("mapping", "version"),
    }[record_type]
    missing = [key for key in required if key not in record]
    if missing:
        raise ProvenanceError(f"{record_type} record missing fields: {', '.join(missing)}")
    return dict(record)


def record_ref(record: Mapping[str, Any]) -> dict[str, str]:
    validate_record(record)
    return {"id": str(record["id"]), "type": str(record["type"])}


def safe_relative_path(value: str | Path) -> str:
    """Validate and normalize a path stored in a portable bundle."""
    raw = str(value).replace("\\", "/")
    path = Path(raw)
    if not raw or path.is_absolute() or "\x00" in raw:
        raise ProvenanceError(f"unsafe bundle path: {value!r}")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ProvenanceError(f"unsafe bundle path: {value!r}")
    return "/".join(parts)


def file_entry(root: str | Path, relative_path: str | Path) -> dict[str, Any]:
    relative = safe_relative_path(relative_path)
    target = Path(root) / Path(relative)
    if target.is_symlink() or not target.is_file():
        raise ProvenanceError(f"bundle payload is not a regular file: {relative}")
    return {"path": relative, "size": target.stat().st_size, "sha256": sha256_file(target)}


def _iter_payload_files(root: Path) -> list[str]:
    result: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ProvenanceError(f"symlinks are not allowed in bundles: {relative}")
        if path.is_file() and relative not in {MANIFEST_NAME, COMPAT_MANIFEST_NAME, SIGNATURE_NAME}:
            result.append(safe_relative_path(relative))
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_record(root: str | Path, record: Mapping[str, Any]) -> Path:
    record = validate_record(record)
    path = Path(root) / "provenance" / "records" / f"{record['id']}.json"
    _write_json(path, record)
    return path


def build_bundle_manifest(
    root: str | Path,
    *,
    records: Iterable[Mapping[str, Any]] = (),
    parents: Iterable[Mapping[str, Any]] = (),
    artifact_type: str = "model",
    source_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Write records and return a non-circular completion manifest."""
    bundle = Path(root)
    bundle.mkdir(parents=True, exist_ok=True)
    refs: list[dict[str, str]] = []
    for record in records:
        refs.append(record_ref(record))
        write_record(bundle, record)
    parent_refs = [record_ref(parent) for parent in parents]
    files = [file_entry(bundle, relative) for relative in _iter_payload_files(bundle)]
    unsigned: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "files": files,
        "records": sorted(refs, key=lambda item: (item["type"], item["id"])),
        "parents": sorted(parent_refs, key=lambda item: (item["type"], item["id"])),
    }
    if source_checkpoint_sha256:
        unsigned["source_checkpoint_sha256"] = source_checkpoint_sha256
    manifest = dict(unsigned)
    manifest["artifact_id"] = record_identity(unsigned)
    _write_json(bundle / MANIFEST_NAME, manifest)
    # Keep the old sidecar name as a compact compatibility pointer.  It is not
    # part of the payload set and never replaces the completion manifest.
    _write_json(
        bundle / COMPAT_MANIFEST_NAME,
        {
            "format": "trashpanda.export.provenance.v2",
            "artifact_id": manifest["artifact_id"],
            "manifest": MANIFEST_NAME,
            "records": manifest["records"],
        },
    )
    return manifest


def _load_object(path: Path) -> Any:
    try:
        return parse_json(path.read_bytes())
    except OSError as exc:
        raise ProvenanceError(f"cannot read {path}: {exc}") from exc


def load_bundle_manifest(root: str | Path) -> dict[str, Any]:
    bundle = Path(root)
    path = bundle / MANIFEST_NAME if bundle.is_dir() else bundle
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.exists() and bundle.is_dir():
        path = bundle / COMPAT_MANIFEST_NAME
    value = _load_object(path)
    if value.get("format") != BUNDLE_FORMAT:
        raise UnsupportedVersionError("bundle has no supported completion manifest")
    return value


def _status(status: str, *reasons: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": status, "reasons": [dict(reason) for reason in reasons]}


@dataclass(frozen=True)
class VerificationResult:
    """Stable structured result used by the API and CLI."""

    path: str
    artifact_id: str | None
    schema: dict[str, Any]
    integrity: dict[str, Any]
    completeness: dict[str, Any]
    signatures: dict[str, Any]
    policy: dict[str, Any]
    records: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.schema["status"] == "valid" and self.integrity["status"] == "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "artifact_id": self.artifact_id,
            "schema": self.schema,
            "integrity": self.integrity,
            "completeness": self.completeness,
            "signatures": self.signatures,
            "policy": self.policy,
            "records": list(self.records),
            "ok": self.ok,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def _record_paths(bundle: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in manifest.get("records", []):
        identity = item.get("id") if isinstance(item, Mapping) else None
        if not isinstance(identity, str):
            continue
        path = bundle / "provenance" / "records" / f"{identity}.json"
        paths[identity] = path
    return paths


def _find_cycles(records: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    graph: dict[str, list[str]] = {}
    for identity, record in records.items():
        parents = record.get("parents", record.get("parent_ids", []))
        if isinstance(parents, list):
            graph[identity] = [
                item.get("id") if isinstance(item, Mapping) else item
                for item in parents
                if isinstance(item, (str, Mapping))
            ]
        else:
            graph[identity] = []
    cycles: list[list[str]] = []
    visiting: list[str] = []

    def visit(node: str) -> None:
        if node in visiting:
            cycles.append(visiting[visiting.index(node) :] + [node])
            return
        if node not in graph:
            return
        visiting.append(node)
        for parent in graph[node]:
            visit(parent)
        visiting.pop()

    for node in graph:
        visit(node)
    return cycles


def inspect_bundle(root: str | Path) -> dict[str, Any]:
    """Inspect JSON metadata without importing model runtimes."""
    bundle = Path(root)
    try:
        manifest = load_bundle_manifest(bundle)
    except UnsupportedVersionError:
        legacy_path = bundle / COMPAT_MANIFEST_NAME
        legacy = _load_object(legacy_path)
        return {
            "format": legacy.get("format", "legacy"),
            "legacy": True,
            "artifact_id": legacy.get("artifact_id"),
            "records": [],
            "limitations": [
                "completion manifest unavailable",
                "payload hashes and evidence closure were not captured by the legacy writer",
                "historical timestamps, approvals, and source revisions remain unknown",
            ],
        }
    records: list[dict[str, Any]] = []
    for identity, path in sorted(_record_paths(bundle, manifest).items()):
        if path.is_file():
            record = validate_record(_load_object(path))
            records.append({
                "id": identity,
                "type": record["type"],
                "status": record.get("status"),
                "summary": {key: record[key] for key in ("run_id", "session_id", "source_id", "artifact_type") if key in record},
            })
    return {
        "format": manifest["format"],
        "schema_version": manifest["schema_version"],
        "artifact_id": manifest.get("artifact_id"),
        "artifact_type": manifest.get("artifact_type"),
        "files": manifest.get("files", []),
        "records": records,
        "parents": manifest.get("parents", []),
    }


def verify_bundle(
    root: str | Path,
    *,
    policy: Mapping[str, Any] | None = None,
    trust: Mapping[str, Any] | None = None,
) -> VerificationResult:
    """Verify a bundle's schema, bytes, evidence closure, and optional layers."""
    bundle = Path(root)
    schema = _status("invalid", {"code": "missing_manifest", "path": MANIFEST_NAME})
    integrity = _status("not_evaluated")
    completeness = _status("not_evaluated")
    signatures = _status("not_evaluated")
    policy_result = _status("not_evaluated")
    records: list[dict[str, Any]] = []
    artifact_id: str | None = None
    try:
        manifest = load_bundle_manifest(bundle)
        artifact_id = manifest.get("artifact_id")
        unsigned = {key: value for key, value in manifest.items() if key != "artifact_id"}
        if not isinstance(artifact_id, str) or record_identity(unsigned) != artifact_id:
            raise ProvenanceError("bundle artifact_id does not match canonical manifest bytes")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise UnsupportedVersionError(f"unsupported bundle schema version: {manifest.get('schema_version')!r}")
        seen: set[str] = set()
        for item in manifest.get("files", []):
            if not isinstance(item, Mapping):
                raise ProvenanceError("manifest file entry is not an object")
            relative = safe_relative_path(str(item.get("path", "")))
            if relative in seen:
                raise ProvenanceError(f"duplicate manifest path: {relative}")
            seen.add(relative)
            actual = file_entry(bundle, relative)
            if actual["size"] != item.get("size") or actual["sha256"] != item.get("sha256"):
                raise ProvenanceError(f"payload hash or size mismatch: {relative}")
        actual_files = set(_iter_payload_files(bundle))
        extra = sorted(actual_files - seen)
        missing = sorted(seen - actual_files)
        if missing:
            raise ProvenanceError(f"manifest payload is missing: {missing[0]}")
        if extra:
            raise ProvenanceError(f"unlisted bundle payload: {extra[0]}")
        integrity = _status("valid")
        schema = _status("valid")
        record_map: dict[str, dict[str, Any]] = {}
        missing_records: list[str] = []
        for identity, path in _record_paths(bundle, manifest).items():
            if not path.is_file():
                missing_records.append(identity)
                continue
            record = validate_record(_load_object(path))
            record_map[identity] = record
            records.append({"id": identity, "type": record["type"], "status": record.get("status")})
        referenced_ids = {item.get("id") for item in manifest.get("records", []) if isinstance(item, Mapping)}
        missing_refs = sorted(str(identity) for identity in referenced_ids - set(record_map))
        # Check declared digest-addressed record links without treating a
        # checkpoint/file SHA-256 in a parent field as a record reference.
        linked_ids: set[str] = set()
        for record in record_map.values():
            for field in ("sources", "input_records", "segments", "evidence_refs", "records"):
                values = record.get(field, [])
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                            linked_ids.add(value)
                        elif isinstance(value, Mapping) and isinstance(value.get("id"), str):
                            linked_ids.add(value["id"])
        missing_refs.extend(sorted(linked_ids - set(record_map)))
        missing_refs = sorted(set(missing_refs))
        cycles = _find_cycles(record_map)
        if missing_records or missing_refs or cycles:
            reasons: list[dict[str, Any]] = []
            if missing_records or missing_refs:
                reasons.append({"code": "missing_evidence", "ids": sorted(set(missing_records + missing_refs))})
            if cycles:
                reasons.append({"code": "ancestry_cycle", "cycles": cycles})
            completeness = _status("incomplete", *reasons)
        else:
            completeness = _status("complete")
        signature_path = bundle / SIGNATURE_NAME
        if signature_path.exists():
            try:
                from .signing import verify_detached_signature

                signatures = verify_detached_signature(bundle, trust=trust or {})
            except Exception as exc:  # optional backend/configuration errors are results, not crashes
                signatures = _status("invalid", {"code": "signature_verification_error", "message": str(exc)})
        else:
            signatures = _status("unsigned", {"code": "unsigned_artifact"})
        if policy is not None:
            from .policy import evaluate_policy

            policy_result = evaluate_policy(record_map.values(), policy, signature_result=signatures)
    except UnsupportedVersionError as exc:
        schema = _status("unsupported", {"code": "unsupported_version", "message": str(exc)})
    except (ProvenanceError, OSError, KeyError, TypeError) as exc:
        schema = _status("invalid", {"code": "malformed_bundle", "message": str(exc)})
        integrity = _status("invalid", {"code": "verification_failed", "message": str(exc)})
    return VerificationResult(
        path=str(bundle),
        artifact_id=artifact_id,
        schema=schema,
        integrity=integrity,
        completeness=completeness,
        signatures=signatures,
        policy=policy_result,
        records=tuple(records),
    )


__all__ = [
    "BUNDLE_FORMAT",
    "COMPAT_MANIFEST_NAME",
    "DuplicateKeyError",
    "MANIFEST_NAME",
    "ProvenanceError",
    "RECORD_FORMAT",
    "RECORD_TYPES",
    "SCHEMA_VERSION",
    "SIGNATURE_NAME",
    "UnsupportedVersionError",
    "VerificationResult",
    "build_bundle_manifest",
    "canonical_bytes",
    "canonical_json_bytes",
    "file_entry",
    "hash_file",
    "inspect_bundle",
    "inspect",
    "load_bundle_manifest",
    "make_record",
    "make_record_id",
    "parse_json",
    "record_identity",
    "record_ref",
    "safe_relative_path",
    "sha256_bytes",
    "sha256_file",
    "validate_record",
    "verify_bundle",
    "verify",
    "write_record",
]

# Friendly aliases used by small integrations that prefer noun-style names.
canonical_json_bytes = canonical_bytes
hash_file = sha256_file
make_record_id = record_identity
verify = verify_bundle
inspect = inspect_bundle
