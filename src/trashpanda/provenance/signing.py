"""Optional detached Ed25519 signatures for offline provenance bundles."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping

from .core import SCHEMA_VERSION, SIGNATURE_NAME, canonical_bytes, load_bundle_manifest, sha256_bytes


SIGNATURE_FORMAT = "trashpanda.provenance.detached-signature"


def _backend() -> Any:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - exercised in environments without optional crypto
        raise RuntimeError("optional signing requires the 'cryptography' package") from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def _read_key(value: Any, *, private: bool) -> Any:
    serialization, Ed25519PrivateKey, Ed25519PublicKey = _backend()
    if private and isinstance(value, Ed25519PrivateKey):
        return value
    if not private and isinstance(value, Ed25519PublicKey):
        return value
    raw = Path(value).read_bytes() if isinstance(value, (str, Path)) else bytes(value)
    if private:
        return serialization.load_pem_private_key(raw, password=None)
    return serialization.load_pem_public_key(raw)


def sign_manifest(root: str | Path, private_key: Any, *, key_id: str, output: str | Path | None = None) -> Path:
    """Sign canonical completion-manifest bytes and store a detached envelope."""
    bundle = Path(root)
    manifest = load_bundle_manifest(bundle)
    key = _read_key(private_key, private=True)
    signature = key.sign(canonical_bytes(manifest))
    value = {
        "format": SIGNATURE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "algorithm": "Ed25519",
        "key_id": key_id,
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    destination = Path(output) if output else bundle / SIGNATURE_NAME
    destination.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return destination


def verify_detached_signature(root: str | Path, *, trust: Mapping[str, Any]) -> dict[str, Any]:
    bundle = Path(root)
    manifest = load_bundle_manifest(bundle)
    value = json.loads((bundle / SIGNATURE_NAME).read_text(encoding="utf-8"))
    if value.get("format") != SIGNATURE_FORMAT or value.get("algorithm") != "Ed25519":
        return {"status": "invalid", "reasons": [{"code": "unsupported_signature", "message": "unsupported signature format or algorithm"}]}
    key_id = value.get("key_id")
    if key_id not in trust:
        return {"status": "untrusted", "reasons": [{"code": "untrusted_signer", "key_id": key_id}]}
    expected_manifest_hash = sha256_bytes(canonical_bytes(manifest))
    if value.get("manifest_sha256") != expected_manifest_hash:
        return {"status": "invalid", "reasons": [{"code": "signed_manifest_mismatch"}]}
    try:
        key = _read_key(trust[key_id], private=False)
        key.verify(base64.b64decode(value["signature"], validate=True), canonical_bytes(manifest))
    except Exception as exc:
        return {"status": "invalid", "reasons": [{"code": "bad_signature", "message": str(exc)}]}
    return {"status": "valid", "reasons": [], "key_id": key_id, "algorithm": "Ed25519"}


__all__ = ["SIGNATURE_FORMAT", "sign_manifest", "verify_detached_signature"]
