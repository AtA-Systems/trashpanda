"""Declarative adopter policy evaluation.

The evaluator makes no governance decision on its own.  Without a supplied
policy it is intentionally not evaluated; policy files are data, never Python.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .core import SCHEMA_VERSION, canonical_bytes, make_record, record_identity


POLICY_FORMAT = "trashpanda.provenance.policy"


def load_policy(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_policy(value)
    return value


def validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if policy.get("format") != POLICY_FORMAT:
        raise ValueError(f"unsupported policy format: {policy.get('format')!r}")
    if policy.get("version") != 1:
        raise ValueError(f"unsupported policy version: {policy.get('version')!r}")
    if not isinstance(policy.get("name"), str) or not policy["name"]:
        raise ValueError("policy name must be nonempty")
    for key in ("required_evidence", "allowed_issuers", "disallowed_source_ids"):
        if key in policy and not isinstance(policy[key], list) or key in policy and not all(isinstance(item, str) for item in policy[key]):
            raise ValueError(f"policy {key} must be a list of strings")
    if policy.get("on_unknown", "indeterminate") not in {"indeterminate", "deny", "allow"}:
        raise ValueError("policy on_unknown must be indeterminate, deny, or allow")
    return dict(policy)


def policy_identity(policy: Mapping[str, Any]) -> str:
    validate_policy(policy)
    return record_identity({"format": POLICY_FORMAT, **dict(policy)})


def _reason(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **extra}


def evaluate_policy(
    records: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    signature_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate explicit facts against an adopter's declarative policy."""
    policy = validate_policy(policy)
    values = list(records)
    inputs = [record for record in values if record.get("type") == "input"]
    claims = [record for record in values if record.get("type") == "claim"]
    reasons: list[dict[str, Any]] = []
    unknown = False
    required = set(policy.get("required_evidence", []))
    available = {str(record.get("type")) for record in values}
    for evidence in sorted(required - available):
        unknown = True
        reasons.append(_reason("missing_required_evidence", f"required evidence is missing: {evidence}", evidence=evidence))
    disallowed = set(policy.get("disallowed_source_ids", []))
    for record in inputs:
        if record.get("source_id") in disallowed:
            reasons.append(_reason("disallowed_input", f"input is disallowed by policy: {record.get('source_id')}", source_id=record.get("source_id")))
    allowed_issuers = set(policy.get("allowed_issuers", []))
    if allowed_issuers:
        matching = [claim for claim in claims if claim.get("issuer") in allowed_issuers]
        if not matching:
            unknown = True
            reasons.append(_reason("no_trusted_claim", "no claim has an issuer trusted by this policy"))
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for claim in claims:
        if claim.get("status") in {"withdrawn", "superseded"}:
            reasons.append(_reason("withdrawn_claim", "a supplied claim is withdrawn or superseded", claim_id=claim.get("id")))
        if isinstance(claim.get("expires_at"), str) and claim["expires_at"] < now:
            unknown = True
            reasons.append(_reason("expired_claim", "a supplied claim has expired", claim_id=claim.get("id")))
    if policy.get("require_signed_claims"):
        if not signature_result or signature_result.get("status") != "valid":
            unknown = True
            reasons.append(_reason("claims_not_authenticated", "policy requires a valid supplied signature"))
    intended_use = policy.get("intended_use")
    if intended_use:
        mismatched = [claim for claim in claims if claim.get("purpose") not in {intended_use, "unknown"}]
        if mismatched:
            reasons.append(_reason("intended_use_mismatch", "a claim does not match the policy intended use", intended_use=intended_use))
    if any(reason["code"] in {"disallowed_input", "intended_use_mismatch"} for reason in reasons):
        decision = "deny"
    elif unknown:
        decision = policy.get("on_unknown", "indeterminate")
    else:
        decision = "allow"
    result = {
        "format": "trashpanda.provenance.policy-decision",
        "schema_version": SCHEMA_VERSION,
        "status": decision,
        "policy_id": policy_identity(policy),
        "policy_name": policy["name"],
        "evaluator_version": "trashpanda-policy-1",
        "decision": decision,
        "reasons": reasons,
    }
    result["record"] = make_record(
        "policy_decision",
        {
            "policy_id": result["policy_id"],
            "decision": decision,
            "reasons": reasons,
            "policy_name": policy["name"],
            "evaluator_version": "trashpanda-policy-1",
        },
    )
    return result


def example_policy(*, strict: bool = False) -> dict[str, Any]:
    """Return a fictional, synthetic-data-only example for docs/tests."""
    return {
        "format": POLICY_FORMAT,
        "version": 1,
        "name": "synthetic-strict-example" if strict else "synthetic-permissive-example",
        "required_evidence": ["input", "preparation", "run"],
        "allowed_issuers": ["fictional.synthetic-reviewer"] if strict else [],
        "require_signed_claims": bool(strict),
        "on_unknown": "deny" if strict else "indeterminate",
        "disallowed_source_ids": [],
        "intended_use": "synthetic-evaluation" if strict else None,
        "notice": "Example only; not a Trashpanda governance recommendation.",
    }


def make_claim(
    *,
    subject: str,
    issuer: str,
    purpose: str,
    restrictions: list[str] | None = None,
    issued_at: str | None = None,
    expires_at: str | None = None,
    evidence_refs: list[str] | None = None,
    status: str = "active",
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Create an immutable adopter assertion; authentication is separate."""
    return make_record(
        "claim",
        {
            "subject": subject,
            "issuer": issuer,
            "purpose": purpose,
            "restrictions": list(restrictions or []),
            "issued_at": issued_at,
            "expires_at": expires_at,
            "evidence_refs": list(evidence_refs or []),
            "status": status,
            "supersedes": supersedes,
        },
    )


def public_projection(record: Mapping[str, Any], allowlist: Iterable[str]) -> dict[str, Any]:
    """Project a record using explicit top-level field allowlisting."""
    allowed = set(allowlist)
    projected = {key: value for key, value in record.items() if key in allowed}
    projected["visibility"] = "public-projection"
    projected["source_record_id"] = record.get("id")
    # A projection is a new record identity, never an edited signed record.
    projected.pop("id", None)
    projected.pop("signature", None)
    return make_record(str(record["type"]), projected)


__all__ = [
    "POLICY_FORMAT",
    "evaluate_policy",
    "example_policy",
    "load_policy",
    "make_claim",
    "policy_identity",
    "public_projection",
    "validate_policy",
]
