# Portable provenance implementation

[Implementation map](README.md) · [Roadmap contract](../roadmap/provenance.md)

Trashpanda now has an offline-first provenance contract. It records mechanisms
and observed facts; it does not decide whether a dataset is legal, safe, or
approved. Adopters supply policy, trusted issuers, keys, required evidence, and
disclosure rules.

## Record graph

The lightweight `trashpanda.provenance` package defines versioned JSON records:

| Record | Captures |
| --- | --- |
| `input` | Stable source ID, acquired content digest, requested/resolved revision, selection, terms/evidence state |
| `preparation` | Tokenizer snapshot/file hashes, packing/EOS settings, split counts, source attribution ranges |
| `execution_segment` | UTC observations, monotonic duration, counters, parent checkpoint digest, software/config identity |
| `run` | Logical run ID, immutable segment references, committed target totals, input references |
| `evaluation` | Actual split, settings, model/input identity, observed counts and metrics |
| `artifact` | Export/checkpoint relationship and evidence state |
| `claim` / `policy_decision` | Adopter-supplied attestation facts and explicit policy outcomes; never implicit approval |

Record IDs are SHA-256 digests of canonical UTF-8 JSON without the ID or
signature. Canonicalization sorts object keys, preserves array order, rejects
duplicate keys and non-finite numbers, and limits integers to interoperable JSON
precision. File hashes stream in bounded chunks. The schema resource is shipped
inside the installed package and is usable without model dependencies.

## Prepared inputs and attribution

`prepare_splits(..., source_descriptors=[...])` accepts one descriptor per local
input. It writes source and preparation records into `manifest.json`, hashes the
actual JSONL bytes before temporary inputs can be deleted, and preserves unknown
revisions/terms explicitly. A source ID supplied by the caller is retained;
without one, the content digest provides a move-independent local identity.

Each split stores compact source ranges for prediction targets. A row of length
`L` contributes `L-1` targets; the first token is never counted. A mixed-source
row attributes each target to the source that supplied that target token, while
context can come from another source. `SequenceReader.target_source_counts()`
maps sampled rows back to these counts.

## Bundle layout and verification

An HF export contains native model/tokenizer files plus:

```text
provenance-manifest.json       # completion manifest; hashes payloads and records
trashpanda-provenance.json     # compatibility pointer to the completion manifest
provenance/records/<sha>.json  # digest-addressed evidence records
provenance-signature.json      # optional detached signature, not a payload hash
```

Payload files and immutable records are written before the completion manifest,
so the manifest never hashes its own final bytes. `verify_bundle()` checks schema
version, safe relative paths, duplicate/missing/extra payloads, sizes, hashes,
record IDs, reference closure, and ancestry cycles without importing torch. It
returns separate results for schema, integrity, completeness, signatures, and
policy. An unsigned but self-consistent bundle is integrity-valid and
unauthenticated; a fully rehashed replacement is still not authenticated.

The CLI is:

```bash
trashpanda provenance inspect EXPORT --json
trashpanda provenance verify EXPORT --json
trashpanda provenance verify EXPORT --policy policy.json --strict --json
```

Verification is local-only by default. Missing parent weights, private terms,
registry records, or newer revocation information remain unavailable rather than
being treated as corruption or approval.

## Policies, signatures, and privacy

Policy files are declarative JSON. With no policy, authorization is `not
evaluated`; with a policy, the result is `allow`, `deny`, or `indeterminate` with
reason codes. Fictional synthetic permissive/strict examples are available from
`trashpanda.provenance.policy.example_policy()` and are not project governance.

Optional Ed25519 detached signatures use the isolated `cryptography` extra:

```python
from trashpanda.provenance.signing import sign_manifest
sign_manifest("export", "private-key.pem", key_id="fictional-synthetic-key")
```

Trust is supplied separately by the verifier; a key shipped beside an artifact
is never trusted automatically. Claims authenticate who asserted a statement,
not whether the statement is independently true or legally sufficient. Public
projections use explicit allowlists and receive new record identities; private
paths, source text, credentials, and contacts are not silently copied.

## Legacy and compatibility

Existing prepared manifests, checkpoints, and exports remain readable through
their original APIs. They are classified as incomplete/legacy when they do not
contain these records. Existing metadata hashes retain their old meaning and
are not relabeled as tokenizer byte hashes. No historical timestamp, source
revision, approval, or trust decision is backfilled.

Normal checkpoints now retain run/session evidence in their primitive payload,
external-weight loading enforces `model_sha256`, and exports use the common
completion-manifest writer. Direct `export_hf()` calls get an explicit
unknown-initialization artifact record instead of silently omitting provenance.

## Sourcelume mapping boundary

The optional exchange adapter targets Apache Sourcelume specification 0.0.1 and
uses an exchange file rather than assuming a stable registry endpoint. Local
record IDs and hashes remain authoritative; unsupported training, policy, and
trust semantics are listed in the mapping result. Preview/publication is an
explicit operation, uses the public projection first, supports idempotency by
local identity, and is never triggered by prepare, train, export, or verify.

The integration decision is documented in
[`docs/implementation/sourcelume.md`](sourcelume.md), including the reviewed
source/version and known lossiness.

For existing artifacts, follow the [migration guide](provenance-migration.md);
it keeps legacy gaps visible instead of fabricating historical evidence.
