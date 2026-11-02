# Sourcelume interoperability decision

Reviewed 2026-11-02 against the Apache Sourcelume specification repository and
its versioned 0.0.1 record documentation:

- [specification repository](https://apache.googlesource.com/sourcelume-spec/)
- [0.0.1 record fields](https://apache.googlesource.com/sourcelume-spec/+/refs/heads/common-spec-to-main/spec/0.0.1/index.md)
- [sample end-to-end project](https://apache.googlesource.com/sourcelume-sample/)

The published shape is a graph-oriented JSON-LD dataset `ProvenanceRecord` with
creator/origin/addition/distribution and licensing-oriented fields. It is not a
complete model-run, authorization, or signature-trust contract for Trashpanda.
The exact upstream version is therefore pinned to `0.0.1` in
`trashpanda.provenance.exchange`; moving to another upstream version requires a
new mapping decision.

| Local record | Sourcelume 0.0.1 mapping | Treatment |
| --- | --- | --- |
| `input.source_id` | `origin` / `@id` | Preserved under local extension; foreign ID never replaces local digest |
| `input.content_sha256` | `distribution[].sha256` | Direct mapping |
| `input.source_revision`, `selection` | No single equivalent | Preserved in `trashpanda.records`; reported lossy |
| `input.terms` | licensing fields | Preserved locally unless adopter supplies a reviewed SPDX/license mapping |
| preparation/run/counts | No stable equivalent in reviewed 0.0.1 dataset record | Preserved locally; unsupported for direct native mapping |
| claims/signatures/policy | No local authorization equivalent | Never flattened to approval or dropped silently |

The exchange adapter creates a versioned local exchange file and reports
`lossy_fields` and `unsupported_fields`. Publication needs an adopter endpoint,
credentials, and an explicit command; no core workflow makes network requests.
