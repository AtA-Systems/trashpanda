# Migrating existing artifacts

This guide preserves historical evidence instead of upgrading it by assertion.

## Prepared directories

Existing `trashpanda.shards.v2` manifests continue to load. If they were
created before source descriptors, their source identity is unavailable; do not
infer it from a filename or a current dataset label. Re-run preparation from
the retained JSONL inputs with `source_descriptors` when those inputs still
exist. Otherwise retain the old manifest as a legacy input record and state
that the source revision, terms, and acquisition time are unknown.

## Checkpoints and exports

Existing `.pt` checkpoints continue to load through `load_checkpoint()`. Their
old `data_identity`, tokenizer snapshot, and metadata hashes retain their
original meanings. They do not become complete run records merely because they
are copied into a new directory. When exporting one, the new export records the
source checkpoint digest and an explicit incomplete/unknown-initialization
relationship; it does not invent UTC dates, approvals, signatures, or source
rights.

Legacy exports with only `trashpanda-provenance.json` can still be inspected. A
verification result reports the missing completion manifest and evidence rather
than treating the artifact as corrupt or approved. Keep historical reports and
curated metadata in their original files and cite their source in any new
claim.

## Historical reports

Reports under `reports/` and retained artifact directories are observations from
their original workflows. Copy them into a new claim only with a source
reference, the original timestamp semantics, and an explicit evidence state.
Never backfill a dataset revision, legal approval, or independent verification
from a label in a report.
