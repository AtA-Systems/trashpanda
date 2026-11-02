# Current data preparation

[Implementation map](README.md) · [Partial provenance](provenance.md)

## Local contract

The core CLI consumes local JSONL files, not a mandatory hosted dataset. Supply
separate train and validation paths. Each nonblank line must be a JSON object
containing a nonempty string `text` (or the selected text field). Identical input
paths across splits are rejected; duplicated content is not detected.

[`prepare_splits()`](../../src/trashpanda/prepare.py) encodes each document without
implicit special tokens, explicitly appends one EOS, and packs the token stream
into complete context-length sequences. The incomplete final tail is counted and
dropped separately for each split. It does not clean, deduplicate, classify,
license-check, approve, or anonymize the data. Whitespace-only strings are preserved.

It writes little-endian uint16 shards, a tokenizer directory, and `manifest.json`.
The manifest records token/sequence counts, dropped tails, context/vocabulary,
tokenizer metadata, saved tokenizer-file hashes, shard hashes, an identity hash,
and versioned input/preparation records. Pass `source_descriptors` to retain a
stable source ID, actual acquired JSONL digest, requested/resolved revision,
selection, terms, and explicit unknown evidence states. No absolute path or
source text is required for the logical identity.

The CLI accepts the same fields through one JSON file per input:

```json
{"source_id":"synthetic:A","source":"project-owned-fixture","source_revision":"fixture-1","selection":{"kind":"all-records"},"terms":"synthetic-only","evidence_state":"observed"}
```

Pass them in input order with repeated `--source-descriptor` options. Omitting a
descriptor records a content-derived local ID and explicit unknown metadata.

For mixed local inputs, preparation also writes compact prediction-target ranges.
`SequenceReader.target_source_counts(indices)` returns per-source counts for the
sampled rows. Counts use the source of each target token, exclude each sequence's
first context token, include repeated sampling only when the caller samples a row
again, and keep dropped tails separate.

[`SequenceReader`](../../src/trashpanda/data.py) checks shard sizes and SHA-256s by
default and memory-maps sequences. A length-L row yields `row[:-1]` inputs and
`row[1:]` targets: **L−1 prediction targets**, not L. Sequences can cross document
boundaries; no padding is introduced.

## Tokenizers and recipes

- `tiny`: project-owned 32-entry WordLevel tokenizer, generated locally; 32-token
  sequences. Preparation still appends EOS even though the tokenizer adds none.
- `300m-a100`: 32K Mistral tokenizer, revision
  `27d67f1b5f57dc0953326b2601d68371d40ea8da`; 2048-token sequences. Loading requests
  no remote code. A reused tokenizer does not mean pretrained model weights are loaded.

The code records hashes of locally saved tokenizer files. Production acquisition
is restricted to the three pinned JSON assets and verifies their upstream raw-file
hash constants before loading. The CLI uses the recipe's pinned revision; a
production revision different from the contract is rejected. Prepared normalized
files are not expected to have the raw Hub hashes.

## Hugging Face example, not the product boundary

[`prepare_fineweb_edu.py`](../../scripts/prepare_fineweb_edu.py) streams
`HuggingFaceFW/fineweb-edu`, config `sample-10BT`, into temporary JSONL and calls
the same preparation path. Defaults select the first 30,000 training documents
and the next 1,000 validation documents. This is a bounded experiment, not a data
approval decision or a representative production corpus guarantee.

Pass `--revision` to pin a requested immutable dataset revision. The adapter
persists both requested and resolved values and records `unknown` when the
streaming backend cannot expose a resolved revision. The acquired temporary
JSONL bytes are still hashed before deletion. A Hugging Face listing alone does
not establish rights to every underlying record; use your own review process.

## Current limitations

Preparation verifies manifest, shard, and tokenizer-file identities on read.
Terms, approvals, and acquisition times remain adopter-supplied observations;
unknown values are not fabricated. The split-wide compatibility digest remains
for existing callers, while per-file provenance hashes stream in bounded chunks.

See the [provenance proposal](../roadmap/provenance.md) for the intended extension.
