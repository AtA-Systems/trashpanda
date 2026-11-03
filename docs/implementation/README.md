# Current implementation

[Executive summary](../../README.md) · [Proposed capabilities](../roadmap/README.md)

This describes the checked-in code, not every acceptance criterion in the First
Model issues. Implementation exists; completing the production run and providing
organizational supply-chain assurances are separate claims.

| Boundary | Source | Details |
| --- | --- | --- |
| CLI and recipes | [`cli.py`](../../src/trashpanda/cli.py), [`config.py`](../../src/trashpanda/config.py) | [Quickstart](../getting-started.md), [training](training.md) |
| Tokenizer and input preparation | [`tokenizer.py`](../../src/trashpanda/tokenizer.py), [`prepare.py`](../../src/trashpanda/prepare.py) | [Data](data.md) |
| Dataset reader and sampler | [`data.py`](../../src/trashpanda/data.py) | [Data](data.md), [recovery](training.md) |
| Model and training loop | [`model.py`](../../src/trashpanda/model.py), [`train.py`](../../src/trashpanda/train.py) | [Training](training.md) |
| State persistence | [`checkpoint.py`](../../src/trashpanda/checkpoint.py) | [Recovery and limitations](training.md) |
| Evaluation and native HF files | [`evaluate.py`](../../src/trashpanda/evaluate.py), [`export.py`](../../src/trashpanda/export.py) | [Evaluation/export](export.md) |
| Metadata across these boundaries | `trashpanda.provenance`, manifests, checkpoint payloads, metrics | [Portable provenance](provenance.md) |
| Local federation protocol | `trashpanda.federation.config`, `bundles`, `participant`, `aggregate`, `coordinator`, `runner` | [Federation implementation](federation.md), [Federation quickstart](../getting-started-federation.md) |

## Validation boundaries

The [tests](../../tests) cover recipe/config validation, tiny model construction,
preparation, token accounting, both model lifecycles, exact CPU split-run
recovery, canonical provenance records, tamper detection, policy comparison,
portable native exports, federation aggregation and invalid contributions,
coordinator crash recovery, multi-process participant recovery, held-out
metrics, baselines and public export evidence. The complete commands and CI
boundaries are documented in [Testing and CI](../testing.md). These tests are
not evidence of legal compliance, independent factual assurance, federated
privacy, or public registry publication.

Documentation of these gaps is not a fix for them. The [first-model completion
plan](../roadmap/first-model.md) identifies the next validation and recovery work.
