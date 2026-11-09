# Trashpanda

**Open-weight models with an accountable supply chain.**

Trashpanda is being built for organizations that need models trained on known,
organization-approved data—and evidence of what went into them. The goal is a
practical workflow for selecting a model, training it, and delivering weights
with inspectable provenance, not another PyTorch or Hugging Face replacement.

**Today:** an experimental single-machine causal-LM training foundation with
CLI-selectable tiny Llama and GPT-2 architectures, plus an offline-first,
inspectable provenance bundle and a local, synthetic-data federated training
prototype. Provenance records are evidence
mechanisms, not a delivered certification, legal conclusion, safety verdict, or
privacy guarantee. [Read the vision and boundaries →](docs/vision.md)

## What is implemented

| Capability | Current scope and implementation details |
| --- | --- |
| Local data preparation | Separate train/validation JSONL → token shards, counts, and integrity hashes. [How data flows](docs/implementation/data.md) |
| Training and checkpoint state | Tiny Llama and GPT-2 recipes selected with `--model`, token-budgeted PyTorch training, model/optimizer/RNG/sampler saves and resume support. [Supported models](docs/models.md) · [How training and recovery work—and their limits](docs/implementation/training.md) |
| CLI-selectable models | `--model llama` and `--model gpt2` use explicit native Hugging Face adapters; architecture identity is retained through checkpoints, resume, evaluation, and export. [Supported models](docs/models.md) |
| Evaluation and model export | Loss/perplexity, sampling, and native Hugging Face model/tokenizer files. [How evaluation and export work](docs/implementation/export.md) |
| Provenance-carrying checkpoints/exports | Versioned input/preparation/run/evaluation/artifact records, canonical identities, payload hashes, offline verification, and native HF export bundles. Adopters still provide policy/trust and interpret claims. [Implementation contract](docs/implementation/provenance.md) |
| Local federated training prototype | Two local CPU participant processes, bounded round-local training, deterministic weighted averaging, crash-resumable file exchange, held-out evaluation, and portable round provenance using synthetic data. This is a research fixture, not a privacy guarantee or cross-machine service. [Federation implementation](docs/implementation/federation.md) · [Quickstart](docs/getting-started-federation.md) |

[First Model (#1)](https://github.com/rawkintrevo/trashpanda/issues/1) is the initial
milestone, not the whole project. Historical A100 records describe bounded
experiments; **the configured 6B-target training and final-artifact delivery are
not established as complete**. [Evidence and artifact availability →](docs/experiments.md)

## What is next

| Direction | Proposed implementation |
| --- | --- |
| Optional provenance integrations | Extend the implemented offline contract with additional standards mappings and adopter-controlled registry publication. [Integration boundary](docs/implementation/sourcelume.md) |
| More model architectures | Add explicit adapters for additional model families while preserving evidence and recovery contracts. [Supported models and roadmap](docs/models.md) · [Roadmap](docs/roadmap/model-selection.md) |
| Federated training next steps | Extend the local fixture to cross-machine/cross-organization transport, authenticated participation, threat-model validation, secure aggregation/differential privacy, and larger-scale evaluation. These remain proposed capabilities. [Roadmap](docs/roadmap/federated-training.md) |

Additional model families and cross-organization/privacy-protected federation remain proposed capabilities.
The provenance CLI is a working local feature; it does not publish implicitly.
The [roadmap](docs/roadmap/README.md) tracks remaining first-model work.

## Where Trashpanda fits

- **Apache Sourcelume Sample** demonstrates training-data provenance records and
  registry exchange. Trashpanda includes an optional, explicitly invoked
  Sourcelume exchange adapter; local provenance and verification remain
  independent of any registry.
- **OLMo / OLMo-core** provide open models and research training infrastructure.
  Trashpanda's intended focus is the organization's approved inputs and evidence
  for *its* model, rather than another model family or general training framework.

[Detailed comparison, current gaps, and upstream sources →](docs/comparisons.md)

## Start here

- [Google Colab walkthrough](examples/trashpanda_colab.ipynb) · [Open in Colab](https://colab.research.google.com/github/rawkintrevo/trashpanda/blob/main/examples/trashpanda_colab.ipynb):
  select Llama or GPT-2, persist data and checkpoints in Google Drive, resume across
  sessions, and inspect/export a provenance-carrying model.
- [CPU quickstart](docs/getting-started.md): exercise preparation, training, resume,
  evaluation, and offline export without a GPU or downloaded training data.
- [Offline provenance fixture](docs/getting-started.md#run-the-offline-provenance-fixture):
  two synthetic sources, a real 100-parameter CPU model, resume evidence, and
  relocatable verification.
- [Local federated training quickstart](docs/getting-started-federation.md): run two
  synthetic participants, resume a local round, evaluate and export offline.
- [Implementation map](docs/implementation/README.md): source modules and current limitations.
- [Testing and CI](docs/testing.md): run the bounded feature acceptance suite and inspect CI/release validation evidence.
- [Data and compute examples](docs/experiments.md): optional FineWeb-Edu and A100/Colab workflows.

Project code is [Apache-2.0 licensed](LICENSE). Data, tokenizer, and weight rights
require separate review. “Approved data” means approved under an organization's
policy—not automatically safe, legally cleared, or certified by Trashpanda.

For contributions, start with the [open issues](https://github.com/rawkintrevo/trashpanda/issues)
and distinguish fixes to current behavior from changes to the proposed designs.
