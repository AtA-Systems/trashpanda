# Roadmap and proposed implementations

[Executive summary](../../README.md) · [Current implementation](../implementation/README.md)

These are directions confirmed by the maintainer, with a mix of shipped and
proposed technical designs. They are not a release schedule or approved final
governance protocol.

| Workstream | Starting point | Proposed next step |
| --- | --- | --- |
| [Complete First Model](first-model.md) | Local Llama pipeline and bounded experiment records | Harden recovery/validation and deliver the actual full-budget run with independent durable artifacts |
| [Model-bound provenance](provenance.md) | Versioned records, portable bundles, offline verify, adopter policy/signing, Sourcelume exchange | Extend optional adapters and retain explicit evidence/governance boundaries |
| [Selectable models](model-selection.md) | Tiny Llama and GPT-2 selected with `--model`; [catalogue](../models.md) | Add more explicit model adapters without abandoning provenance and recovery contracts |
| [Federated training](federated-training.md) | Local synthetic CPU multi-process prototype | Extend to cross-organization transport, authentication and validated privacy protections |

## Sequencing and decisions

Recovery and truthful artifact identity are foundations for model-bound provenance.
Additional model adapters should preserve that contract. Federation needs both,
plus separate security/privacy and governance design; keeping data local is not
by itself a confidentiality guarantee.

The remaining proposed designs deliberately list open decisions and acceptance
evidence. Provenance's local record schema and CLI, plus the bounded local
federation fixture, are implemented; registry publication, adopter governance,
cross-organization transport and privacy guarantees remain proposed rather than
silently advertised as project authority.

First Model's exclusions keep that milestone small. They do not define permanent
exclusions for Trashpanda's broader mission.
