# Roadmap and proposed implementations

[Executive summary](../../README.md) · [Current implementation](../implementation/README.md)

These are directions confirmed by the maintainer, with a mix of shipped and
proposed technical designs. They are not a release schedule or approved final
governance protocol.

| Workstream | Starting point | Proposed next step |
| --- | --- | --- |
| [Complete First Model](first-model.md) | Local Llama pipeline and bounded experiment records | Harden recovery/validation and deliver the actual full-budget run with independent durable artifacts |
| [Model-bound provenance](provenance.md) | Versioned records, portable bundles, offline verify, adopter policy/signing, Sourcelume exchange | Extend optional adapters and retain explicit evidence/governance boundaries |
| [Selectable models](model-selection.md) | Two fixed Llama recipes | Add explicit model adapters/selection without abandoning provenance and recovery contracts |
| [Federated training](federated-training.md) | Single-machine training only | Design cross-organization training with local proprietary data, explicit threat model, and validated protections |

## Sequencing and decisions

Recovery and truthful artifact identity are foundations for model-bound provenance.
Additional model adapters should preserve that contract. Federation needs both,
plus separate security/privacy and governance design; keeping data local is not
by itself a confidentiality guarantee.

The remaining proposed designs deliberately list open decisions and acceptance
evidence. Provenance's local record schema and CLI are implemented; registry
publication, adopter governance choices, and federation remain explicitly
optional or proposed rather than silently advertised as project authority.

First Model's exclusions keep that milestone small. They do not define permanent
exclusions for Trashpanda's broader mission.
