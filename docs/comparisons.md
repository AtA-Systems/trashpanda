# How Trashpanda relates to Sourcelume and OLMo

[Executive summary](../README.md) · [Vision](vision.md)

These projects overlap in useful ways. Trashpanda's differentiation is its
**intended organizational workflow**, not a claim that it already delivers more
provenance, stronger privacy, or better models than either project.

## Apache Sourcelume Sample

[Apache Sourcelume Sample](https://github.com/apache/sourcelume-sample) demonstrates
an end-to-end provenance loop. Its tiny, NumPy-only character-level model trains
on *Alice's Adventures in Wonderland*. It creates a `TrainingProvenanceRecord`
from actual training-run counts, references a dataset record, validates records,
and demonstrates publication and queries using Sourcelume tooling and the registry.

This is more than a model with a handwritten dataset name attached. The sample's
README explicitly distinguishes the producer's unsigned, self-asserted claims
from certification. Its registry example resolves referenced dataset records'
current status at query time; that status is not an independent legal approval.

**Trashpanda's intended distinction:** an organization-facing path for producing
usable models on organization-approved data, selecting supported model recipes,
retaining auditable evidence, and eventually collaborating across organizations.
First Model is the initial training milestone, not that whole product.

**Where we are today:** Trashpanda has local shard hashes, tokenizer metadata,
configuration, and checkpoint state, but no Sourcelume record generation,
registry integration, or signed model-bound provenance. Sourcelume Sample already
demonstrates a provenance exchange workflow that Trashpanda does not yet have.
The [provenance proposal](roadmap/provenance.md) considers compatibility with
Sourcelume rather than inventing a competing standard or claiming integration exists.

## OLMo (Ai2)

[OLMo](https://github.com/allenai/OLMo) is an open language-model family and research
training ecosystem. Its published materials include weights, training
configurations, data references, and checkpoints. The original repository directs
new work to [OLMo-core](https://github.com/allenai/OLMo-core), which provides modeling
and training building blocks and official training scripts.

**Trashpanda's intended distinction:** helping a particular organization answer
“which inputs were approved for *our* model, what actually happened in this run,
and what evidence travels with the resulting weights?” This complements open
research releases rather than dismissing their transparency. OLMo's openness is
valuable; whether a specific release satisfies an organization's policy still
requires review of that release and its inputs.

**Where we are today:** Trashpanda constructs Transformers Llama and tiny GPT-2
models; it neither uses OLMo as a dependency nor trains an OLMo recipe. OLMo could be evaluated as a
future [model-selection](roadmap/model-selection.md) candidate, subject to
architecture, tokenizer, licensing, export, and recovery validation. That is a
proposal, not a supported CLI option or a promise to adopt OLMo-core.

## At a glance

| Project | Primary role described by its documentation | Relationship to Trashpanda |
| --- | --- | --- |
| Sourcelume Sample | Small executable demonstration of training-data provenance records and registry exchange | Reference for provenance integration; not something to dismiss as “just a toy model” |
| OLMo / OLMo-core | Open model releases and modeling/training infrastructure | Relevant model ecosystem and transparency reference; not an implemented Trashpanda backend |
| Trashpanda | Currently a small Llama/GPT-2 training pipeline; intended to become an organizational model-production and supply-chain assurance workflow | Reuse frameworks and standards; implement the organization-specific evidence and approval workflow |

## Sources and scope

Reviewed 2026-10-30. These comparisons concern the projects' documented scope,
not an independent security, legal, privacy, or performance audit. Upstream
capabilities may change; check the linked version before selecting an integration.

- [Sourcelume Sample README](https://github.com/apache/sourcelume-sample#readme)
- [Sourcelume specification](https://github.com/apache/sourcelume-spec)
- [Sourcelume Registry](https://github.com/apache/sourcelume-registry)
- [OLMo README, reviewed revision](https://github.com/allenai/OLMo/blob/090253dac6688f2532509daa7aa2eb5fae50e956/README.md)
- [OLMo-core README, reviewed revision](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/README.md)
