# Vision: model supply-chain assurance

[Executive summary](../README.md) · [Roadmap](roadmap/README.md)

## Who Trashpanda is for

Organizations that need an open-weight model built from data they can identify,
review, and approve—not just a downloadable set of weights. The intended users
include the engineers building models and the security, legal, procurement, and
governance teams deciding whether those models can be used.

The question is not only **“Can we train this model?”** It is also **“Can we
explain what went into this particular model, how it was produced, and which
organization accepted the evidence?”**

## The intended workflow

1. Identify candidate data and its sources, versions, usage terms, and restrictions.
2. Apply the organization's review and approval process.
3. Select a supported model and run a reproducible training workflow on approved inputs.
4. Retain training state and evidence through interruptions and machine replacement.
5. Deliver open-weight model artifacts with a portable record of their lineage.
6. Eventually support participating organizations training together while keeping
   proprietary source data within their own environments.

This is the **direction**, not a claim that all six steps exist today. The
[first milestone](roadmap/first-model.md) establishes the training path. The
[implementation documentation](implementation/README.md) describes the current
code; the remaining mechanisms have [proposal write-ups](roadmap/README.md).

## Not another PyTorch or Hugging Face

Trashpanda should build on training frameworks, model implementations, tokenizer
libraries, and interchange formats rather than replace them. Its intended value
is the organization-facing workflow connecting approved inputs, model production,
and portable evidence. Today it uses PyTorch and two explicit Transformers
causal-LM adapters: Llama and tiny GPT-2.

Likewise, it is not intended to replace a provenance standard or a registry.
[Apache Sourcelume and OLMo](comparisons.md) address related concerns at different
levels. Integration or reuse should be considered before building equivalent
infrastructure.

## What “known safe data” means here

“Safe” is an organization's policy decision with evidence and limits, not a
property Trashpanda can infer from a dataset name or a permissive code license.
A useful record should say who approved which data for which purpose, under what
terms, and when. It must distinguish:

- **Integrity:** are these the exact bytes referenced by the record?
- **Lineage:** which data, code, configuration, tokenizer, and parent weights were used?
- **Authorization:** what evidence supports the claimed permission to use those inputs?
- **Assurance:** who made or verified each claim, and what remains unknown?

A content hash establishes identity, not consent, license compliance, absence of
personal information, or model safety. A signed claim identifies an issuer; it
does not make the claim true. Validation must preserve those distinctions.

Starting from third-party weights also inherits their upstream training history.
Fine-tuning on approved data does **not** erase unknown pretraining data. Future
support for pretrained initialization must disclose that ancestry, rather than
presenting the resulting model as trained solely on the organization's own data.

## Boundaries

The current project is an experimental foundation, not an enterprise assurance
product or a certification authority. No Apache Software Foundation affiliation
or endorsement is claimed. The [Apache-2.0 license](../LICENSE) covers project
code; it does not settle rights to datasets, tokenizers, or generated weights.
