# Proposal: federated training across organizations

**Status: local synthetic CPU prototype implemented; cross-organization federation remains proposed.**

[Roadmap](README.md) · [Provenance proposal](provenance.md)

## Outcome

Participating organizations train a shared model while keeping proprietary raw
training data within their own environments. Each participant needs evidence of
what process ran and what the resulting model contains, subject to agreed
confidentiality boundaries.

This is different from ordinary multi-GPU/distributed training under one operator.
It introduces separate owners, trust boundaries, policies, and failure modes.

## Delivered local prototype

The repository now includes a first-party, offline runner for two participant
processes on one machine. It uses the tiny Llama recipe, synthetic prepared data,
immutable JSON/safetensors round bundles, bounded local AdamW updates, weighted
full-model averaging, crash-resumable private state, held-out metrics and a
portable public provenance projection. See the [implementation guide](../implementation/federation.md)
and [quickstart](../getting-started-federation.md).

This fixture exposes individual updates to the coordinator, uses participant IDs
as labels rather than cryptographic identities, and is not an OS security sandbox.
It must not be used as evidence of privacy, convergence, certification or
proprietary-data readiness.

## Proposed next architecture

1. Agree a model/tokenizer identity, protocol version, participant authorization,
   training policy, permitted update format, and disclosure rules.
2. Keep approved data and its detailed records local to each participant. Run local
   training against a verified round checkpoint; emit bounded updates and permitted
   provenance/approval claims, not raw records.
3. Use a coordinator or agreed aggregation service to combine compatible updates.
   Evaluate cross-machine transport and existing federated-learning frameworks
   before expanding the first-party protocol. The local round-based averaging
   experiment is a bounded research baseline, not a claim of convergent
   large-language-model training.
4. Bind the resulting model to round IDs, parent checkpoints, aggregate counters,
   participant attestations, and the limits of disclosed dataset lineage.
5. Define recovery and audit semantics for participant dropout, failed rounds,
   duplicate/replayed updates, revoked participants, and model rollback.

## Privacy and threat model come first

**Not sharing raw data does not guarantee privacy.** Gradients, model updates,
metadata, and released weights can leak information. A design review must state
what the coordinator and participants can observe and whether collusion, malicious
clients, poisoning, and membership/reconstruction attacks are in scope.

Assess secure aggregation, update clipping, differential privacy and privacy-budget
accounting, authentication, access control, and transport protection against that
threat model. These mechanisms have different guarantees and utility costs; none
is currently implemented or promised by Trashpanda. An experiment without approved
protections must not accept real proprietary data or claim that those data cannot leak.

Provenance must support explicit partial disclosure: a participant's attestation
is not the coordinator independently inspecting its corpus. Document exactly what
a receiving organization can and cannot verify.

## Decisions still needed

- Trusted coordinator, consortium governance, or another trust arrangement?
- Which algorithms and frameworks fit the target models and network constraints?
- Required privacy guarantees, adversaries, utility limits, and audit obligations?
- What happens when a participant or its approved data must be withdrawn?
- Which evidence can be shared without exposing sensitive data or business metadata?

## Acceptance evidence

Start with synthetic corpora in at least two isolated participant environments.
Verify raw records do not cross the declared boundary, reject incompatible or
replayed updates, recover an interrupted round without double-counting, and retain
truthful aggregate provenance. Evaluate learning behavior and leakage under the
chosen threat model. Require security/privacy review before proprietary-data pilots.
