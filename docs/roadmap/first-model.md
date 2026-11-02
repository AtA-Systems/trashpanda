# Initial milestone: finish and validate First Model

**Status: partially implemented; production delivery incomplete.**

[Roadmap](README.md) · [Issue #1](https://github.com/rawkintrevo/trashpanda/issues/1)

The initial milestone is a local JSONL → scratch-trained ~300M base model → native
Hugging Face export workflow, using one A100 80 GB. It is a foundation for the
organization-facing product, not the complete Trashpanda vision.

## Existing foundation

The code has preparation, fixed Llama recipes plus a tiny GPT-2 adapter, a token-budgeted optimizer loop,
checkpoint state, evaluation/sampling, and export. Historical A100 records are
bounded benchmarks, not evidence that the full 6B-target run has completed.
See the [implementation map](../implementation/README.md) and
[evidence inventory](../experiments.md).

## Proposed completion sequence

1. Reconcile the code with the authoritative #1/child-issue contracts: immutable
   resume settings and data identity, warmup/decay schedule, repetition warnings,
   periodic bounded evaluation, checkpoint retention, and consistent CLI errors.
2. Make checkpoint delivery self-contained and verifiable: tokenizer files,
   configuration, model, optimizer/schedule, all RNG domains, sampler position,
   data identity, and a completion manifest. Test interrupted copies, corruption,
   restoration at another path, and a real resumed update in a replacement process.
3. Re-run the bounded A100 benchmark and GPU recovery/export checks, recording
   environment, counters, timing scope, and evidence rather than only a summary claim.
4. Execute the actual full configured budget with a tested independent checkpoint
   copy/restore procedure. A detached process is not durable storage. Record repeated
   data exposure separately from unique tokens, and monitor disk/compute limits.
5. Verify final checkpoint and export, publish the actual validation/sampling report,
   and document an authorized durable retrieval location. Public uploads require
   separate permission and rights review.

Do not satisfy a full-budget acceptance criterion with the tiny recipe, a shortened
benchmark, or a fresh optimizer initialized from an exported model. A no-gradient
`optimizer.step()` after loading does not demonstrate resumed training.

## Completion evidence

Require the actual rounded completed-target count, consistent optimizer/schedule
and sampler progress, reproducible evaluation settings, final file hashes, native
HF reload, and an independently exercised recovery path. The current test suite
and historical reports do not establish all of these.

Related issues: [training #5](https://github.com/rawkintrevo/trashpanda/issues/5),
[recovery #6](https://github.com/rawkintrevo/trashpanda/issues/6),
[checks #7](https://github.com/rawkintrevo/trashpanda/issues/7),
[export #8](https://github.com/rawkintrevo/trashpanda/issues/8),
[tests #9](https://github.com/rawkintrevo/trashpanda/issues/9),
[benchmark #10](https://github.com/rawkintrevo/trashpanda/issues/10),
[quickstart #11](https://github.com/rawkintrevo/trashpanda/issues/11),
[delivery #12](https://github.com/rawkintrevo/trashpanda/issues/12).
