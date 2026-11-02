# Current training and recovery

[Implementation map](README.md) · [Executable tiny workflow](../getting-started.md)

## Model and update loop

Both recipes construct Transformers' `LlamaForCausalLM` from random initialization,
with tied embeddings and SDPA. They do not load a pretrained model. `tiny` and
`300m-a100` are fixed configurations of the same architecture, not different model
families. Source: [`model.py`](../../src/trashpanda/model.py),
[`config.py`](../../src/trashpanda/config.py).

| Setting | tiny | 300m-a100 |
| --- | --- | --- |
| Vocabulary / layers / hidden / intermediate | 32 / 2 / 32 / 64 | 32000 / 22 / 1024 / 2560 |
| Context / effective batch / microbatch | 32 / 4 / 2 | 2048 / 128 / 8 |
| Device / precision | CPU / FP32 | CUDA / BF16 autocast |
| Requested prediction targets | 744 | 6,000,000,000 |
| Checkpoint interval | 3 updates | 500 updates |

The production architecture has 298,103,808 unique parameters. The loop uses
AdamW, gradient accumulation, global norm clipping, and nonfinite loss/gradient
checks. Parameters remain FP32 in normal recipe construction. The optimizer uses
checkpointed linear warmup followed by cosine decay to the configured minimum
learning-rate ratio.

Targets per update are `batch × (context − 1)`; budgets round up to full updates.
`--max-steps` is an absolute update ceiling, not an extra-step count. PCG64 sampling
repeats data by default. The loop does not yet provide the prominent repetition
warnings specified in #1. A 6B target on a small prepared corpus means repeated
passes—not 6B unique source tokens.

`--config` accepts only `token_budget`, `microbatch_size`, `learning_rate`, and
`repeat_data`. See [`config.py`](../../src/trashpanda/config.py) for constraints.
Current CLI defaults to `tiny`; select the production recipe explicitly.

## What a normal checkpoint stores

[`save_checkpoint()`](../../src/trashpanda/checkpoint.py) saves a
`trashpanda.checkpoint.v1` `.pt` file with:

- model tensors and AdamW optimizer state;
- completed update/target/microstep counters;
- the complete resolved configuration and tokenizer metadata plus tokenizer files;
- PCG64 generator state, permutation, epoch, and cursor;
- Python, NumPy, Torch CPU and available CUDA RNG state.

The normal writer uses a temporary file, flush/fsync, and same-filesystem replace.
SIGINT/SIGTERM handlers request a stop at an optimizer boundary; abrupt machine
loss cannot save work still in memory. Resume loads the above state. Only load
locally trusted checkpoints; `weights_only=True` is not provenance certification.

An additional bundle helper can reference external safetensors instead of storing
model tensors in the `.pt` file. It is not the normal CLI output and its loader does
not verify the optional recorded model digest; do not treat it as a verified
self-contained delivery contract.

## Recovery is partial, not a machine-loss guarantee

Resume rejects changes to the saved model/training configuration and checks the
prepared manifest identity (including shard and tokenizer file hashes). Sampler
restoration also checks dataset size. This does not prove source rights or external
dataset lineage. Keep an independent verified inventory and test a real resumed
forward/backward/update in the replacement environment.

Keep checkpoints, prepared shards/manifest, tokenizer assets, config/commands,
and logs together. Copy a completed checkpoint to storage independent of the
training machine, verify hashes, then test on a replacement machine with a **real
forward/backward/update**, not just successful deserialization. Export alone lacks
optimizer/RNG/sampler state. A local workspace path or Git ignore rule is not a
backup guarantee.

The [quickstart](../getting-started.md) shows the current CLI's explicit `--recipe`,
`--data`, `--output`, and checkpoint-file `--resume` arguments. The production driver
has no resume flag; use the core CLI for a continuation of an interrupted run.

Additional gaps: no automatic checkpoint retention policy, detached signing is
optional and adopter-configured, and metrics append on resume is still a
diagnostic log rather than recovery authority. Provenance records in the
checkpoint preserve committed state and parent digests, but an abrupt process
death can leave a running segment with an unknown end time. Do not launch a
long production run without planning disk capacity and independent checkpoint copies. See the
[first-model completion proposal](../roadmap/first-model.md).
