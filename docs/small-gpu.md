# Training on a small local NVIDIA GPU

[Executive summary](../README.md) · [CPU quickstart](getting-started.md) · [Supported models](models.md)

Two recipes target a single consumer GPU. Both are randomly initialized Llama
models that use the pinned 32k Mistral tokenizer and a 512-token context, so one
prepared dataset serves both.

| Recipe | Parameters | Sized for | Batch / microbatch | Requested targets | Checkpoint interval |
| --- | --- | --- | --- | --- | --- |
| `22m-gtx1060` | 22,467,456 | one 6 GB card | 32 / 8 | 100,000,000 | 1,000 updates |
| `40m-titanx` | 40,509,952 | one 12 GB card | 32 / 8 | 200,000,000 | 500 updates |

These recipes pass the CPU test suite and a one-step CPU smoke test. **They have
not been validated on GPU hardware**; memory headroom and throughput are for you to
measure. Both use FP32 because pre-Turing cards (GTX 900/1000 series, Titan X) have
no fast half precision.

## Install

Pre-Turing cards (Maxwell, Pascal, Volta) need NVIDIA's 580 driver branch and a
PyTorch build that still targets them. PyTorch 2.14 is the last release with a
prebuilt wheel for these architectures, from the CUDA 12.6 index:

```bash
nvidia-smi   # must list the card
python3 -m venv .venv && . .venv/bin/activate
python -m pip install "torch<2.15" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[test,datasets]'
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Turing and newer cards (RTX 20 series onward) can use the default PyTorch install.

## Prepare data

```bash
python scripts/prepare_fineweb_edu.py prepared \
  --recipe 22m-gtx1060 --allow-download
```

This downloads the three pinned tokenizer files and streams a bounded FineWeb-Edu
sample. `--recipe` sets the context length the shards are cut to; training refuses
a dataset whose context length differs from the model's. See
[experiments](experiments.md#optional-fineweb-edu-preparation) for the caveats on
dataset approval and repetition.

## Train, resume, evaluate

Start with a bounded run to confirm the install and read the throughput:

```bash
python -m trashpanda train --recipe 22m-gtx1060 \
  --data prepared --output runs/22m --max-steps 20
```

`runs/22m/metrics.jsonl` records targets and elapsed seconds per update. Then run
without `--max-steps` for the full budget. Ctrl-C stops at the next update boundary
and writes a checkpoint. To continue:

```bash
python -m trashpanda train --data prepared --output runs/22m \
  --resume runs/22m/checkpoint-00001000.pt
```

```bash
python -m trashpanda evaluate --checkpoint runs/22m/checkpoint-00001000.pt \
  --data prepared --device cuda --max-batches 50
python -m trashpanda sample --checkpoint runs/22m/checkpoint-00001000.pt \
  --prompt 'The water cycle' --max-new-tokens 64 --device cuda
python -m trashpanda export --checkpoint runs/22m/checkpoint-00001000.pt \
  --output runs/22m-export
```

### Tuning

- **Out of memory:** lower `--microbatch-size` (it must divide the batch size of
  32). The effective batch and the schedule are unchanged; only speed drops.
- **Spare memory:** raise `--microbatch-size` to 16 or 32 for fewer, larger steps.
- **Shorter or longer run:** `--tokens N` sets the requested target budget.
- **Disk:** there is no checkpoint retention policy. Each checkpoint holds the model
  and optimizer state (roughly three times the FP32 model size), so delete old ones
  during long runs.

## Two or more GPUs in one machine

### Split each update across cards: `--gpus`

```bash
python -m trashpanda train --recipe 40m-titanx \
  --data prepared --output runs/40m --gpus 2
```

An update is already a sum of microbatch gradients (`batch_size / microbatch_size`
of them; four in these recipes). With `--gpus N`, those microbatches run at the same
time on per-card replicas of the model, and their gradients are added on the first
card before clipping and the optimizer step. Source:
[`parallel.py`](../src/trashpanda/parallel.py).

What stays the same as a one-card run: the sampler order, which sequences are in
each update, token accounting, the optimizer and schedule, and the checkpoint
format. A checkpoint written with `--gpus 2` resumes without the flag, and the
reverse. Trained weights match a one-card run up to floating-point summation order,
not bit for bit.

Things to know:

- **Validation status.** The logic is tested on CPU by emulating several devices,
  including equivalence with one device and resume in both directions. It has
  **not been run on real multi-GPU hardware** by the author of this guide. On a
  machine with two visible cards, `python -m pytest tests/test_multi_device.py`
  also runs a CUDA comparison that is skipped elsewhere. Run it before a long job.
- **Which cards.** `--gpus N` uses `cuda:0` to `cuda:N-1`. Use
  `CUDA_VISIBLE_DEVICES` to choose or reorder cards.
- **Microbatches per update.** There must be at least `N`, and a multiple of `N`
  keeps every card equally busy. Microbatch `i` goes to card `i % N`.
- **Memory.** Every card holds the model, its gradients, and one microbatch of
  activations. The first card also holds the optimizer state.
- **Speed.** Expect less than `N` times one card: parameters are copied out and
  gradients copied back once per update, and each update waits for the slowest
  card. Measure with `metrics.jsonl` as above.
- **Precision.** FP32 and BF16 recipes are supported; FP16 with more than one
  device is rejected.
- **Provenance.** Multi-device execution segments record a `devices` list.
  Single-device records are unchanged.

There is no multi-process (DDP) or multi-machine path.

### Or run two independent experiments

`CUDA_VISIBLE_DEVICES` pins a process to one card, and both runs can read the same
prepared directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m trashpanda train --recipe 40m-titanx \
  --data prepared --output runs/40m-a

# in a second terminal
echo '{"learning_rate": 0.0006}' > lr-high.json
CUDA_VISIBLE_DEVICES=1 python -m trashpanda train --recipe 40m-titanx \
  --data prepared --output runs/40m-b --config lr-high.json
```

`--config` accepts only `token_budget`, `microbatch_size`, `learning_rate`, and
`repeat_data`. Give each run its own `--output`.
