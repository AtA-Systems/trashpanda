# CPU quickstart

[Executive summary](../README.md) · [Implementation details](implementation/README.md)

This exercises the plumbing with a generated 32-token vocabulary. It is not a
useful language model, production training, or evidence that the assurance roadmap
is implemented. No Colab account, GPU, Hugging Face account, or downloaded training
data is needed. Installation requires package downloads; the fixture run is offline.

## Install from a source checkout

Use Python 3.11+ and a virtual environment. For a CPU-only PyTorch installation
on a supported platform:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

For other platforms, use the appropriate [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
The project's current dependency ranges are in [`pyproject.toml`](../pyproject.toml);
this is not a locked or security-certified environment.

## Prepare, train, and resume

Run these blocks in the same shell from the repository root. `mktemp` gives a fresh
output directory; do not reuse an existing prepared/export directory.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DEMO="$(mktemp -d)"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DEMO"])
for name, documents, words, offset in (
    ("train", 8, 128, 0),
    ("validation", 4, 96, 13),
):
    with (root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
        for document in range(documents):
            text = " ".join(f"w{(i + document + offset) % 29}" for i in range(words))
            handle.write(json.dumps({"text": text}) + "\n")
print(root)
PY

python -m trashpanda prepare --recipe tiny \
  --train "$DEMO/train.jsonl" --validation "$DEMO/validation.jsonl" \
  --output "$DEMO/prepared"

python -m trashpanda train --recipe tiny \
  --data "$DEMO/prepared" --output "$DEMO/run" --max-steps 3

python -m trashpanda train --recipe tiny \
  --data "$DEMO/prepared" --output "$DEMO/run" \
  --resume "$DEMO/run/checkpoint-00000003.pt"
```

The first train command stops at update 3; the second continues to the tiny
recipe's six updates / 744 prediction targets. Both use the same data, recipe,
and settings. The explicit output path is important: the current CLI otherwise
reuses the resume path as its output directory, which is invalid for a file path.

The same fixture can use the second supported architecture. GPT-2 currently
supports the tiny recipe only; when resuming, the checkpoint supplies the model
identity, so `--model` is not required on the second command:

```bash
python -m trashpanda prepare --model gpt2 --recipe tiny \
  --train "$DEMO/train.jsonl" --validation "$DEMO/validation.jsonl" \
  --output "$DEMO/prepared-gpt2"
python -m trashpanda train --model gpt2 --recipe tiny \
  --data "$DEMO/prepared-gpt2" --output "$DEMO/run-gpt2" --max-steps 3
python -m trashpanda train --data "$DEMO/prepared-gpt2" \
  --output "$DEMO/run-gpt2" --resume "$DEMO/run-gpt2/checkpoint-00000003.pt"
```

This exercises process restart, not independently durable machine-loss recovery.
See [recovery limitations](implementation/training.md) before relying on it.

## Evaluate, sample, and export

```bash
python -m trashpanda evaluate \
  --checkpoint "$DEMO/run/checkpoint-00000006.pt" \
  --data "$DEMO/prepared" --max-batches 2

python -m trashpanda sample \
  --checkpoint "$DEMO/run/checkpoint-00000006.pt" \
  --prompt 'w0 w1' --max-new-tokens 8 --seed 42

python -m trashpanda export \
  --checkpoint "$DEMO/run/checkpoint-00000006.pt" \
  --output "$DEMO/export"

python - <<'PY'
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = Path(os.environ["DEMO"]) / "export"
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, trust_remote_code=False
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    path, local_files_only=True, trust_remote_code=False
)
ids = tokenizer("w0 w1", return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = model(input_ids=ids).logits
assert logits.shape[-1] == 32
assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
print("Offline HF load:", tuple(logits.shape), "with tied embeddings")
PY

python -m pytest -q
```

Retain the printed temporary directory if you want to inspect the generated
manifest, metrics and checkpoints. A fresh Git clone does not include the
historical large model binaries. [Artifact availability and GPU examples →](experiments.md)

## Run the offline provenance fixture

For the provenance acceptance path, use the project-owned fixture below. It
creates two synthetic sources, performs real forward/backward updates on an
exactly 100-parameter model, saves and resumes a tiny run, then emits only a
portable bundle. The original temporary sources and checkpoint are not needed
for verification.

```bash
export PROVENANCE_DEMO="$(mktemp -d)/export"
python scripts/provenance_demo.py "$PROVENANCE_DEMO"
python -m trashpanda provenance inspect "$PROVENANCE_DEMO" --json
python -m trashpanda provenance verify "$PROVENANCE_DEMO" --json
cp -a "$PROVENANCE_DEMO" "${PROVENANCE_DEMO}.relocated"
python -m trashpanda provenance verify "${PROVENANCE_DEMO}.relocated" --json
```

The default result should report valid schema and payload integrity, complete
local evidence, and an unsigned artifact. Unsigned means “not authenticated,”
not “invalid.” To demonstrate integrity failure, edit `model.pt` and rerun
verification. Rehashing every file can make integrity valid again, but it does
not create a trusted signature. See
[`implementation/provenance.md`](implementation/provenance.md) for policy,
signing, privacy projections, legacy migration, and the optional Sourcelume
exchange boundary.
