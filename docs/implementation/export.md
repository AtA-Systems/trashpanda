# Current evaluation and export

[Implementation map](README.md) · [Quickstart](../getting-started.md)

## Evaluation and samples

[`evaluate.py`](../../src/trashpanda/evaluate.py) runs inference-mode loss evaluation
without gradients, counts shifted prediction targets, and reports mean loss and
perplexity. The CLI accepts batch size and an optional maximum number of batches.
Sampling supports a seed, temperature, top-k, and generation length with an
isolated Torch generator.

These checks are useful diagnostics, not tests of legal compliance, absence of
memorization, factual reliability, privacy, or readiness for an enterprise task.
Always supply a real validation split: the current CLI falls back to training data
if the requested split is missing. The current perplexity implementation caps the
exponent input at 50; extremely high-loss results should not be interpreted as an
uncapped exact perplexity.

The core training loop does not automatically run the periodic evaluation described
in #1. The optional production driver evaluates and samples after training returns.

## Native Hugging Face export

[`export_hf()`](../../src/trashpanda/export.py) uses the model's standard
`save_pretrained(..., safe_serialization=True)` and the tokenizer's
`save_pretrained()`. It enables generation caching and refuses an existing export
directory. Resulting native model/tokenizer files do not require custom remote
Python classes. [The quickstart](../getting-started.md) includes local-only loading.

For a production checkpoint, the normal checkpoint contains the captured tokenizer
files and can be exported offline. A CLI `--tokenizer` may still explicitly supply
the matching local tokenizer. The export contains no optimizer state, but it adds
`trashpanda-provenance.json` linking the export to the checkpoint, resolved config,
prepared-data identity, and tokenizer snapshot. This does not establish source
rights, approvals, timestamps, signatures, or a license inventory; see
[tokenizer notices](tokenizer-notices.md).

The existing sampling/evaluation CLI may also need the configured production
tokenizer in its local cache; it does not reconstruct tokenizer files from a `.pt`
checkpoint. Preserve those assets separately until the self-contained artifact
contract is implemented.

[`compare_export()`](../../src/trashpanda/export.py) checks an export against its
checkpoint the way a Hugging Face consumer would load it: `AutoModelForCausalLM` and
`AutoTokenizer` from the export folder only, then logits on a prompt (within 1e-4)
and greedy generation (token for token). [`scripts/run_export.py`](../../scripts/run_export.py)
uses it for `--check`, and the container end-to-end test uses it as its last stage.
[`scripts/publish_export.py`](../../scripts/publish_export.py) verifies an export,
adds a model card and uploads it to a Hugging Face repository; see
[Containers](../containers.md).

An exported model can be used for inference; it is not by itself a faithful
training-resume checkpoint. See [training/recovery](training.md) and
[model-bound provenance](../roadmap/provenance.md).
