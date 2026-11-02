# Supported models

Trashpanda supports a small, explicit model catalogue. A model is supported
only when its architecture adapter, tokenizer/data contract, checkpoint resume,
evaluation, sampling, native Hugging Face export, and offline CPU test path are
implemented together.

## Implemented

| Model | Selectors | Recipes | Initialization | Tokenizer and test scope |
| --- | --- | --- | --- | --- |
| Llama | `--model llama` (default) | `tiny`, `300m-a100` | Scratch initialization through `LlamaForCausalLM` | Project-owned 32-token WordLevel tokenizer for `tiny`; pinned Mistral tokenizer for `300m-a100`; tiny CPU workflow is CI-safe |
| GPT-2 | `--model gpt2` | `tiny` | Scratch initialization through `GPT2LMHeadModel` | Project-owned 32-token WordLevel tokenizer; context 32; FP32 CPU workflow is CI-safe |

For example, select GPT-2 while preparing and training the tiny fixture:

```bash
trashpanda prepare --model gpt2 --recipe tiny \
  --train train.jsonl --validation validation.jsonl --output prepared-gpt2/
trashpanda train --model gpt2 --recipe tiny \
  --data prepared-gpt2/ --output run-gpt2/ --tokens 744
```

Both implemented models are randomly initialized demonstration/training
recipes. “Implemented” does not mean that Trashpanda ships pretrained weights,
claims model quality, or accepts arbitrary remote Hugging Face code. Resume,
evaluation, sampling, export, and checkpoint identity are derived from the
selected architecture and reject incompatible configurations.

## Roadmap candidates

These are architecture candidates, not supported models or commitments. Each
would need its own small CPU recipe, tokenizer and vocabulary review, licensing
review, checkpoint/provenance integration, and offline export test before it
could move to Implemented.

| Candidate | Why investigate | Official reference |
| --- | --- | --- |
| OLMo | Open research model family already relevant to Trashpanda's ecosystem | [Transformers OLMo docs](https://huggingface.co/docs/transformers/model_doc/olmo) |
| Mistral | Extends decoder coverage alongside the pinned production tokenizer | [Transformers Mistral docs](https://huggingface.co/docs/transformers/model_doc/mistral) |
| GPT-NeoX | Exercises another causal decoder implementation and configuration shape | [Transformers GPT-NeoX docs](https://huggingface.co/docs/transformers/model_doc/gpt_neox) |
| OPT | Adds a decoder family with learned positional embeddings and distinct defaults | [Transformers OPT docs](https://huggingface.co/docs/transformers/model_doc/opt) |
| BLOOM | Broadens adapter coverage with a distinct decoder implementation | [Transformers BLOOM docs](https://huggingface.co/docs/transformers/model_doc/bloom) |

Vocabulary sizes, tokenizer rights, model licenses, and upstream dependency
support must be checked when any candidate is scheduled. The current uint16
prepared-data format may require a documented change for larger vocabularies.

See the [model-selection roadmap](roadmap/model-selection.md) for the adapter
contract and acceptance evidence.
