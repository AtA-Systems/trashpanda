import importlib.util
import json

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("PyTorch/transformers are not installed in this environment", allow_module_level=True)

import torch

from trashpanda.checkpoint import list_checkpoints
from trashpanda.data import SequenceReader
from trashpanda.evaluate import evaluate
from trashpanda.evaluate import sample_text
from trashpanda.export import export_checkpoint
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer
from trashpanda.train import train

pytestmark = pytest.mark.acceptance


def _dataset(tmp_path):
    text = " ".join(f"w{i % 29}" for i in range(256))
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(64))}) + "\n", encoding="utf-8")
    result = prepare_splits(tmp_path / "prepared", train_path=source, validation_path=validation, tokenizer=TinyWordLevelTokenizer(), context_length=32)
    return SequenceReader(result["train"].directory, split="train"), SequenceReader(result["train"].directory, split="validation")


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_tiny_train_resume_evaluate_export(tmp_path, architecture):
    from trashpanda.config import model_recipe

    config = model_recipe("tiny", architecture)
    dataset, validation = _dataset(tmp_path)
    first = train(config, dataset, checkpoint_dir=tmp_path / f"checkpoints-{architecture}", max_steps=3)
    assert first.state.consumed_tokens == 372
    assert first.checkpoint is not None
    first_weights = {name: value.detach().clone() for name, value in first.model.state_dict().items()}
    resumed = train(config, dataset, resume=first.checkpoint, checkpoint_dir=tmp_path / f"checkpoints-{architecture}")
    assert resumed.state.consumed_tokens == 744
    assert any(not torch.equal(first_weights[name], value) for name, value in resumed.model.state_dict().items())
    assert len(list_checkpoints(tmp_path / f"checkpoints-{architecture}")) >= 2
    result = evaluate(resumed.model, validation, max_batches=1)
    assert result.tokens == 31
    assert result.loss == result.loss and result.loss < float("inf")
    assert isinstance(sample_text(resumed.model, TinyWordLevelTokenizer(), "w0", max_new_tokens=2, seed=7), str)
    output = export_checkpoint(resumed.checkpoint, tmp_path / f"export-{architecture}")
    assert (output / "config.json").exists()
    assert (output / "pytorch_model.bin").exists() or (output / "model.safetensors").exists()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    offline_model = AutoModelForCausalLM.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    offline_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    assert offline_model.config.model_type == {"llama": "llama", "gpt2": "gpt2"}[architecture]
    assert offline_tokenizer.vocab_size == config.model.vocab_size

    other = model_recipe("tiny", "gpt2" if architecture == "llama" else "llama")
    with pytest.raises(ValueError, match="model configuration"):
        train(other, dataset, resume=first.checkpoint, checkpoint_dir=tmp_path / f"cross-{architecture}")
