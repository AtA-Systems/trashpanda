"""Sampling and evaluating a non-tiny checkpoint without a Hub cache."""
from dataclasses import replace
import importlib.util
import json

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("PyTorch/transformers are not installed in this environment", allow_module_level=True)

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from trashpanda.checkpoint import _register_safe_checkpoint_globals
from trashpanda.cli import _checkpoint_model, _main, _tokenizer_from_checkpoint
from trashpanda.config import DataConfig, get_recipe
from trashpanda.data import SequenceReader
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer, load_tokenizer
from trashpanda.train import train

from .helpers import make_prepared_fixture

pytestmark = pytest.mark.acceptance

VOCAB_SIZE = 48


def _fast_tokenizer():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"w{index}": index + 3 for index in range(VOCAB_SIZE - 3)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>")


def _trained_checkpoint(tmp_path):
    """Train one update of a 48-token model whose tokenizer exists only in the prepared directory."""
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        "".join(json.dumps({"text": " ".join(f"w{(i + row) % 40}" for i in range(256))}) + "\n" for row in range(8)),
        encoding="utf-8",
    )
    validation_file = tmp_path / "validation.jsonl"
    validation_file.write_text(json.dumps({"text": " ".join(f"w{(i + 7) % 40}" for i in range(128))}) + "\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepare_splits(prepared, train_path=train_file, validation_path=validation_file, tokenizer=_fast_tokenizer(), context_length=32, vocab_size=VOCAB_SIZE)
    # Like the CLI, train with the tokenizer reloaded from the prepared snapshot.
    tokenizer = load_tokenizer(str(prepared / "tokenizer"), revision="main", vocab_size=VOCAB_SIZE, local_files_only=True)
    base = get_recipe("tiny")
    config = replace(
        base,
        model=replace(base.model, vocab_size=VOCAB_SIZE),
        # A name that resolves only through a Hub cache, like the pinned production tokenizer.
        data=DataConfig(tokenizer_name="example-org/not-in-any-cache", tokenizer_revision="main"),
    )
    result = train(config, SequenceReader(prepared, split="train"), tokenizer=tokenizer, checkpoint_dir=tmp_path / "run", max_steps=1)
    return prepared, result.checkpoint


def test_checkpoint_embeds_a_loadable_tokenizer(tmp_path):
    _prepared, checkpoint = _trained_checkpoint(tmp_path)
    _config, _model, tokenizer = _checkpoint_model(checkpoint)
    assert not isinstance(tokenizer, TinyWordLevelTokenizer)
    ids = tokenizer.encode("w1 w2 w39", add_special_tokens=False)
    assert ids == [4, 5, 42]
    assert tokenizer.decode(ids) == "w1 w2 w39"


def test_sample_and_evaluate_work_without_hub_or_prepared_directory(tmp_path, capsys):
    prepared, checkpoint = _trained_checkpoint(tmp_path)
    assert _main(["sample", "--checkpoint", str(checkpoint), "--prompt", "w1 w2", "--max-new-tokens", "4"]) == 0
    text = capsys.readouterr().out.strip()
    assert text.startswith("w1 w2") and len(text.split()) == 6
    assert _main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(prepared), "--max-batches", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["tokens"] > 0


def test_malformed_embedded_tokenizer_paths_are_rejected(tmp_path):
    _prepared, checkpoint = _trained_checkpoint(tmp_path)
    _register_safe_checkpoint_globals()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config, _model, _tokenizer = _checkpoint_model(checkpoint)
    assert "tokenizer.json" in payload["tokenizer_files"]
    for name in ("../outside.json", "/tmp/absolute.json"):
        with pytest.raises(ValueError, match="malformed"):
            _tokenizer_from_checkpoint({**payload, "tokenizer_files": {name: b"{}"}}, config)
    with pytest.raises(ValueError, match="malformed"):
        _tokenizer_from_checkpoint({**payload, "tokenizer_files": {"tokenizer.json": "not bytes"}}, config)
    # No embedded files: the caller falls back to the name-based lookup.
    assert _tokenizer_from_checkpoint({**payload, "tokenizer_files": {}}, config) is None


def test_tiny_checkpoints_keep_the_project_tokenizer(tmp_path):
    prepared = make_prepared_fixture(tmp_path / "fixture")
    result = train(get_recipe("tiny"), SequenceReader(prepared, split="train"), checkpoint_dir=tmp_path / "run", max_steps=1)
    _config, _model, tokenizer = _checkpoint_model(result.checkpoint)
    assert isinstance(tokenizer, TinyWordLevelTokenizer)
