import json

from trashpanda.data import PCG64Sampler, SequenceReader
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer

import pytest

pytestmark = pytest.mark.unit


def test_prepare_reader_and_pcg64(tmp_path):
    text = " ".join(f"w{i % 29}" for i in range(17))
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"text": "w0 w1 w2 w3 w4 w5 w6 w7"}) + "\n", encoding="utf-8")
    result = prepare_splits(tmp_path / "prepared", train_path=source, validation_path=validation, tokenizer=TinyWordLevelTokenizer(), context_length=8, shard_tokens=8)
    assert result["train"].sequence_count == 2
    assert result["train"].dropped_tokens == 2
    assert result["validation"].sequence_count >= 1
    dataset = SequenceReader(result["train"].directory, split="train", verify_hash=True)
    tokens = dataset[0]
    assert len(tokens) == 8
    assert str(tokens.dtype) == "torch.int64"
    sampler_a = PCG64Sampler(len(dataset), seed=7)
    sampler_b = PCG64Sampler(len(dataset), seed=7)
    assert sampler_a.next_indices(10) == sampler_b.next_indices(10)
    manifest = json.loads((tmp_path / "prepared" / "manifest.json").read_text())
    assert manifest["dtype"] == "uint16"
    assert manifest["splits"]["train"]["dropped_tokens"] == 2
