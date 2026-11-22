import importlib.util
import json
import math

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("PyTorch/transformers are not installed in this environment", allow_module_level=True)

from trashpanda.config import model_recipe
from trashpanda.data import SequenceReader
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer
from trashpanda.train import train

pytestmark = pytest.mark.acceptance


def test_metrics_log_the_scheduled_learning_rate(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(256))}) + "\n", encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(64))}) + "\n", encoding="utf-8")
    result = prepare_splits(tmp_path / "prepared", train_path=source, validation_path=validation, tokenizer=TinyWordLevelTokenizer(), context_length=32)
    config = model_recipe("tiny", "llama")
    train(config, SequenceReader(result["train"].directory, split="train"), tokenizer=TinyWordLevelTokenizer(), checkpoint_dir=tmp_path / "run")
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines() if line.strip()]
    t = config.training
    total = len(rows)
    assert total == 6
    for row in rows:
        step = row["step"] - 1  # the schedule index this update used
        progress = min(1.0, max(0.0, (step - t.warmup_steps) / max(1, total - t.warmup_steps)))
        factor = t.min_lr_ratio + (1.0 - t.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        assert row["learning_rate"] == pytest.approx(t.learning_rate * factor)
    assert rows[0]["learning_rate"] == pytest.approx(t.learning_rate)
    assert rows[-1]["learning_rate"] < rows[0]["learning_rate"]
