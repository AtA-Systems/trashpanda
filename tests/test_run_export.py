import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("PyTorch/transformers are not installed in this environment", allow_module_level=True)

from trashpanda.checkpoint import list_checkpoints
from trashpanda.config import model_recipe
from trashpanda.data import SequenceReader
from trashpanda.export import compare_export, export_checkpoint
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer
from trashpanda.train import train

pytestmark = pytest.mark.acceptance

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_export.py"


@pytest.fixture
def tiny_export(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(256))}) + "\n", encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(64))}) + "\n", encoding="utf-8")
    result = prepare_splits(tmp_path / "prepared", train_path=source, validation_path=validation, tokenizer=TinyWordLevelTokenizer(), context_length=32)
    dataset = SequenceReader(result["train"].directory, split="train")
    train(model_recipe("tiny", "llama"), dataset, tokenizer=TinyWordLevelTokenizer(), checkpoint_dir=tmp_path / "run")
    checkpoint = list_checkpoints(tmp_path / "run")[-1]
    export = export_checkpoint(checkpoint, tmp_path / "export", tokenizer=TinyWordLevelTokenizer())
    return checkpoint, export


def test_compare_export_matches_trained_model(tiny_export):
    checkpoint, export = tiny_export
    report = compare_export(checkpoint, export, prompt="w0 w1 w2")
    assert report["ok"], report["problems"]
    assert report["architectures"] == ["LlamaForCausalLM"]
    assert report["tied_embeddings"] and report["greedy_match"]
    assert report["max_abs_logit_diff"] <= 1e-4


def test_run_export_script_checks_and_generates(tiny_export):
    checkpoint, export = tiny_export
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(export), "--prompt", "w0 w1 w2", "--max-new-tokens", "4", "--greedy", "--device", "cpu", "--check", str(checkpoint)],
        text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "greedy tokens identical: True" in result.stdout
    assert "--- w0 w1 w2" in result.stdout
