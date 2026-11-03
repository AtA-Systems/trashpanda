from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import torch

from .conftest import run_cli


pytestmark = pytest.mark.acceptance


def _write_inputs(root: Path) -> tuple[Path, Path]:
    train = root / "train.jsonl"
    validation = root / "validation.jsonl"
    train.write_text("".join(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(256))}) + "\n" for _ in range(8)), encoding="utf-8")
    validation.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(128))}) + "\n", encoding="utf-8")
    return train, validation


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_cli_lifecycle_is_architecture_specific_and_portable(tmp_path: Path, architecture: str):
    train_input, validation_input = _write_inputs(tmp_path)
    prepared = tmp_path / f"prepared-{architecture}"
    run_cli("prepare", "--model", architecture, "--recipe", "tiny", "--train", train_input, "--validation", validation_input, "--output", prepared)
    run_dir = tmp_path / f"run-{architecture}"
    run_cli("train", "--model", architecture, "--recipe", "tiny", "--data", prepared, "--output", run_dir, "--tokens", "744")
    checkpoint = run_dir / "checkpoint-00000006.pt"
    assert checkpoint.is_file()
    evaluation = json.loads(run_cli("evaluate", "--checkpoint", checkpoint, "--data", prepared, "--max-batches", "1").stdout)
    assert evaluation["tokens"] > 0 and evaluation["loss"] < float("inf")
    export = tmp_path / f"export-{architecture}"
    run_cli("export", "--checkpoint", checkpoint, "--output", export)
    run_cli("provenance", "verify", export, "--json")

    from trashpanda.cli import _checkpoint_model
    _, before_model, before_tokenizer = _checkpoint_model(checkpoint)
    before_inputs = torch.tensor([before_tokenizer.encode("w0 w1 w2", add_special_tokens=False)], dtype=torch.long)
    with torch.no_grad():
        before_logits = before_model(input_ids=before_inputs).logits

    relocated = tmp_path / f"relocated-{architecture}"
    shutil.copytree(export, relocated)
    shutil.rmtree(prepared)
    shutil.rmtree(run_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(relocated, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(relocated, local_files_only=True, trust_remote_code=False)
    assert model.config.model_type == architecture
    inputs = torch.tensor([tokenizer.encode("w0 w1 w2", add_special_tokens=False)], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=inputs).logits
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(before_logits, logits, rtol=1e-5, atol=1e-6)
