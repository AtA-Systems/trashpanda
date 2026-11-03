"""Small deterministic fixtures shared by acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import TinyWordLevelTokenizer


def make_prepared_fixture(root: Path, *, rows: int = 8, offset: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    train = root / "train.jsonl"
    validation = root / "validation.jsonl"
    train.write_text(
        "".join(json.dumps({"text": " ".join(f"w{(i + offset + row) % 29}" for i in range(256))}) + "\n" for row in range(rows)),
        encoding="utf-8",
    )
    validation.write_text(
        "".join(json.dumps({"text": " ".join(f"w{(i + offset + 13 + row) % 29}" for i in range(128))}) + "\n" for row in range(max(1, rows // 2))),
        encoding="utf-8",
    )
    prepared = root / "prepared"
    prepare_splits(
        prepared,
        train_path=train,
        validation_path=validation,
        tokenizer=TinyWordLevelTokenizer(),
        context_length=32,
        source_descriptors=[
            {"source_id": f"fixture:{offset}", "source": "test-fixture", "source_revision": "v1", "evidence_state": "observed"},
            {"source_id": f"fixture:{offset}", "source": "test-fixture", "source_revision": "v1", "evidence_state": "observed"},
        ],
    )
    return prepared


def state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    if isinstance(state, torch.Tensor):
        return {"dtype": str(state.dtype), "shape": tuple(state.shape), "value": state.detach().cpu().clone()}
    if isinstance(state, dict):
        return {str(key): state_snapshot(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return [state_snapshot(value) for value in state]
    return state
