"""Notebook stopping policy can be checked without running a long training job."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def decision(losses, **overrides):
    notebook = Path(__file__).resolve().parents[1] / "examples" / "trashpanda_colab.ipynb"
    cell = next(c for c in json.loads(notebook.read_text())["cells"] if c["id"] == "stopping")
    namespace = {}
    exec("".join(cell["source"]), namespace)
    criteria = dict(enabled=True, patience=2, min_delta=0.01, target_loss=0)
    criteria.update(overrides)
    observations = [{"loss": loss, "checkpoint": f"checkpoint-{index}.pt"} for index, loss in enumerate(losses)]
    return namespace["stopping_decision"](observations, **criteria)


def test_plateau_and_meaningful_improvement():
    assert decision([2.0, 1.5, 1.495])["stop_reason"] is None
    result = decision([2.0, 1.5, 1.495, 1.499])
    assert result["stop_reason"] == "validation_plateau"
    assert result["best_checkpoint"] == "checkpoint-2.pt"
    assert decision([2.0, 1.5, 1.495, 1.4])["stale_evaluations"] == 0


def test_target_disabled_and_nonfinite():
    assert decision([2.0, 1.5], target_loss=1.5)["stop_reason"] == "target_validation_loss"
    assert decision([2.0, 2.1, 2.2], enabled=False)["stop_reason"] is None
    assert decision([2.0, None])["stop_reason"] == "non_finite_validation_loss"
    assert decision([float("nan")], enabled=False)["stop_reason"] == "non_finite_validation_loss"
