from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from trashpanda.federation.evaluate import evaluate_federation
from trashpanda.federation.runner import compare_baselines, make_demo_config, run_federation
from trashpanda.evaluate import EvaluationResult


pytestmark = pytest.mark.acceptance


def test_federated_evaluation_reports_participant_and_weighted_metrics(tmp_path: Path):
    config, _ = make_demo_config(tmp_path / "fixture")
    run_federation(config, tmp_path / "run")
    report = evaluate_federation(tmp_path / "run", max_batches=1)
    assert report["round"] == 3
    assert report["tokens"] == sum(item["tokens"] for item in report["participants"].values())
    expected = sum(item["loss"] * item["tokens"] for item in report["participants"].values()) / report["tokens"]
    assert report["loss"] == pytest.approx(expected)
    assert report["perplexity"] is not None and math.isfinite(report["perplexity"])


def test_baseline_report_contains_actual_work_and_finite_metrics(tmp_path: Path):
    config, _ = make_demo_config(tmp_path / "fixture")
    run_federation(config, tmp_path / "run")
    report = compare_baselines(tmp_path / "run")
    names = {row["name"] for row in report["rows"]}
    assert names == {"local-only-alice", "local-only-bob", "pooled-synthetic"}
    assert all(row["targets"] > 0 and row["validation_tokens"] > 0 and math.isfinite(row["loss"]) for row in report["rows"])
    assert (tmp_path / "run" / "baseline-comparison.json").is_file()
