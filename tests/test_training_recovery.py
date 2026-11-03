from __future__ import annotations

from pathlib import Path
import random

import pytest
import torch
import numpy as np

from trashpanda.config import model_recipe
from trashpanda.data import SequenceReader
from trashpanda.train import train
from trashpanda.checkpoint import restore_rng_state
from trashpanda.model import build_model

from .helpers import make_prepared_fixture


pytestmark = pytest.mark.acceptance


def _assert_nested_equal(left, right, path="state"):
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_nested_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_split_resume_matches_uninterrupted_training(architecture: str, tmp_path: Path):
    prepared = make_prepared_fixture(tmp_path / "input")
    dataset = SequenceReader(prepared)
    config = model_recipe("tiny", architecture)
    uninterrupted = train(config, dataset, checkpoint_dir=tmp_path / "full", max_steps=6)
    split_first = train(config, dataset, checkpoint_dir=tmp_path / "split", max_steps=3)
    assert split_first.checkpoint is not None
    resumed = train(config, dataset, resume=split_first.checkpoint, checkpoint_dir=tmp_path / "split", max_steps=6)
    assert uninterrupted.state.to_dict() == resumed.state.to_dict() == {"update_step": 6, "consumed_tokens": 744, "micro_steps": 12}
    for name, value in uninterrupted.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    _assert_nested_equal(uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict(), "optimizer")
    assert uninterrupted.sampler.state_dict() == resumed.sampler.state_dict()
    full_payload = torch.load(uninterrupted.checkpoint, map_location="cpu", weights_only=True)
    resumed_payload = torch.load(resumed.checkpoint, map_location="cpu", weights_only=True)
    _assert_nested_equal(full_payload["rng"], resumed_payload["rng"], "rng")
    restore_rng_state(full_payload["rng"])
    draws_full = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng_state(resumed_payload["rng"])
    draws_resumed = (random.random(), float(np.random.random()), torch.rand(3))
    assert draws_full[0:2] == draws_resumed[0:2]
    assert torch.equal(draws_full[2], draws_resumed[2])

    complete = train(config, dataset, resume=resumed.checkpoint, checkpoint_dir=tmp_path / "split", max_steps=6)
    assert complete.state.to_dict() == resumed.state.to_dict()


def test_resume_restores_rng_and_rejects_architecture_change(tmp_path: Path):
    prepared = make_prepared_fixture(tmp_path / "input")
    dataset = SequenceReader(prepared)
    checkpoint = train(model_recipe("tiny", "llama"), dataset, checkpoint_dir=tmp_path / "run", max_steps=3).checkpoint
    assert checkpoint is not None
    with pytest.raises(ValueError, match="model configuration"):
        train(model_recipe("tiny", "gpt2"), dataset, resume=checkpoint, checkpoint_dir=tmp_path / "bad", max_steps=6)
