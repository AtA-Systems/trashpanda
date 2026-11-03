from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trashpanda.config import model_recipe
from trashpanda.data import SequenceReader
from trashpanda.train import train

from .helpers import make_prepared_fixture


pytestmark = pytest.mark.extended


@pytest.mark.parametrize("seed", [17, 23])
def test_extra_cpu_seed_produces_finite_changed_weights(tmp_path: Path, seed: int):
    prepared = make_prepared_fixture(tmp_path / f"input-{seed}", offset=seed)
    config = model_recipe("tiny", "llama")
    config = config.__class__(model=config.model, data=config.data, training=config.training.__class__(**{**config.training.__dict__, "seed": seed}), runtime=config.runtime, recipe_name=config.recipe_name)
    result = train(config, SequenceReader(prepared), checkpoint_dir=tmp_path / f"run-{seed}", max_steps=1)
    assert result.state.consumed_tokens == 124
    assert all(torch.isfinite(value).all() for value in result.model.state_dict().values() if value.dtype.is_floating_point)
