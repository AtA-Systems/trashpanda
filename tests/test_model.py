import importlib.util

import pytest

from trashpanda.config import get_recipe, model_recipe

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("PyTorch/transformers are not installed in this environment", allow_module_level=True)

import torch

from trashpanda.model import build_model, count_parameters


def test_tiny_model_shape_count_and_seed():
    config = get_recipe("tiny")
    first = build_model(config.model, seed=11)
    second = build_model(config.model, seed=11)
    assert count_parameters(first) == config.parameter_count
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    outputs = first(input_ids=torch.zeros((2, 7), dtype=torch.long))
    assert outputs.logits.shape == (2, 7, config.model.vocab_size)
    assert first.config._attn_implementation == "sdpa"
    assert first.get_input_embeddings().weight is first.get_output_embeddings().weight


def test_tiny_gpt2_model_shape_count_and_seed():
    config = model_recipe("tiny", "gpt2")
    first = build_model(config.model, seed=11)
    second = build_model(config.model, seed=11)
    assert count_parameters(first) == config.parameter_count
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    outputs = first(input_ids=torch.zeros((2, 7), dtype=torch.long))
    assert outputs.logits.shape == (2, 7, config.model.vocab_size)
    assert first.config.model_type == "gpt2"
    assert first.get_input_embeddings().weight is first.get_output_embeddings().weight
