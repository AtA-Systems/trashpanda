import pytest

from trashpanda.config import config_from_dict, get_recipe, load_config, model_recipe, resolve_training_config

pytestmark = pytest.mark.unit


def test_fixed_recipes():
    tiny = get_recipe("tiny")
    assert (tiny.model.vocab_size, tiny.model.num_layers, tiny.model.hidden_size) == (32, 2, 32)
    assert tiny.model.context_length == 32
    assert (tiny.training.batch_size, tiny.training.microbatch_size) == (4, 2)
    assert tiny.training.token_budget == 744
    assert tiny.training.checkpoint_interval == 3
    assert tiny.training.precision == "fp32"
    production = get_recipe("300m-a100")
    assert production.parameter_count == 298_103_808
    assert production.training.token_budget == 6_000_000_000


def test_strict_flat_overrides_and_precedence():
    config = load_config("tiny", overrides={"token_budget": 248, "repeat_data": False})
    assert config.training.token_budget == 248
    assert config.training.repeat_data is False
    config = resolve_training_config(config, microbatch_size=1, learning_rate=1e-4)
    assert config.training.microbatch_size == 1
    assert config.training.learning_rate == 1e-4
    try:
        load_config("tiny", overrides={"training": {"token_budget": 1}})
    except ValueError as exc:
        assert "only allow" in str(exc)
    else:
        raise AssertionError("nested or unknown config keys must be rejected")


def test_model_selection_is_explicit_and_round_trips():
    gpt2 = model_recipe("tiny", "gpt2")
    assert gpt2.model.architecture == "gpt2"
    assert gpt2.model.parameter_count() == 19_200
    assert config_from_dict(gpt2.to_dict()).model.architecture == "gpt2"
    legacy = {
        **gpt2.to_dict(),
        "model": {key: value for key, value in gpt2.to_dict()["model"].items() if key != "architecture"},
    }
    assert config_from_dict(legacy).model.architecture == "llama"
    try:
        model_recipe("300m-a100", "gpt2")
    except ValueError as exc:
        assert "only supports" in str(exc)
    else:
        raise AssertionError("GPT-2 must reject the production Llama recipe")
