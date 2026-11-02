import json

import pytest

from trashpanda.cli import _main, build_parser


def test_cli_exposes_model_selector():
    args = build_parser().parse_args(["show-config", "--model", "gpt2"])
    assert args.model == "gpt2"


def test_show_config_reports_gpt2(capsys):
    assert _main(["show-config", "--model", "gpt2"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model"]["architecture"] == "gpt2"
    assert output["model"]["vocab_size"] == 32


def test_cli_rejects_gpt2_production_pair(capsys):
    with pytest.raises(ValueError, match="only supports"):
        _main(["show-config", "--model", "gpt2", "--recipe", "300m-a100"])
