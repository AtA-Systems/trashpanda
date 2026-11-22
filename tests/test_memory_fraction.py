import pytest

from trashpanda.cli import MEMORY_FRACTION_ENV, apply_memory_fraction, main, parse_memory_fraction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("  ", None), ("0.25", 0.25), ("1", 1.0)])
def test_parse_memory_fraction_accepts(value, expected):
    assert parse_memory_fraction(value) == expected


@pytest.mark.parametrize("value", ["0", "-0.1", "1.5", "quarter"])
def test_parse_memory_fraction_rejects(value):
    with pytest.raises(ValueError, match=MEMORY_FRACTION_ENV):
        parse_memory_fraction(value)


def test_apply_memory_fraction_unset_is_a_no_op():
    assert apply_memory_fraction({}) is None


def test_bad_fraction_is_a_clean_cli_error(monkeypatch, capsys):
    monkeypatch.setenv(MEMORY_FRACTION_ENV, "2")
    assert main(["evaluate", "--checkpoint", "missing.pt", "--data", "missing"]) == 2
    assert MEMORY_FRACTION_ENV in capsys.readouterr().err
