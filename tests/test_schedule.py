import pytest

from trashpanda.train import TokenBudgetSchedule

pytestmark = pytest.mark.unit


def test_shifted_token_schedule():
    schedule = TokenBudgetSchedule(context_length=32, effective_batch_size=4, token_budget=744)
    assert schedule.tokens_per_sequence == 31
    assert schedule.tokens_per_update == 124
    assert schedule.updates == 6
    assert schedule.accounted_tokens == 744
