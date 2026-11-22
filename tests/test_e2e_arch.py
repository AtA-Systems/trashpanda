"""The container end-to-end test's GPU/flavor check (docker/tests/e2e.py)."""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location("tp_e2e", Path(__file__).resolve().parents[1] / "docker" / "tests" / "e2e.py")
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)

PASCAL_IMAGE = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]  # torch 2.14 cu126, as reported on Bob
MODERN_IMAGE = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


@pytest.mark.parametrize("archs, major, minor, expected", [
    (PASCAL_IMAGE, 6, 1, True),    # TITAN X (Pascal) runs sm_60 code
    (PASCAL_IMAGE, 6, 0, True),
    (PASCAL_IMAGE, 8, 9, True),    # Ada runs sm_86 code
    (PASCAL_IMAGE, 12, 0, False),  # Blackwell workstation: not in the cu126 build
    (MODERN_IMAGE, 6, 1, False),   # wrong flavor for Pascal
    (MODERN_IMAGE, 12, 0, True),   # RTX PRO 6000 Blackwell
    (MODERN_IMAGE, 12, 1, True),
    (["compute_80"], 9, 0, True),  # PTX JIT for newer GPUs
    (["compute_90"], 8, 0, False),
])
def test_built_for(archs, major, minor, expected):
    assert e2e.built_for(archs, major, minor) is expected
