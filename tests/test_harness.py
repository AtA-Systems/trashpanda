from pathlib import Path
import sys

import pytest

from .conftest import run_command


pytestmark = pytest.mark.unit


def test_subprocess_helper_captures_output_and_nonzero_status(tmp_path: Path):
    result = run_command([sys.executable, "-c", "print('stdout'); raise SystemExit(7)"], cwd=tmp_path, check=False)
    assert result.returncode == 7
    assert result.stdout.strip() == "stdout"


def test_subprocess_helper_enforces_timeout():
    with pytest.raises(pytest.fail.Exception, match="timed out"):
        run_command([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.01)
