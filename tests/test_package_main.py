"""``python -m gomoveshift`` must work: it is the argv the console launches."""

import subprocess
import sys


def test_python_dash_m_gomoveshift_runs():
    res = subprocess.run(
        [sys.executable, "-m", "gomoveshift", "--version"], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    assert "gomoveshift.__main__" not in res.stderr
