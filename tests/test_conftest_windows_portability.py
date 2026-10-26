"""``tests/conftest.py`` loads on Windows, where ``pwd`` and ``st_blocks`` do not exist.

Windows CI (``.github/workflows/windows.yml``) will run tests through this
conftest, so its module-level code may use only what Windows has. Windows also
reads the home directory from ``USERPROFILE``, not ``HOME``, so the fake home
has to be set there too or import-time ``Config.load()`` reads the real user's
config. Both are checked here on Linux by hiding ``pwd``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import conftest  # pytest's rootdir conftest, already loaded under this name

CONFTEST = Path(__file__).parent / "conftest.py"
SERVICE_ROOT = CONFTEST.parent.parent


def test_conftest_loads_without_pwd_and_fakes_both_home_variables(tmp_path):
    script = textwrap.dedent(
        f"""
        import json, os, runpy, sys
        sys.modules["pwd"] = None  # what Windows has: no pwd module
        sys.path.insert(0, {str(SERVICE_ROOT)!r})
        ns = runpy.run_path({str(CONFTEST)!r})
        print(json.dumps({{
            "test_home": ns["_TEST_HOME"],
            "HOME": os.environ["HOME"],
            "USERPROFILE": os.environ["USERPROFILE"],
        }}))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "USERPROFILE"}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["HOME"] == got["test_home"]
    assert got["USERPROFILE"] == got["test_home"]
    assert Path(got["test_home"]).name.startswith("gms-test-home-")


def test_disk_use_falls_back_to_size_when_stat_has_no_block_count(tmp_path, monkeypatch):
    (tmp_path / "f.bin").write_bytes(b"x" * 5000)
    real_lstat = os.lstat

    def lstat_without_blocks(path):
        st = real_lstat(path)
        return SimpleNamespace(st_size=st.st_size)  # Windows os.stat_result has no st_blocks

    monkeypatch.setattr(os, "lstat", lstat_without_blocks)
    assert conftest._bytes_on_disk(tmp_path) == 5000
