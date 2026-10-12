"""Hermeticity guarantees provided by ``tests/conftest.py``.

Two leak classes (both via ``/tmp/pipeline.log``):

1. ``test_pipeline_resilience`` called the real ``cmd_pipeline`` with
   ``output_dir=Path("/tmp")`` and MagicMock args, attaching a JSON
   ``RotatingFileHandler`` to the root logger with ``force=True``. The
   handler survived for the rest of the xdist worker, so every later
   test's log records leaked into ``/tmp/pipeline.log``.
2. Phase-BC tests reached ``destination_resolver``'s POI source, which
   attempted three live Overpass HTTP queries per day (Nominatim was
   already stubbed in conftest; Overpass never was).
"""

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

CONFTEST = Path(__file__).parent / "conftest.py"
# The conftest import chain (drive_orchestrator → tools.validate_pipeline_output)
# resolves the non-packaged top-level ``tools/`` from the service root. The main
# suite gets it via pytest's rootdir insertion; the mini-suite subprocess runs
# from tmp_path and must be handed it explicitly (CI installs the package
# non-editable, so site-packages provides no path back to the source tree).
SERVICE_ROOT = Path(__file__).resolve().parent.parent


def test_overpass_is_hermetic(monkeypatch):
    """Production caller paths must not reach urllib for Overpass in tests.

    ``destination_resolver`` and ``OverpassChapterSource`` both resolve
    ``_query_overpass`` through a module attribute at call time — the same
    lookup used here. The conftest ``hermetic_overpass`` fixture must have
    replaced it with a stub that raises the degraded-path ``RuntimeError``
    without touching the network.
    """
    from gomoveshift.pipeline.chapters import overpass_source

    attempts = []

    class _Resp:
        status = 200

        def read(self):
            return b'{"elements": []}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        overpass_source.urllib.request,
        "urlopen",
        lambda req, timeout=None: (attempts.append(req), _Resp())[1],
    )

    with pytest.raises(RuntimeError):
        overpass_source._query_overpass((51.0, -10.0, 52.0, -9.0))

    assert not attempts, "live Overpass HTTP attempted from a test"


def test_root_logger_file_handler_restored_between_tests(tmp_path):
    """A file handler attached by one test must be gone in the next.

    Runs a two-test mini-suite under this repo's conftest in a subprocess
    (deterministic ordering — no xdist), reproducing the original leak
    shape: test one configures a log file, test two requires a clean root.
    """
    mini = tmp_path / "test_leak_demo.py"
    mini.write_text(
        textwrap.dedent(
            """
            import logging
            from logging.handlers import RotatingFileHandler

            from gomoveshift.cli.logging_setup import configure_logging

            def _file_handlers():
                return [
                    h
                    for h in logging.getLogger().handlers
                    if isinstance(h, RotatingFileHandler)
                ]

            def test_attach(tmp_path):
                configure_logging(log_file=tmp_path / "leak.log")
                assert _file_handlers()

            def test_root_clean_after_previous():
                assert not _file_handlers()
            """
        )
    )
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(mini),
            "-o",
            "addopts=",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(SERVICE_ROOT), os.environ.get("PYTHONPATH")])
            ),
        },
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"


def test_no_leaked_pipeline_log_handler():
    """No earlier test in this worker may have left a root file handler."""
    leaked = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert not leaked, f"leaked root file handlers: {[h.baseFilename for h in leaked]}"


def test_dod_stub_survives_missing_tools_sentinel(monkeypatch):
    """The autouse DoD stub must work under a wheel install too.

    ``drive_orchestrator`` binds ``ValidationMode`` to a sentinel that raises
    ``RuntimeError`` on any attribute when repo-root ``tools`` is missing (CI
    installs non-editable). The gate evaluates ``ValidationMode.AUTO`` before
    it calls ``run_validation``, so stubbing only the callable leaves every
    orchestrator unit test red there.
    """
    from types import SimpleNamespace

    from conftest import _stub_dod_gate

    class _Sentinel:
        def __getattr__(self, name):
            raise RuntimeError("tools/validate_pipeline_output.py is not installed")

    module = SimpleNamespace(ValidationMode=_Sentinel(), run_validation=None)
    _stub_dod_gate(monkeypatch, module)

    assert module.ValidationMode.AUTO == "auto"
    assert module.run_validation(mode=module.ValidationMode.AUTO).failed_count == 0
