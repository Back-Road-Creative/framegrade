"""The suite answers the NVENC probe itself unless a test is marked ``real_nvenc``.

``detect_nvenc()`` is a real ``hevc_nvenc`` ffmpeg encode with a 10 s timeout,
called from ``SinglePassProcessor.__init__``, ``shorts_extractor``,
``ost_applier``, ``stop_processor``, ``state_invariants`` and
``route_story.encode``. Tests that only assemble a command read the host GPU
through it: on a CI runner whose GPU was busy the probe timed out and turned an
unrelated test red (``NVENC probe failed (TimeoutExpired)`` at setup of
``test_single_pass_blur_encode``), and with the GPU hidden 5 tests across 3
probe call sites failed. The conftest answers only the probe's exact argv with
returncode 0, so ``detect_nvenc``'s own code still runs, and a test that
patches ``subprocess.run`` or ``detect_nvenc`` itself still wins.
"""

import subprocess

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import shorts_extractor as se
from gomoveshift.pipeline import single_pass as sp
from gomoveshift.pipeline.color_filters import NVENCUnavailableError, detect_nvenc


def _no_process(*args, **kwargs):
    raise FileNotFoundError("no process may launch in this test")


def test_probe_is_answered_without_launching_ffmpeg(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_process)
    assert detect_nvenc() is True


def test_call_sites_see_the_answered_probe(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_process)
    monkeypatch.setattr(se, "_nvenc_available", None)
    assert se._detect_nvenc_cached() is True
    assert sp.SinglePassProcessor(Config())._has_nvenc() is True


def test_other_commands_still_reach_subprocess(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_process)
    with pytest.raises(FileNotFoundError):
        subprocess.run(["ffmpeg", "-version"], capture_output=True)


@pytest.mark.real_nvenc
def test_real_nvenc_marker_runs_the_real_probe(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _no_process)
    with pytest.raises(NVENCUnavailableError):
        detect_nvenc()
