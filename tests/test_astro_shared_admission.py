"""Supported heavy astro CLI commands share the private night worker fence."""

import fcntl
from argparse import Namespace
from unittest.mock import Mock

import pytest

from gomoveshift.cli.handlers import photos

# How long a freshly spawned child interpreter may take to import the package
# and report. 10 s was hit by load alone on a busy CI box
# (`child failed to report admission`); the bound
# guards a hang, not a speed, so it is generous.
CHILD_SPAWN_BUDGET_S = 60


@pytest.mark.parametrize("command", ["astro", "astro-combine", "astro-inspect"])
def test_busy_night_lock_refuses_astro_before_handler(tmp_path, monkeypatch, capsys, command):
    lock_path = tmp_path / "admission.lock"
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(lock_path))
    handler = Mock(return_value=0)
    monkeypatch.setattr(photos, "_cmd_photos_" + command.replace("-", "_"), handler)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert photos.cmd_photos(Namespace(photos_command=command), None) == 2
    handler.assert_not_called()
    assert "admission" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["astro", "astro-combine", "astro-inspect"])
def test_astro_holds_admission_until_handler_finishes(tmp_path, monkeypatch, command):
    lock_path = tmp_path / "admission.lock"
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(lock_path))

    def handler(args):
        with lock_path.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return 7

    monkeypatch.setattr(photos, "_cmd_photos_" + command.replace("-", "_"), handler)
    assert photos.cmd_photos(Namespace(photos_command=command), None) == 7
    with lock_path.open("a") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_lock_symlink_fails_without_decode(tmp_path, monkeypatch):
    target = tmp_path / "private"
    target.write_text("unchanged")
    lock_path = tmp_path / "admission.lock"
    lock_path.symlink_to(target)
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(lock_path))
    handler = Mock(return_value=0)
    monkeypatch.setattr(photos, "_cmd_photos_astro", handler)
    assert photos.cmd_photos(Namespace(photos_command="astro"), None) == 1
    assert target.read_text() == "unchanged"
    handler.assert_not_called()


@pytest.mark.parametrize("mode", ["trails", "timelapse", "triage_only", "from_master"])
def test_heavy_astro_modes_cannot_bypass_fence(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    from gomoveshift.pipeline.media_admission import media_admission

    handler = Mock(return_value=0)
    monkeypatch.setattr(photos, "_cmd_photos_astro", handler)
    with media_admission() as acquired:
        assert acquired
        assert photos.cmd_photos(Namespace(photos_command="astro", **{mode: True}), None) == 2
    handler.assert_not_called()


def test_helper_releases_after_error_and_never_unlinks_lock(tmp_path, monkeypatch):
    from gomoveshift.pipeline.media_admission import media_admission

    path = tmp_path / "admission.lock"
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(path))
    with pytest.raises(RuntimeError, match="render failed"):
        with media_admission() as acquired:
            assert acquired
            inode = path.stat().st_ino
            raise RuntimeError("render failed")
    with media_admission() as acquired:
        assert acquired
        assert path.stat().st_ino == inode
        with media_admission() as nested:
            assert nested is False


def test_admission_excludes_other_process_and_releases_after_death(tmp_path, monkeypatch):
    import select
    import subprocess
    import sys
    from gomoveshift.pipeline.media_admission import media_admission

    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    child_code = (
        "from gomoveshift.pipeline.media_admission import media_admission\n"
        "import sys\n"
        "with media_admission() as acquired:\n"
        " print(acquired, flush=True)\n"
        " sys.stdin.read()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        readable, _, _ = select.select([proc.stdout], [], [], CHILD_SPAWN_BUDGET_S)
        assert readable, "child failed to report admission"
        assert proc.stdout.readline().strip() == "True"
        with media_admission() as acquired:
            assert not acquired
        proc.kill()
        proc.wait(timeout=5)
        with media_admission() as acquired:
            assert acquired
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()


def test_legacy_and_night_workers_mutually_exclude(tmp_path, monkeypatch):
    import json
    from gomoveshift.pipeline.media_admission import media_admission
    from gomoveshift.pipeline.photos.night_project import NightStore, process_project

    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    source = tmp_path / "stack.linear"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"payloads": {"sky_sum.npy": {"shape": [10, 20, 3]}}})
    )
    store = NightStore(tmp_path / "different-workspace")
    project = store.create("brand", [source], [tmp_path], "natural")
    render = Mock()
    with media_admission() as acquired:
        assert acquired
        assert process_project(store, "brand", project, render=render, admitted=lambda _: True) == 2
    render.assert_not_called()
    handler = Mock(return_value=0)
    monkeypatch.setattr(photos, "_cmd_photos_astro", handler)

    def night_render(*args):
        assert photos.cmd_photos(Namespace(photos_command="astro"), None) == 2

    assert (
        process_project(
            store,
            "brand",
            project,
            render=night_render,
            validate=lambda *a: True,
            admitted=lambda _: True,
        )
        == 0
    )
    handler.assert_not_called()


def test_cli_help_needs_no_heavy_admission(tmp_path, monkeypatch, capsys):
    from gomoveshift.pipeline.media_admission import media_admission
    from gomoveshift.cli.parser import _build_parser

    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    with media_admission():
        with pytest.raises(SystemExit) as stopped:
            _build_parser().parse_args(["photos", "astro", "--help"])
    assert stopped.value.code == 0
    assert "--stack-len" in capsys.readouterr().out
