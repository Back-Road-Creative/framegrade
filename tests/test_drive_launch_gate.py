"""``drive`` renders only when ``health backlog`` started it (``GMS_LAUNCHED_BY``), else the logged
operator override ``GMS_ALLOW_MANUAL_DRIVE=1``. Discovery, ``--dry-run`` and ``--preview`` are exempt."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift import run_origin
from gomoveshift.cli.handlers import drive as drive_handler


@pytest.fixture(autouse=True)
def _bare_env(monkeypatch):
    monkeypatch.delenv(run_origin.LAUNCHED_BY_ENV, raising=False)
    monkeypatch.delenv(drive_handler.ALLOW_MANUAL_DRIVE_ENV, raising=False)


def _args(**kw):
    base = dict(dry_run=False, list_lfs=False, label_dry_run=False, preview=False)
    return SimpleNamespace(**{**base, **kw})


def test_a_hand_launched_render_is_refused():
    why = drive_handler._launch_origin_refusal(_args())
    assert why and "health backlog" in why and drive_handler.ALLOW_MANUAL_DRIVE_ENV in why


def test_the_backlog_marker_lets_a_render_start(monkeypatch):
    monkeypatch.setenv(run_origin.LAUNCHED_BY_ENV, run_origin.BACKLOG_LAUNCHER)
    assert drive_handler._launch_origin_refusal(_args()) is None


def test_an_unknown_marker_value_is_refused_and_named(monkeypatch):
    monkeypatch.setenv(run_origin.LAUNCHED_BY_ENV, "my-script")
    why = drive_handler._launch_origin_refusal(_args())
    assert why and "my-script" in why


@pytest.mark.parametrize("launcher", run_origin.LAUNCHERS)
def test_every_tracked_launcher_value_lets_a_render_start(monkeypatch, launcher):
    monkeypatch.setenv(run_origin.LAUNCHED_BY_ENV, launcher)
    assert drive_handler._launch_origin_refusal(_args()) is None


def test_the_launcher_set_names_the_tracked_launchers():
    assert set(run_origin.LAUNCHERS) == {"health-backlog", "health-heal", "console", "hostworker"}


@pytest.mark.parametrize("flag", ["dry_run", "list_lfs", "label_dry_run", "preview"])
def test_discovery_and_preview_modes_are_exempt(flag):
    assert drive_handler._launch_origin_refusal(_args(**{flag: True})) is None


def test_the_operator_override_is_logged(monkeypatch, caplog):
    monkeypatch.setenv(drive_handler.ALLOW_MANUAL_DRIVE_ENV, "1")
    with caplog.at_level("WARNING"):
        assert drive_handler._launch_origin_refusal(_args()) is None
    assert drive_handler.ALLOW_MANUAL_DRIVE_ENV in caplog.text


def test_cmd_drive_refuses_before_any_work(capsys):
    args = _args(publish="fri", lf_day="fri", day_gpx=None)
    assert drive_handler.cmd_drive(args, object()) == 2
    assert "health backlog" in capsys.readouterr().err
