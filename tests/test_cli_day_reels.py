"""``gomoveshift day-reels`` (cli/handlers/day_reels.py)."""

from __future__ import annotations

import base64
import json
import subprocess
from types import SimpleNamespace

from gomoveshift.cli.handlers import day_reels as handler
from gomoveshift.cli.parser import _build_parser

RCLONE_REPLY = '{"access_token":"ya29.secret","refresh_token":"1//secret"}'


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "queue")))


def test_parser_accepts_run_and_authorize():
    args = _build_parser().parse_args(["day-reels", "--apply"])
    assert (args.command, args.action, args.apply) == ("day-reels", "run", True)
    args = _build_parser().parse_args(["day-reels", "authorize", "--folder-id", "1hn4wHfc10-N7"])
    assert (args.action, args.folder_id) == ("authorize", "1hn4wHfc10-N7")


def test_run_without_a_sign_in_is_a_quiet_skip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GMS_DAY_REELS_DRIVE_ENC", str(tmp_path / "missing.json.enc"))
    args = SimpleNamespace(action="run", apply=True, folder_id=None)
    assert handler.cmd_day_reels(args, _config(tmp_path)) == 0
    assert "DAY_REELS skipped" in capsys.readouterr().out


def test_authorize_asks_read_only_scope_and_stores_the_token_encrypted(
    tmp_path, monkeypatch, capsys
):
    enc = tmp_path / "day_reels_drive.json.enc"
    monkeypatch.setenv("GMS_DAY_REELS_DRIVE_ENC", str(enc))
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        out = f"Paste the following into your remote machine --->\n{RCLONE_REPLY}\n<---End paste\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr=None)

    stored = {}
    monkeypatch.setattr(handler.subprocess, "run", fake_run)
    monkeypatch.setattr(handler, "_encrypt", lambda data, path: stored.update(data=data, path=path))

    args = SimpleNamespace(action="authorize", apply=False, folder_id="1hn4wHfc10-N7OdTo5vzn")
    assert handler.cmd_day_reels(args, _config(tmp_path)) == 0

    assert seen["argv"][1:3] == ["authorize", "drive"]
    arg = seen["argv"][3]
    assert "=" not in arg and "+" not in arg and "/" not in arg, "rclone wants URL-safe, unpadded"
    blob = json.loads(base64.urlsafe_b64decode(arg + "=" * (-len(arg) % 4)))
    assert blob == {"scope": "drive.readonly"}
    assert seen["kwargs"]["stdout"] == subprocess.PIPE
    assert stored["path"] == enc
    assert stored["data"] == {"rclone_token": RCLONE_REPLY, "folder_id": "1hn4wHfc10-N7OdTo5vzn"}
    assert "secret" not in capsys.readouterr().out


def test_authorize_refuses_a_bad_folder_id(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_DAY_REELS_DRIVE_ENC", str(tmp_path / "x.enc"))
    args = SimpleNamespace(action="authorize", apply=False, folder_id="../x")
    assert handler.cmd_day_reels(args, _config(tmp_path)) == 2


def test_run_prints_one_line_per_request(tmp_path, monkeypatch, capsys):
    enc = tmp_path / "day_reels_drive.json.enc"
    enc.write_text("{}")
    monkeypatch.setenv("GMS_DAY_REELS_DRIVE_ENC", str(enc))
    monkeypatch.setattr(
        handler, "_decrypt", lambda path: {"rclone_token": RCLONE_REPLY, "folder_id": "F1234567890"}
    )
    captured = {}

    def fake_process(drive, **kwargs):
        captured.update(kwargs, drive=drive)
        return [handler.Outcome("day10.json", "enqueued", "2 new, 0 already queued", (7, 8))]

    monkeypatch.setattr(handler, "process_requests", fake_process)
    args = SimpleNamespace(action="run", apply=True, folder_id=None)
    assert handler.cmd_day_reels(args, _config(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "DAY_REEL enqueued day10.json" in out and "ids=7,8" in out
    assert "secret" not in out
    assert captured["apply"] is True
    assert str(captured["storage_path"]) == str(tmp_path / "queue")
