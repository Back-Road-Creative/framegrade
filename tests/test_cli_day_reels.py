"""``gomoveshift day-reels`` (cli/handlers/day_reels.py)."""

from __future__ import annotations

import base64
import json
import subprocess
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers import day_reels as handler
from gomoveshift.cli.parser import _build_parser

RCLONE_REPLY = '{"access_token":"ya29.secret","refresh_token":"1//secret"}'
OWN_CLIENT = ("own-client.apps.googleusercontent.com", "own-client-secret")


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
    monkeypatch.setattr(handler, "_pipeline_client", lambda: OWN_CLIENT)

    args = SimpleNamespace(action="authorize", apply=False, folder_id="1hn4wHfc10-N7OdTo5vzn")
    assert handler.cmd_day_reels(args, _config(tmp_path)) == 0

    assert seen["argv"][1:3] == ["authorize", "drive"]
    arg = seen["argv"][3]
    assert "=" not in arg and "+" not in arg and "/" not in arg, "rclone wants URL-safe, unpadded"
    blob = json.loads(base64.urlsafe_b64decode(arg + "=" * (-len(arg) % 4)))
    assert blob == {"scope": "drive.readonly"}
    assert seen["kwargs"]["stdout"] == subprocess.PIPE
    assert stored["path"] == enc
    assert stored["data"] == {
        "rclone_token": RCLONE_REPLY,
        "folder_id": "1hn4wHfc10-N7OdTo5vzn",
        "client_id": OWN_CLIENT[0],
        "client_secret": OWN_CLIENT[1],
    }
    # The pipeline's own Google client, not rclone's shared one, and never in argv.
    assert seen["kwargs"]["env"]["RCLONE_DRIVE_CLIENT_ID"] == OWN_CLIENT[0]
    assert seen["kwargs"]["env"]["RCLONE_DRIVE_CLIENT_SECRET"] == OWN_CLIENT[1]
    assert all(OWN_CLIENT[1] not in part for part in seen["argv"])
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


def test_authorize_refuses_without_the_pipelines_own_client(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GMS_DAY_REELS_DRIVE_ENC", str(tmp_path / "x.enc"))

    def no_client():
        raise handler.ClientUnavailableError("drive token unreadable")

    monkeypatch.setattr(handler, "_pipeline_client", no_client)
    monkeypatch.setattr(
        handler.subprocess, "run", lambda *a, **k: pytest.fail("rclone must not run")
    )
    monkeypatch.setattr(handler, "_encrypt", lambda *a: pytest.fail("nothing may be saved"))
    args = SimpleNamespace(action="authorize", apply=False, folder_id="1hn4wHfc10-N7OdTo5vzn")
    assert handler.cmd_day_reels(args, _config(tmp_path)) == 1
    assert "own Google client" in capsys.readouterr().out


def test_pipeline_client_comes_from_the_drive_token(monkeypatch, tmp_path):
    token = tmp_path / "drive_token.json.enc"
    monkeypatch.setenv("GMS_DRIVE_TOKEN_ENC", str(token))
    seen = {}

    def fake_decrypt(path):
        seen["path"] = path
        return {"client_id": OWN_CLIENT[0], "client_secret": OWN_CLIENT[1], "refresh_token": "r"}

    monkeypatch.setattr(handler, "_decrypt", fake_decrypt)
    assert handler._pipeline_client() == OWN_CLIENT
    assert seen["path"] == token
    monkeypatch.setattr(handler, "_decrypt", lambda path: {"refresh_token": "r"})
    with pytest.raises(handler.ClientUnavailableError):
        handler._pipeline_client()


def test_run_hands_the_stored_client_to_the_drive_reader(tmp_path, monkeypatch):
    enc = tmp_path / "day_reels_drive.json.enc"
    enc.write_text("{}")
    monkeypatch.setenv("GMS_DAY_REELS_DRIVE_ENC", str(enc))
    creds = {"rclone_token": RCLONE_REPLY, "folder_id": "F1234567890"}
    creds.update(client_id=OWN_CLIENT[0], client_secret=OWN_CLIENT[1])
    monkeypatch.setattr(handler, "_decrypt", lambda path: creds)
    seen = {}

    def fake_process(drive, **kwargs):
        seen["env"] = drive._env()
        return []

    monkeypatch.setattr(handler, "process_requests", fake_process)
    args = SimpleNamespace(action="run", apply=False, folder_id=None)
    assert handler.cmd_day_reels(args, _config(tmp_path)) == 0
    assert seen["env"]["RCLONE_CONFIG_GMSDAYREELS_CLIENT_ID"] == OWN_CLIENT[0]
    assert seen["env"]["RCLONE_CONFIG_GMSDAYREELS_CLIENT_SECRET"] == OWN_CLIENT[1]
