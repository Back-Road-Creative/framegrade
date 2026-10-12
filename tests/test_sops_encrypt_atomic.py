"""The SOPS write-back must be atomic and cross-process safe.

The encrypted OAuth sidecar is shared by the photo run and the drive run —
two separate OS processes. The in-process ``threading.Lock`` guards neither,
and a temp file in the default temp dir sits on a different filesystem from
the target, so the install degrades to a non-atomic copy-then-delete. Either
hole can tear the file, and a torn credential costs a hand re-mint.

These tests exercise the temp-dir/replace seam only: SOPS itself is stubbed
and every fixture uses obviously-fake placeholder content — no real
credential file is read or written.
"""

import fcntl
import os
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from gomoveshift.pipeline import uploader

FAKE_PAYLOAD = {"client_id": "fake-client", "refresh_token": "PLACEHOLDER-NOT-A-REAL-TOKEN"}
ORIGINAL = "ORIGINAL-CIPHERTEXT-PLACEHOLDER\n"
REPLACEMENT = "NEW-CIPHERTEXT-PLACEHOLDER\n"


def _fake_sops(seen=None, hook=None):
    """Stand in for ``sops -e -i <tmp>``: overwrite the temp file, exit 0."""

    def _run(cmd, **kwargs):
        tmp = cmd[-1]
        if seen is not None:
            seen.append(tmp)
        if hook is not None:
            hook(tmp)
        Path(tmp).write_text(REPLACEMENT)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


@pytest.fixture
def enc_path(tmp_path):
    target = tmp_path / "secrets" / "youtube_upload_token.json.enc"
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL)
    return target


def _plaintext_strays(enc_path):
    """Temp files left behind — the lock sidecar is an intentional resident."""
    return [p.name for p in enc_path.parent.iterdir() if p.name.startswith(".sops-")]


def test_temp_file_is_created_in_the_targets_own_directory(enc_path, monkeypatch):
    """Same-filesystem staging is what makes ``os.replace`` atomic."""
    seen = []
    monkeypatch.setattr(uploader.subprocess, "run", _fake_sops(seen=seen))

    uploader.sops_encrypt_json(FAKE_PAYLOAD, enc_path)

    assert Path(seen[0]).parent == enc_path.parent
    assert enc_path.read_text() == REPLACEMENT


def test_original_survives_a_failure_between_write_and_install(enc_path, monkeypatch):
    """A crash after the ciphertext is written must leave the old file intact."""
    monkeypatch.setattr(uploader.subprocess, "run", _fake_sops())

    def _boom(src, dst):
        raise OSError("simulated crash between write and replace")

    monkeypatch.setattr(uploader.os, "replace", _boom)

    with pytest.raises(OSError):
        uploader.sops_encrypt_json(FAKE_PAYLOAD, enc_path)

    assert enc_path.read_text() == ORIGINAL
    assert _plaintext_strays(enc_path) == []


def test_file_mode_is_preserved_across_the_install(enc_path, monkeypatch):
    """The ``.enc`` is dev-owned; a 0600 regression locks the pipeline out."""
    os.chmod(enc_path, 0o640)
    monkeypatch.setattr(uploader.subprocess, "run", _fake_sops())

    uploader.sops_encrypt_json(FAKE_PAYLOAD, enc_path)

    assert stat.S_IMODE(os.stat(enc_path).st_mode) == 0o640


def test_write_holds_a_lock_other_processes_cannot_take(enc_path, monkeypatch):
    """flock is per open-file-description, so a second fd stands in for a second process."""
    lock_path = uploader._sops_lock_path(enc_path)
    blocked = []

    def _probe(_tmp):
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            blocked.append(True)
        finally:
            os.close(fd)

    monkeypatch.setattr(uploader.subprocess, "run", _fake_sops(hook=_probe))

    uploader.sops_encrypt_json(FAKE_PAYLOAD, enc_path)

    assert blocked == [True], "a concurrent writer could enter the critical section"


def test_two_concurrent_writers_serialize_instead_of_interleaving(enc_path, monkeypatch):
    events = []
    guard = threading.Lock()

    def _hook(_tmp):
        with guard:
            events.append("enter")
        time.sleep(0.05)
        with guard:
            events.append("exit")

    monkeypatch.setattr(uploader.subprocess, "run", _fake_sops(hook=_hook))

    threads = [
        threading.Thread(target=uploader.sops_encrypt_json, args=(FAKE_PAYLOAD, enc_path))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert events == ["enter", "exit", "enter", "exit"]
    assert enc_path.read_text() == REPLACEMENT


# --- recipient inheritance ----------
#
# The pipeline's sops config (GMS_SOPS_CONFIG) matches a seo-repo token only by
# its catch-all rule, so a re-mint silently switched the file to recipients the
# `dev` keyring cannot open. Re-encrypting a file must never change who can
# read it: the recipients come from the file being replaced, or explicitly.

_ENC_WITH_RECIPIENTS = (
    '{"data": "ENC[PLACEHOLDER]", "sops": {"age": ['
    '{"recipient": "age1aaaa", "enc": "x"}, {"recipient": "age1bbbb", "enc": "y"}]}}\n'
)


def _age_arg(cmd):
    return cmd[cmd.index("--age") + 1] if "--age" in cmd else None


def _capturing_sops(cmds):
    def _run(cmd, **kwargs):
        cmds.append(list(cmd))
        Path(cmd[-1]).write_text(REPLACEMENT)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


def test_fresh_file_without_recipients_falls_back_to_config_rules(tmp_path, monkeypatch):
    target = tmp_path / "new" / "token.json.enc"
    target.parent.mkdir()
    cmds = []
    monkeypatch.setattr(uploader.subprocess, "run", _capturing_sops(cmds))

    uploader.sops_encrypt_json(FAKE_PAYLOAD, target)

    assert _age_arg(cmds[0]) is None


def test_reencrypt_inherits_the_replaced_files_recipients(enc_path, monkeypatch):
    enc_path.write_text(_ENC_WITH_RECIPIENTS)
    cmds = []
    monkeypatch.setattr(uploader.subprocess, "run", _capturing_sops(cmds))

    uploader.sops_encrypt_json(FAKE_PAYLOAD, enc_path)

    assert _age_arg(cmds[0]) == "age1aaaa,age1bbbb"
    assert "--config" in cmds[0]  # config still passed; --age takes precedence in sops


def test_explicit_recipients_beat_the_existing_files(enc_path, monkeypatch):
    enc_path.write_text(_ENC_WITH_RECIPIENTS)
    cmds = []
    monkeypatch.setattr(uploader.subprocess, "run", _capturing_sops(cmds))

    uploader.sops_encrypt_json(FAKE_PAYLOAD, enc_path, age_recipients=["age1cccc"])

    assert _age_arg(cmds[0]) == "age1cccc"


def test_unwritable_target_directory_fails_before_any_plaintext_is_staged(
    enc_path, tmp_path, monkeypatch
):
    """No temp-dir fallback: a write that cannot succeed stages nothing and says why."""
    real_access = os.access
    monkeypatch.setattr(
        uploader.os,
        "access",
        lambda p, mode, *a, **k: (
            False
            if Path(p) == enc_path.parent and mode & os.X_OK
            else real_access(p, mode, *a, **k)
        ),
    )
    scratch_tmp = tmp_path / "scratch-tmp"
    scratch_tmp.mkdir()
    monkeypatch.setattr(uploader.tempfile, "tempdir", str(scratch_tmp))
    calls = []
    monkeypatch.setattr(
        uploader.subprocess, "run", lambda *a, **k: calls.append(a) or _fake_sops()(*a, **k)
    )

    with pytest.raises(PermissionError) as excinfo:
        uploader.sops_encrypt_json(FAKE_PAYLOAD, enc_path)

    assert "GMS_DRIVE_TOKEN_ENC" in str(excinfo.value)
    assert "GMS_DRIVE_CREDENTIALS_ENC" in str(excinfo.value)
    assert calls == [], "sops must not run when the install cannot succeed"
    assert list(scratch_tmp.iterdir()) == [], "no plaintext may be staged in the temp dir"
    assert enc_path.read_text() == ORIGINAL
    assert _plaintext_strays(enc_path) == []
