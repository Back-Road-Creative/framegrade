"""No test in this suite may ever resolve a pipeline secret path onto a real
credential directory.

Root cause of the original hazard: a module that binds
``GMS_DRIVE_CREDENTIALS_ENC`` / ``GMS_DRIVE_TOKEN_ENC`` /
``GMS_AGE_KEY_FILE`` once at import time (for example as *default argument
values*) can never be redirected by a per-test
``monkeypatch.setenv`` -- pytest imports
every test module (and any module it imports) during collection, which
completes before any fixture, autouse or not, ever runs. ``tests/conftest.py``
sets these three env vars as plain module-level code, before its own
``import gomoveshift``, so the session-wide override protects anything
imported during collection. An autouse fixture then layers a fresh
``tmp_path`` per test on top; that per-test value is what both the console
and the uploader resolve at call time.

A runner without read access to the real credential directory would
otherwise fail with ``PermissionError``.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from gomoveshift.console import settings as console_settings
from gomoveshift.paths import gms_secret_path
from gomoveshift.pipeline import uploader

SECRET_ENV_NAMES = ("GMS_DRIVE_CREDENTIALS_ENC", "GMS_DRIVE_TOKEN_ENC", "GMS_AGE_KEY_FILE")
ALL_SECRET_ENV_NAMES = (*SECRET_ENV_NAMES, "GMS_SOPS_CONFIG")


# ``conftest.py`` creates the throwaway secrets directory with this prefix.
_THROWAWAY_DIR_MARKER = "gms-test-secrets-"


def test_env_vars_point_at_the_throwaway_secrets_dir():
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        assert value, f"{name} must be set for every test (see tests/conftest.py)"
        assert _THROWAWAY_DIR_MARKER in value, f"{name}={value} is not a throwaway test path"


def test_uploader_module_constants_point_at_the_throwaway_secrets_dir():
    """The uploader's names resolve to a throwaway path (the per-test override
    layered on conftest.py's session-wide one), never to a real credential
    directory."""
    assert _THROWAWAY_DIR_MARKER in str(uploader.DEFAULT_CREDENTIALS_ENC)
    assert _THROWAWAY_DIR_MARKER in str(uploader.DEFAULT_TOKEN_ENC)
    assert _THROWAWAY_DIR_MARKER in str(uploader._AGE_KEY_FILE)


def test_settings_resolves_through_same_env_names_as_uploader():
    resolved = console_settings.resolved_secret_paths()
    for name in SECRET_ENV_NAMES:
        assert resolved[name] == Path(os.environ[name])


def test_unset_env_defaults_agree_between_settings_and_uploader(tmp_path):
    """With the four env vars unset, the console resolves the same default paths
    the uploader resolves, SOPS config included. Runs in a clean child
    process: this one has the throwaway overrides already bound."""
    probe = (
        "import json\n"
        "from gomoveshift.console import settings\n"
        "from gomoveshift.pipeline import uploader\n"
        "r = {k: str(v) for k, v in settings.resolved_secret_paths().items()}\n"
        "print(json.dumps({\n"
        "  'settings': r,\n"
        "  'uploader': {\n"
        "    'GMS_DRIVE_CREDENTIALS_ENC': str(uploader.DEFAULT_CREDENTIALS_ENC),\n"
        "    'GMS_DRIVE_TOKEN_ENC': str(uploader.DEFAULT_TOKEN_ENC),\n"
        "    'GMS_AGE_KEY_FILE': str(uploader._AGE_KEY_FILE),\n"
        "    'GMS_SOPS_CONFIG': str(uploader._SOPS_CONFIG),\n"
        "  },\n"
        "}))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ALL_SECRET_ENV_NAMES}
    env["GMS_CONFIG_HOME"] = str(tmp_path / "cfg")
    done = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["settings"] == out["uploader"]
    assert out["settings"]["GMS_AGE_KEY_FILE"] == str(tmp_path / "cfg" / "age.key")
    assert out["settings"]["GMS_SOPS_CONFIG"] == str(tmp_path / "cfg" / ".sops.yaml")


def test_sops_config_override_is_resolved(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SOPS_CONFIG", str(tmp_path / "custom.sops.yaml"))
    assert console_settings.resolved_secret_paths()["GMS_SOPS_CONFIG"] == (
        tmp_path / "custom.sops.yaml"
    )


@pytest.mark.parametrize("name", ALL_SECRET_ENV_NAMES)
def test_empty_override_counts_as_unset(monkeypatch, tmp_path, name):
    """An empty value resolves to the config-home default, never to an empty path."""
    monkeypatch.setenv("GMS_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv(name, "")
    resolved = console_settings.resolved_secret_paths()[name]
    assert resolved.parent == tmp_path / "cfg"
    assert resolved.name


@pytest.mark.parametrize("name", ALL_SECRET_ENV_NAMES)
def test_override_set_after_import_is_seen(monkeypatch, tmp_path, name):
    target = tmp_path / "late" / "file"
    monkeypatch.setenv(name, str(target))
    assert console_settings.resolved_secret_paths()[name] == target
    assert gms_secret_path(name, "ignored") == target


@pytest.mark.parametrize(
    ("name", "attr"),
    [
        ("GMS_DRIVE_CREDENTIALS_ENC", "DEFAULT_CREDENTIALS_ENC"),
        ("GMS_DRIVE_TOKEN_ENC", "DEFAULT_TOKEN_ENC"),
        ("GMS_AGE_KEY_FILE", "_AGE_KEY_FILE"),
        ("GMS_SOPS_CONFIG", "_SOPS_CONFIG"),
    ],
)
def test_uploader_sees_override_set_after_import(monkeypatch, tmp_path, name, attr):
    """The uploader resolves its secret paths at call time, like the console."""
    target = tmp_path / "late" / "file"
    monkeypatch.setenv(name, str(target))
    assert Path(getattr(uploader, attr)) == target

    # The constructor defaults and the sops helpers use the late value too.
    if attr in ("DEFAULT_CREDENTIALS_ENC", "DEFAULT_TOKEN_ENC"):
        instance = uploader.PipelineUploader()
        used = instance.credentials_enc if attr == "DEFAULT_CREDENTIALS_ENC" else instance.token_enc
        assert used == target
    else:
        enc = tmp_path / "x.enc"
        enc.write_text("{}")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["env"] = kwargs["env"]
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

        monkeypatch.setattr(uploader.subprocess, "run", fake_run)
        uploader.sops_decrypt_json(enc)
        if attr == "_AGE_KEY_FILE":
            assert seen["env"]["SOPS_AGE_KEY_FILE"] == str(target)
        else:
            assert seen["cmd"][seen["cmd"].index("--config") + 1] == str(target)


def test_reproduces_original_permission_error_without_root(tmp_path):
    """Root cause: ``Path.exists()`` only swallows ENOENT/ENOTDIR/EBADF/ELOOP
    -- NOT ``PermissionError`` (EACCES) -- so a credential path under a
    directory the process can't traverse raises instead of returning
    False. Reproduced here without root by locking down our own tmp dir,
    the same shape as a runner's unreadable credential directory.
    """
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    unreadable = locked_dir / "drive_credentials.json.enc"
    unreadable.write_bytes(b"x")
    os.chmod(locked_dir, 0o000)
    try:
        try:
            unreadable.exists()
        except PermissionError:
            pass
        else:
            raise AssertionError(
                "expected PermissionError reproducing the unreadable-directory failure; "
                "if this environment has elevated access, the reproduction can't "
                "run here (report this rather than weakening the assertion)"
            )
    finally:
        os.chmod(locked_dir, stat.S_IRWXU)
