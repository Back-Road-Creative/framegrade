"""YouTube tracking must find the same credentials every other consumer finds.

`gomoveshift.pipeline._yt_creds` exists so "both analytics sync and YouTube upload
can share a single source of truth" (its own docstring). The tracking collector
did not use it. It resolved its own token path from `SEOConfig()` *dataclass
defaults* rather than the loaded config, so `manager_path` was the default "."
and the token resolved to the bare relative string `youtube_credentials.json` —
whatever the working directory happened to be.

The real credential is `youtube_upload_token.json.enc`, SOPS-encrypted, beside
`meta_publishing.json.enc` and `tiktok_publishing.json.enc`. Different directory,
different name, and encrypted, so the plaintext path could never match it.

Nothing surfaced this: `is_available()` returns False for a missing file, and a
False there is indistinguishable from a deliberate opt-out, so YouTube metrics
were silently absent rather than reported as broken.
"""

from pathlib import Path

import pytest

from gomoveshift.tracking import collector as mod


def test_the_default_token_path_comes_from_the_loaded_config_not_the_dataclass():
    """`SEOConfig()` defaults `manager_path` to "." on purpose — every deployment
    overrides it in config.yaml. Reading the default instead of the loaded value
    makes the path cwd-relative, which is never right."""
    client_secret, token = mod._resolve_default_cred_paths()
    assert token.is_absolute(), f"token path is cwd-relative: {token}"
    assert client_secret.is_absolute(), f"client secret path is cwd-relative: {client_secret}"


def test_a_sops_only_install_still_reports_youtube_as_available(monkeypatch, tmp_path):
    """The plaintext token file is optional. When only the SOPS store is present —
    which is the real deployment — the collector must still report available, via
    the shared loader rather than its own path guess."""
    calls = []

    def fake_load(credentials_dir=None, brand=None):
        calls.append(credentials_dir)
        return {"token": "t", "refresh_token": "r", "client_id": "c", "client_secret": "s"}

    monkeypatch.setattr(mod, "_load_shared_youtube_credentials", fake_load, raising=False)
    c = mod.YouTubeCollector(token_path=tmp_path / "definitely-absent.json")
    assert c.is_available() is True
    assert calls, "the shared loader was never consulted"


def test_no_credentials_anywhere_is_still_unavailable(monkeypatch, tmp_path):
    """The negative control: the fallback must not turn every install available."""

    def fake_load(credentials_dir=None, brand=None):
        raise FileNotFoundError("no store here")

    monkeypatch.setattr(mod, "_load_shared_youtube_credentials", fake_load, raising=False)
    c = mod.YouTubeCollector(token_path=tmp_path / "definitely-absent.json")
    assert c.is_available() is False


def test_the_shared_loader_is_only_asked_once(monkeypatch, tmp_path):
    """SOPS decrypt is a subprocess. `is_available()` is called per collection
    pass, so a cache miss that re-shells every time is a real cost."""
    calls = []

    def fake_load(credentials_dir=None, brand=None):
        calls.append(1)
        return {"token": "t"}

    monkeypatch.setattr(mod, "_load_shared_youtube_credentials", fake_load, raising=False)
    c = mod.YouTubeCollector(token_path=tmp_path / "absent.json")
    assert c.is_available() is True
    assert c.is_available() is True
    assert len(calls) == 1, f"shared loader ran {len(calls)} times"


@pytest.mark.parametrize("path_attr", ["_client_secret", "_token_path"])
def test_an_explicit_path_still_wins(tmp_path, path_attr):
    """An explicit constructor path is an override and must not be second-guessed."""
    p = tmp_path / "explicit.json"
    c = mod.YouTubeCollector(client_secret_path=p, token_path=p)
    assert getattr(c, path_attr) == Path(p)
