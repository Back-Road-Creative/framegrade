"""TDD for the brand-scoped credential path resolver."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.publishing.credential_paths import resolve_credentials_dir, resolve_sops_file


def test_brand_dir_exists_wins(tmp_path):
    base = tmp_path / "credentials"
    brand_dir = base / "brands" / "acme" / "creds"
    brand_dir.mkdir(parents=True)

    assert resolve_credentials_dir("acme", base) == brand_dir


def test_brand_dir_missing_falls_back_to_base(tmp_path):
    base = tmp_path / "credentials"
    base.mkdir()

    assert resolve_credentials_dir("acme", base) == base


def test_brand_none_falls_back_to_base(tmp_path):
    base = tmp_path / "credentials"
    base.mkdir()

    assert resolve_credentials_dir(None, base) == base


def test_resolve_sops_file_shared_layout(tmp_path):
    base = tmp_path / "credentials"
    base.mkdir()

    result = resolve_sops_file("meta_publishing.json.enc", None, base)

    assert result == base.parent / "meta_publishing.json.enc"


def test_resolve_sops_file_brand_layout(tmp_path):
    base = tmp_path / "credentials"
    brand_dir = base / "brands" / "acme" / "creds"
    brand_dir.mkdir(parents=True)

    result = resolve_sops_file("meta_publishing.json.enc", "acme", base)

    assert result == base / "brands" / "acme" / "meta_publishing.json.enc"


def test_resolve_credentials_dir_default_base_reads_config(monkeypatch, tmp_path):
    """``base=None`` reads ``seo.credentials_dir`` from config — the same
    default every loader already had before this module existed."""
    fake_base = tmp_path / "credentials"

    class _FakeSeo:
        credentials_dir = str(fake_base)

    class _FakeConfig:
        seo = _FakeSeo()

    monkeypatch.setattr("gomoveshift.config.get_config", lambda: _FakeConfig(), raising=True)

    result = resolve_credentials_dir(None)

    assert result == fake_base
    assert isinstance(result, Path)


def _fake_sops_decrypt(payload: dict):
    def _run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    return _run


class TestLoadersReadBrandScopedDir:
    """``brand=`` on each loader reads the brand-scoped ``.enc`` path;
    ``brand=None`` stays byte-for-byte the pre-brand shared path."""

    def test_load_meta_credentials_brand_scoped(self, monkeypatch, tmp_path):
        from gomoveshift.publishing import meta_auth

        base = tmp_path / "credentials"
        brand_dir = base / "brands" / "acme" / "creds"
        brand_dir.mkdir(parents=True)
        (base / "brands" / "acme" / meta_auth._META_CREDENTIALS_FILENAME).write_text("x")

        monkeypatch.setattr(meta_auth.subprocess, "run", _fake_sops_decrypt({"page_id": "acme"}))

        creds = meta_auth.load_meta_credentials(base, brand="acme")

        assert creds == {"page_id": "acme"}

    def test_load_meta_credentials_brand_none_unchanged(self, monkeypatch, tmp_path):
        from gomoveshift.publishing import meta_auth

        base = tmp_path / "credentials"
        base.mkdir()
        (tmp_path / meta_auth._META_CREDENTIALS_FILENAME).write_text("x")

        monkeypatch.setattr(meta_auth.subprocess, "run", _fake_sops_decrypt({"page_id": "shared"}))

        creds = meta_auth.load_meta_credentials(base)

        assert creds == {"page_id": "shared"}

    def test_load_tiktok_credentials_brand_scoped(self, monkeypatch, tmp_path):
        from gomoveshift.publishing import tiktok_auth

        base = tmp_path / "credentials"
        brand_dir = base / "brands" / "acme" / "creds"
        brand_dir.mkdir(parents=True)
        (base / "brands" / "acme" / tiktok_auth._TIKTOK_CREDENTIALS_FILENAME).write_text("x")

        monkeypatch.setattr(
            tiktok_auth.subprocess, "run", _fake_sops_decrypt({"access_token": "acme-tok"})
        )

        creds = tiktok_auth.load_tiktok_credentials(base, brand="acme")

        assert creds == {"access_token": "acme-tok"}

    def test_get_valid_tiktok_credentials_brand_scoped(self, monkeypatch, tmp_path):
        from gomoveshift.publishing import tiktok_auth

        base = tmp_path / "credentials"
        brand_dir = base / "brands" / "acme" / "creds"
        brand_dir.mkdir(parents=True)
        # A fixed moment past any plausible run date: the test asserts the
        # brand-scoped path resolves, not the refresh arithmetic.
        far_future = "2099-01-01T00:00:00+00:00"
        (base / "brands" / "acme" / tiktok_auth._TIKTOK_CREDENTIALS_FILENAME).write_text("x")

        monkeypatch.setattr(
            tiktok_auth.subprocess,
            "run",
            _fake_sops_decrypt({"access_token": "acme-tok", "access_expires_at": far_future}),
        )

        creds = tiktok_auth.get_valid_tiktok_credentials(base, brand="acme")

        assert creds["access_token"] == "acme-tok"  # noqa: S105 — test fixture, not a secret

    def test_load_youtube_credentials_brand_scoped(self, monkeypatch, tmp_path):
        from gomoveshift.pipeline import _yt_creds

        base = tmp_path / "credentials"
        brand_dir = base / "brands" / "acme" / "creds"
        brand_dir.mkdir(parents=True)
        (base / "brands" / "acme" / "youtube_upload_token.json.enc").write_text("x")

        monkeypatch.setattr(_yt_creds.subprocess, "run", _fake_sops_decrypt({"token": "acme-yt"}))

        creds = _yt_creds.load_youtube_credentials(base, brand="acme")

        assert creds == {"token": "acme-yt"}

    def test_load_youtube_credentials_brand_none_unchanged(self, monkeypatch, tmp_path):
        from gomoveshift.pipeline import _yt_creds

        base = tmp_path / "credentials"
        base.mkdir()
        (tmp_path / "youtube_upload_token.json.enc").write_text("x")

        monkeypatch.setattr(_yt_creds.subprocess, "run", _fake_sops_decrypt({"token": "shared-yt"}))

        creds = _yt_creds.load_youtube_credentials(base)

        assert creds == {"token": "shared-yt"}


class TestMetricsCollectorThreadsBrand:
    """The console/dispatch seam: ``MetricsCollector(brand=...)`` reaches
    every platform's credential loader, so the metrics CLI's already-known
    ``current_brand()`` actually selects a brand-scoped store."""

    def test_tiktok_collector_passes_brand(self, monkeypatch):
        from gomoveshift.tracking import collector as collector_mod

        seen = {}

        def _fake(credentials_dir=None, brand=None):
            seen["brand"] = brand
            return {"access_token": "tok"}

        monkeypatch.setattr(collector_mod, "get_valid_tiktok_credentials", _fake)

        tk = collector_mod.TikTokCollector(brand="acme")
        tk._access_token()

        assert seen["brand"] == "acme"

    def test_meta_collectors_pass_brand(self, monkeypatch):
        from gomoveshift.tracking import collector as collector_mod

        seen = {}

        def _fake(credentials_dir=None, brand=None):
            seen["brand"] = brand
            return {}

        monkeypatch.setattr(collector_mod, "load_meta_credentials", _fake)

        collector_mod.InstagramCollector(brand="acme")._creds()
        assert seen["brand"] == "acme"

        seen.clear()
        collector_mod.FacebookCollector(brand="acme")._creds()
        assert seen["brand"] == "acme"

    def test_youtube_collector_passes_brand(self, monkeypatch):
        from gomoveshift.tracking import collector as collector_mod

        seen = {}

        def _fake(credentials_dir=None, brand=None):
            seen["brand"] = brand
            return {"token": "tok"}

        monkeypatch.setattr(collector_mod, "_load_shared_youtube_credentials", _fake)

        yt = collector_mod.YouTubeCollector(brand="acme")
        yt._shared_credentials()

        assert seen["brand"] == "acme"

    def test_metrics_collector_threads_brand_to_children(self):
        from gomoveshift.tracking.collector import MetricsCollector

        mc = MetricsCollector(brand="acme")

        assert mc.tiktok._brand == "acme"
        assert mc.instagram._brand == "acme"
        assert mc.facebook._brand == "acme"
        assert mc.youtube._brand == "acme"

    def test_metrics_collector_brand_default_none(self):
        from gomoveshift.tracking.collector import MetricsCollector

        mc = MetricsCollector()

        assert mc.tiktok._brand is None
