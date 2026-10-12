"""Config-driven LF description footer (playlist link, cadence, site).

The footer is keyed by the SAME derived region as playlist auto-assignment
(``region_resolver.derive_region`` off the canonical ResolvedLocation) and
renders between the CTA block and the hashtag divider. ALL config keys
default empty, so an unconfigured run produces a byte-identical description.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import longform_metadata
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

PLAYLIST_URL_IE = "https://www.youtube.com/playlist?list=PLireland"
PLAYLIST_URL_US = "https://www.youtube.com/playlist?list=PLusa"
CADENCE_IE = "New Ireland drive every Sunday"
CADENCE_US = "New USA drives Tuesday & Friday"
SITE_URL = "https://gomoveshift.com"


@pytest.fixture
def generator():
    """Generator with mocked SEO manager (no credentials needed)."""
    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator()
            gen._seo_degraded = True
            return gen


def _footer_config() -> Config:
    return Config.model_validate(
        {
            "drive": {
                "playlist_url_by_region": {
                    "ireland": PLAYLIST_URL_IE,
                    "usa": PLAYLIST_URL_US,
                },
                "cadence_line_by_region": {
                    "ireland": CADENCE_IE,
                    "usa": CADENCE_US,
                },
                "site_url": SITE_URL,
            }
        }
    )


def _patch_config(monkeypatch, cfg: Config) -> None:
    monkeypatch.setattr(longform_metadata, "_get_config", lambda config_path=None: cfg)


def _build(generator, location="County Kerry, Ireland", **kwargs) -> str:
    return generator._build_gms_description(
        location=location,
        seo_description="POV drive.",
        duration_str="26m 59s",
        publish_day="Sunday",
        **kwargs,
    )


class TestFooterConfigured:
    def test_footer_lines_render_above_hashtags(self, generator, monkeypatch):
        _patch_config(monkeypatch, _footer_config())
        desc = _build(generator)
        playlist_line = f"▶ Full Ireland playlist: {PLAYLIST_URL_IE}"
        site_line = f"🌍 {SITE_URL}"
        assert playlist_line in desc
        assert CADENCE_IE in desc
        assert site_line in desc
        hashtag_pos = desc.index("#CountyKerry")
        assert desc.index(playlist_line) < hashtag_pos
        assert desc.index(CADENCE_IE) < hashtag_pos
        assert desc.index(site_line) < hashtag_pos

    def test_usa_region_display_is_uppercase(self, generator, monkeypatch):
        _patch_config(monkeypatch, _footer_config())
        desc = _build(generator, location="Colleton County, SC")
        assert f"▶ Full USA playlist: {PLAYLIST_URL_US}" in desc
        assert CADENCE_US in desc

    def test_resolved_location_region_beats_label(self, generator, monkeypatch):
        """Region reuses the canonical resolver verdict, never a label re-parse."""
        _patch_config(monkeypatch, _footer_config())
        resolved = SimpleNamespace(is_us=True, country_code="US")
        desc = _build(generator, location="County Kerry, Ireland", resolved_location=resolved)
        assert f"▶ Full USA playlist: {PLAYLIST_URL_US}" in desc
        assert PLAYLIST_URL_IE not in desc

    def test_unmatched_region_gets_site_line_only(self, generator, monkeypatch):
        _patch_config(monkeypatch, _footer_config())
        desc = _build(generator, location="Provence, France")
        assert "▶ Full" not in desc
        assert CADENCE_IE not in desc
        assert CADENCE_US not in desc
        assert f"🌍 {SITE_URL}" in desc

    def test_chapters_still_embed_between_footer_and_hashtags(self, generator, monkeypatch):
        _patch_config(monkeypatch, _footer_config())
        desc = _build(generator)
        embedded = LongFormMetadataGenerator._embed_chapters(desc, "0:00 Start\n5:00 Killarney")
        chapters_pos = embedded.index("📍 CHAPTERS")
        assert embedded.index(f"▶ Full Ireland playlist: {PLAYLIST_URL_IE}") < chapters_pos
        assert chapters_pos < embedded.index("#CountyKerry")


class TestFooterUnconfigured:
    def test_default_config_is_byte_identical(self, generator, monkeypatch):
        """Empty defaults → no footer markers, CTA→divider→hashtags unchanged."""
        _patch_config(monkeypatch, Config())
        desc = _build(generator)
        assert "▶" not in desc
        assert "🌍" not in desc
        assert "💬 Comment your favorite moment!\n\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n#" in desc
