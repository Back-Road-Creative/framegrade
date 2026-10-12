"""TDD: the Metricool shorts CSV must never post a platform that is on the
``first_party`` publishing route -- Phase D's publish-plan sidecar already
posts it directly, so a live Metricool toggle for that
same platform double-posts, and for Instagram also burns the shared
Content Publishing API quota (docs/social-publishing-meta.md).

``platforms_for_publishing_routes`` is the robust-by-construction fix: it
derives the CSV's per-platform toggle dict from ``DriveConfig``, forcing any
``first_party``-routed platform's toggle to ``"FALSE"`` so the generator
cannot emit a live-toggle row for a platform it doesn't own.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

from gomoveshift.pipeline.csv_generator import (
    DEFAULT_PLATFORMS,
    build_consolidated_shorts_csv,
    platforms_for_publishing_routes,
)


def _drive_cfg(**routes):
    defaults = dict(
        publishing_route_youtube="metricool",
        publishing_route_instagram="metricool",
        publishing_route_facebook="metricool",
        publishing_route_tiktok="metricool",
    )
    defaults.update(routes)
    return SimpleNamespace(**defaults)


class TestPlatformsForPublishingRoutes:
    def test_all_metricool_returns_defaults_unchanged(self):
        assert platforms_for_publishing_routes(_drive_cfg()) == DEFAULT_PLATFORMS

    def test_none_config_returns_defaults_unchanged(self):
        assert platforms_for_publishing_routes(None) == DEFAULT_PLATFORMS

    def test_youtube_first_party_forces_youtube_false_only(self):
        plat = platforms_for_publishing_routes(_drive_cfg(publishing_route_youtube="first_party"))
        assert plat["YouTube"] == "FALSE"
        for col, val in DEFAULT_PLATFORMS.items():
            if col != "YouTube":
                assert plat[col] == val

    def test_instagram_first_party_forces_instagram_false_only(self):
        plat = platforms_for_publishing_routes(_drive_cfg(publishing_route_instagram="first_party"))
        assert plat["Instagram"] == "FALSE"
        for col, val in DEFAULT_PLATFORMS.items():
            if col != "Instagram":
                assert plat[col] == val

    def test_tiktok_first_party_forces_tiktok_false_only(self):
        plat = platforms_for_publishing_routes(_drive_cfg(publishing_route_tiktok="first_party"))
        assert plat["TikTok"] == "FALSE"
        for col, val in DEFAULT_PLATFORMS.items():
            if col != "TikTok":
                assert plat[col] == val


def _base_kwargs(tmp_path, **overrides):
    slot = SimpleNamespace(date="2026-03-15", time="08:00:00")
    kwargs = dict(
        output_path=tmp_path / "out.csv",
        shorts_by_lf={
            1: [
                SimpleNamespace(
                    short_index=10, caption="hi", first_comment="", tiktok_title="tt-10"
                )
            ]
        },
        slots_by_short_index={10: slot},
        drive_links_by_short_index={10: "https://drive.example/10"},
        parent_lf_video_id_by_short_index={},
        parent_recording_date_by_short_index={10: "2026-03-15"},
    )
    kwargs.update(overrides)
    return kwargs


def _read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestCsvHonoursPlatformsOverride:
    def test_default_metricool_output_unchanged(self, tmp_path):
        path = build_consolidated_shorts_csv(**_base_kwargs(tmp_path))
        rows = _read_rows(path)
        assert rows[0]["YouTube"] == "TRUE"
        assert rows[0]["Instagram"] == "TRUE"
        assert rows[0]["TikTok"] == "TRUE"
        assert rows[0]["TikTok Title"] != ""

    def test_youtube_first_party_route_suppresses_youtube_toggle(self, tmp_path):
        plat = platforms_for_publishing_routes(_drive_cfg(publishing_route_youtube="first_party"))
        path = build_consolidated_shorts_csv(**_base_kwargs(tmp_path, platforms=plat))
        rows = _read_rows(path)
        assert rows[0]["YouTube"] == "FALSE"
        assert rows[0]["Instagram"] == "TRUE"

    def test_tiktok_first_party_route_also_empties_tiktok_specific_columns(self, tmp_path):
        plat = platforms_for_publishing_routes(_drive_cfg(publishing_route_tiktok="first_party"))
        path = build_consolidated_shorts_csv(**_base_kwargs(tmp_path, platforms=plat))
        rows = _read_rows(path)
        assert rows[0]["TikTok"] == "FALSE"
        assert rows[0]["TikTok Title"] == ""
        assert rows[0]["TikTok Post Privacy"] == ""
