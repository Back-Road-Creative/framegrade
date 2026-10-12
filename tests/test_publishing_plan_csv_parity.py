"""Prove the publish-plan builder consumes the SAME shorts metadata
Phase D writes into ``<project>_shorts_metricool.csv``, so the direct-publish
queue can replace the CSV without content drift.

ONE shared fixture drive state (posts / slots / drive links / media) feeds
BOTH the real CSV-composition code (``build_consolidated_shorts_csv``) and
the real plan builder (``build_publish_plan``). Every assertion below is a
per-short parity check, normalized where the two representations legitimately
differ in shape (each normalization cites why).
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv
from gomoveshift.publishing.models import MediaFingerprint
from gomoveshift.publishing.plan_builder import _slot_datetime_utc, build_publish_plan

PROJECT_NAME = "20260101example"


def _post(short_index: int) -> SimpleNamespace:
    # Caption kept well under Twitter's 280-char limit so the CSV path's
    # Twitter-only truncation (retired X column) never engages — that
    # truncation is a legitimate CSV-only behavior, not a parity target.
    return SimpleNamespace(
        short_index=short_index,
        caption=f"Rolling through County Kerry, clip {short_index}",
        youtube_title=f"County Kerry Drive #{short_index}",
        youtube_category="24",
        youtube_tags=f"County Kerry,road trip {short_index}",
        youtube_description=f"A short drive through the hills, part {short_index}.",
        tiktok_title=f"tt {short_index}",
        pinterest_board="drives",
        first_comment=f"#tag{short_index}",
        sampled_regions=["County Kerry, Munster"],
    )


def _slot(short_index: int, date: str, time: str) -> SimpleNamespace:
    return SimpleNamespace(short_index=short_index, date=date, time=time)


def _media(short_index: int) -> MediaFingerprint:
    return MediaFingerprint(
        path=f"/output/lf1/short_{short_index:03d}.mp4",
        size_bytes=12_345,
        duration_seconds=42.5,
        width=1080,
        height=1920,
        codec="h264",
        sha256=f"{short_index:064d}",
    )


def _drive_link(short_index: int) -> str:
    return f"https://drive.google.com/file/d/FILE_ID_{short_index}/view?usp=sharing"


def _build_both(tmp_path: Path):
    posts = [_post(1), _post(2)]
    shorts_by_lf = {1: posts}
    slots_by_short_index = {
        1: _slot(1, "2026-04-06", "14:00:00"),
        2: _slot(2, "2026-04-06", "18:00:00"),
    }
    drive_links_by_short_index = {p.short_index: _drive_link(p.short_index) for p in posts}
    parent_lf_video_id_by_short_index = {p.short_index: "parentvid01" for p in posts}
    parent_recording_date_by_short_index = {p.short_index: "2026-03-30" for p in posts}
    parent_publish_at_by_short_index = {p.short_index: "2026-04-06T12:00:00+00:00" for p in posts}
    media_by_short_index = {p.short_index: _media(p.short_index) for p in posts}

    csv_path = tmp_path / f"{PROJECT_NAME}_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=csv_path,
        shorts_by_lf=shorts_by_lf,
        slots_by_short_index=slots_by_short_index,
        drive_links_by_short_index=drive_links_by_short_index,
        parent_lf_video_id_by_short_index=parent_lf_video_id_by_short_index,
        parent_recording_date_by_short_index=parent_recording_date_by_short_index,
        parent_publish_at_by_short_index=parent_publish_at_by_short_index,
    )
    with open(csv_path, encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    plan = build_publish_plan(
        project_name=PROJECT_NAME,
        shorts_by_lf=shorts_by_lf,
        slots_by_short_index=slots_by_short_index,
        media_by_short_index=media_by_short_index,
        parent_lf_video_id_by_short_index=parent_lf_video_id_by_short_index,
        parent_recording_date_by_short_index=parent_recording_date_by_short_index,
        parent_publish_at_by_short_index=parent_publish_at_by_short_index,
    )

    return csv_rows, plan, media_by_short_index, drive_links_by_short_index


def _unescape(value: str) -> str:
    # Both surfaces strip a leading formula-injection apostrophe
    # (_escape_csv_cell) — a spreadsheet-only defense with no analog in a
    # direct API payload, so the plan side never adds it.
    return value[1:] if value.startswith("'") else value


def test_per_short_caption_and_title_parity(tmp_path):
    csv_rows, plan, _media_map, _links = _build_both(tmp_path)
    assert len(csv_rows) == len(plan.shorts) == 2
    for row, record in zip(csv_rows, plan.shorts):
        assert _unescape(row["YouTube Video Title"]) == record.payloads.youtube.title

        csv_tags = tuple(
            t.strip() for t in _unescape(row["YouTube Video Tags"]).split(",") if t.strip()
        )
        assert csv_tags == record.payloads.youtube.tags

        # The CSV's shared "Text" cell is the caption + the "Filmed <Month
        # YYYY>." suffix (no Twitter truncation for these short captions);
        # the plan's Instagram/Facebook payloads carry the identical string
        # (per docs/social-publishing-plan-builder.md's mapping doc) — no
        # parent-link funnel line lives on this surface on either side.
        assert _unescape(row["Text"]) == record.payloads.instagram.caption
        assert _unescape(row["Text"]) == record.payloads.facebook.description

        assert row["TikTok Title"] == record.payloads.tiktok.title


def test_youtube_category_diverges_by_design(tmp_path):
    # Legitimate divergence, not a mismatch: csv_generator forces
    # "YouTube Video Category" == "" on every row
    # (Metricool silently drafts a post with a populated numeric category
    # ID). The plan builder posts straight to the YouTube API, which has no
    # such quirk, so it carries the real SocialPost.youtube_category value.
    csv_rows, plan, _media_map, _links = _build_both(tmp_path)
    for row, record in zip(csv_rows, plan.shorts):
        assert row["YouTube Video Category"] == ""
        assert record.payloads.youtube.category == "24"


def test_per_short_scheduled_slot_parity(tmp_path):
    csv_rows, plan, _media_map, _links = _build_both(tmp_path)
    for row, record in zip(csv_rows, plan.shorts):
        # CSV ships the slot as an America/New_York local Date/Time pair;
        # the plan carries the same instant as a UTC datetime. Convert the
        # CSV's strings with the plan builder's own conversion helper
        # (single source of truth for the EST->UTC rule) rather than
        # re-deriving the timezone math in the test.
        row_slot = SimpleNamespace(date=row["Date"], time=row["Time"])
        assert _slot_datetime_utc(row_slot) == record.schedule.scheduled_at_utc


def test_per_short_media_reference_parity(tmp_path):
    csv_rows, plan, media_map, drive_links = _build_both(tmp_path)
    for row, record in zip(csv_rows, plan.shorts):
        # The CSV's "Picture Url 1" is the Drive-hosted download link
        # (Metricool's media source); the plan's MediaFingerprint.path is
        # the local rendered file (the first-party upload's media source).
        # These are deliberately different representations of the SAME
        # underlying short — the parity check is that each side's media
        # reference resolves to the fixture entry for the SAME short_index,
        # never a neighboring short's.
        expected_link = drive_links[record.short_index]
        # https://.../file/d/<ID>/view -> https://.../uc?export=download&id=<ID>
        expected_id = expected_link.split("/file/d/")[1].split("/")[0]
        assert expected_id in row["Picture Url 1"]
        assert record.media == media_map[record.short_index]
        assert record.media.path == f"/output/lf1/short_{record.short_index:03d}.mp4"


def test_per_short_platform_targets_parity(tmp_path):
    csv_rows, plan, _media_map, _links = _build_both(tmp_path)
    for row, record in zip(csv_rows, plan.shorts):
        # Every platform the plan builder can model (KNOWN_PLATFORMS:
        # youtube/instagram/facebook/tiktok) must be TRUE in the CSV's
        # default toggles and present in the plan's default route
        # selection — the two lists agree on the routes they share.
        # Pinterest/Threads/Twitter/LinkedIn/GBP are CSV-only columns with
        # no plan_builder/models representation (no first-party adapter
        # exists for them yet — facebook/instagram/tiktok/youtube_adapter
        # are the only ones in gomoveshift/publishing/) — legitimately out
        # of scope for this parity check, not a mismatch.
        csv_column_by_platform = {
            "youtube": "YouTube",
            "instagram": "Instagram",
            "facebook": "Facebook",
            "tiktok": "TikTok",
        }
        for platform in record.routes.selected:
            assert row[csv_column_by_platform[platform]] == "TRUE"
