"""Per-short region gate on the shipped Metricool CSV surfaces.

A short's title/caption/tags publish verbatim from the consolidated Metricool
CSV. Each must describe *the short's own footage* — a county named on any
surface that is absent from the clip's sampled regions is a cross-region desync
(the far-corner class; the parent-LF county leaking into a short cut from a
different county). The gate fires UPSTREAM of the irreversible CSV write.

RED-first proof that ``build_consolidated_shorts_csv`` (the drive-path CSV
writer) region-checks title, caption, AND tags against each short's own regions
— while never false-failing an in-region short whose prose merely mentions its
own county.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv


def _make_post(
    short_index: int,
    *,
    sampled_regions,
    youtube_title="County Kerry Drive",
    caption="Rolling through County Kerry today",
    youtube_tags="County Kerry,road trip",
):
    return SimpleNamespace(
        short_index=short_index,
        caption=caption,
        youtube_title=youtube_title,
        youtube_category="19",
        youtube_tags=youtube_tags,
        tiktok_title=f"tt {short_index}",
        pinterest_board="drives",
        first_comment=f"#tag{short_index}",
        sampled_regions=sampled_regions,
    )


def _slot(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        date="2026-04-01",
        time="16:00:00",
        datetime_est="2026-04-01T16:00:00-04:00",
        day_of_week="Wednesday",
    )


def _write(out: Path, post):
    return build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [post]},
        slots_by_short_index={post.short_index: _slot(post.short_index)},
        drive_links_by_short_index={post.short_index: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={post.short_index: "abc"},
        parent_recording_date_by_short_index={post.short_index: "2026-04-01"},
    )


def test_cross_region_tag_hard_fails_before_csv_write(tmp_path: Path):
    """A short whose TAGS carry a county absent from its regions hard-fails."""
    out = tmp_path / "p_shorts_metricool.csv"
    post = _make_post(
        1,
        sampled_regions=["County Kerry, Munster"],
        youtube_tags="County Cork,road trip",  # Cork is NOT in the clip's Kerry regions
    )
    with pytest.raises(ValueError, match="(?i)region") as exc:
        _write(out, post)
    assert "tag" in str(exc.value)
    assert not out.exists(), "CSV must not be written when the region gate fails"


def test_cross_region_title_hard_fails_on_drive_path(tmp_path: Path):
    """A cross-region county in the short's TITLE hard-fails on the drive path.

    The parent county rides both the ``{county}`` title template and the tag
    list, so the shipped title is region-checked (``build_consolidated_shorts_csv``
    is the drive-path CSV writer). Proves social surfaces reach the gate.
    """
    out = tmp_path / "p_shorts_metricool.csv"
    post = _make_post(
        1,
        sampled_regions=["County Kerry, Munster"],
        youtube_title="County Cork Drive — County Kerry, Munster",
        youtube_tags="County Cork,County Kerry,road trip",
    )
    with pytest.raises(ValueError, match="(?i)region") as exc:
        _write(out, post)
    assert "title" in str(exc.value)
    assert not out.exists()


def test_in_region_short_passes(tmp_path: Path):
    """An all-in-region short writes the CSV normally."""
    out = tmp_path / "p_shorts_metricool.csv"
    post = _make_post(1, sampled_regions=["County Kerry, Munster"])
    _write(out, post)
    assert out.exists()


def test_in_region_prose_does_not_false_fail(tmp_path: Path):
    """An in-region short whose prose title/caption mention its OWN county must
    pass — the label regex misfires on prose, so the gate must not fire on it."""
    out = tmp_path / "p_shorts_metricool.csv"
    post = _make_post(
        1,
        sampled_regions=["County Kerry, Munster"],
        youtube_title="POV: On the Roads of County Kerry",
        caption="Just driving through County Kerry\n\nRolling through County Kerry",
        youtube_tags="County Kerry,road trip,dashcam",
    )
    _write(out, post)
    assert out.exists()


def test_gps_less_short_is_noop(tmp_path: Path):
    """A GPS-less/legacy short (no sampled regions) is a no-op pass even with
    a cross-region-looking county on its surfaces."""
    out = tmp_path / "p_shorts_metricool.csv"
    post = _make_post(
        1,
        sampled_regions=[],
        youtube_title="County Cork Drive",
        youtube_tags="County Cork,road trip",
    )
    _write(out, post)
    assert out.exists()


def test_social_post_carries_sampled_regions():
    """``generate_all`` stamps each post with its own resolved region label so
    the CSV gate has a per-short region reference. This also fixes
    the leak at the SOURCE: a short cut in a DIFFERENT county no longer inherits
    the parent-LF county, so the parent county can no longer ride into that
    short's tags (the CSV county gate remains as defence-in-depth)."""
    from gomoveshift.pipeline.social_generator import SocialGenerator

    gen = SocialGenerator(
        location="County Cork, Munster",  # parent LF county
        project_name="proj",
        series_part=1,
        county="County Cork",  # parent county — must NOT ride onto a far-corner short
    )
    posts = gen.generate_all(
        num_shorts=1,
        per_short_locations=["County Kerry, Munster"],  # this short's OWN region
    )
    post = posts[0]
    assert post.sampled_regions == ["County Kerry, Munster"]
    # The parent county no longer leaks into a cross-county short's tags.
    assert "County Cork" not in post.youtube_tags
