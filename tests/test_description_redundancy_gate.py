"""TDD tests for the Phase D hard-fail description-redundancy gate.

Locks in the cleaned-description contract by construction. The gate raises at Phase D
BEFORE any ``<project>_lf01_youtube.json`` write so a redundant
description is unrepresentable on disk.

Allowed-occurrences table (summary):
- ``recording_date`` (e.g. ``"2025-09-14"``): max 1× — ``📅 Filmed:`` line.
- ``location`` exact string: max 2× — hook + ``📍 Location:`` line.
- ``source_duration`` pretty (e.g. ``"30m 0s"``): max 1× — disclosure line.
- ``video_duration`` pretty (e.g. ``"9m 48s"``): max 1× — disclosure line.
- Body line (between hook and ``📍 Location:``): if non-empty, must add
  at least one noun NOT in the structured block (chapter town, named
  route, terrain). Pure restatement fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD
from gomoveshift.pipeline.validation.content_gates import (
    check_description_redundancy,
)


LOCATION = "Dingle Peninsula toward Gortmore, County Kerry"
RECORDING_DATE = "2025-09-14"
SOURCE_DURATION = 1800.0  # 30m 0s
VIDEO_DURATION = 588.0  # 9m 48s


def _clean_description() -> str:
    """A cleaned description: hook + structured block, no body."""
    return (
        f"Highlights from 30m 0s of driving through {LOCATION}.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"🔔 Subscribe for more POV driving videos from around the world!\n"
        f"\n"
        f"#PovDriving #ScenicDrive\n"
    )


def _meta(description: str, **overrides) -> dict:
    base = {
        "title": f"Highlights from {LOCATION}",
        "description": description,
        "location": LOCATION,
        "recording_date": RECORDING_DATE,
        "source_duration": SOURCE_DURATION,
        "video_duration": VIDEO_DURATION,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────
# Pure-function gate tests
# ─────────────────────────────────────────────────────────────────────────


def test_passes_clean_description():
    """Cleaned description (location 2×, date 1×, durations 1×
    each, no body) passes."""
    result = check_description_redundancy(_meta(_clean_description()))
    assert result.passed, f"clean description failed: {result.message}"
    assert result.gate == "description_redundancy"


def test_fails_when_recording_date_in_hook_and_filmed_line():
    """Date appears 2× — once in the hook, once in the 📅 Filmed line."""
    desc = (
        f"POV drive through {LOCATION} on {RECORDING_DATE}.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    result = check_description_redundancy(_meta(desc))
    assert not result.passed
    assert RECORDING_DATE in result.message
    assert "2" in result.message  # references the over-count


def test_fails_when_location_in_hook_body_and_location_line():
    """Location appears 3× — hook + body restatement + 📍 Location line."""
    desc = (
        f"POV drive through {LOCATION}.\n"
        f"\n"
        f"A 9m 48s drive through {LOCATION}.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    result = check_description_redundancy(_meta(desc))
    assert not result.passed
    assert "location" in result.message.lower()


def test_fails_when_source_duration_in_hook_and_disclosure():
    """Source duration appears 3× — hook (allowed in Highlights band),
    disclosure (allowed), and a body restatement (over budget).

    Allowed cap is 2× because the Highlights-band hook legitimately
    surfaces source_duration alongside the disclosure line. Three
    occurrences forces a body restatement, which the gate must catch.
    """
    desc = (
        f"Highlights from 30m 0s of driving through {LOCATION}.\n"
        f"\n"
        f"This 30m 0s of footage was a great drive.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    result = check_description_redundancy(_meta(desc))
    assert not result.passed
    assert "30m 0s" in result.message


def test_passes_when_body_adds_chapter_town():
    """Body contains a chapter town not in the structured block."""
    desc = (
        f"Highlights from 30m 0s of driving through {LOCATION}.\n"
        f"\n"
        f"A scenic route passing through Dingle and Slea Head.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    meta = _meta(desc, chapter_towns=["Dingle", "Slea Head"])
    result = check_description_redundancy(meta)
    assert result.passed, f"body with chapter towns rejected: {result.message}"


def test_fails_when_body_is_pure_restatement():
    """Body equals ``A {duration} drive through {location}.`` — no new noun."""
    desc = (
        f"Highlights from 30m 0s of driving through {LOCATION}.\n"
        f"\n"
        f"A 9m 48s drive through {LOCATION}.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    result = check_description_redundancy(_meta(desc))
    assert not result.passed
    assert (
        "body" in result.message.lower()
        or "restatement" in result.message.lower()
        or "location" in result.message.lower()
    )


# ─────────────────────────────────────────────────────────────────────────
# Embedded 📍 CHAPTERS block is NOT counted as prose redundancy
# ─────────────────────────────────────────────────────────────────────────


def _chapter_block(*places: str) -> str:
    """An embedded 📍 CHAPTERS block (the real shape from longform_metadata
    ._embed_chapters: ``\\n\\n📍 CHAPTERS\\n<timestamped lines>``). Chapter
    names are place names BY DESIGN (chapters double as search terms)."""
    lines = "\n".join(f"0:0{i} {p}" for i, p in enumerate(places))
    return f"\n\n📍 CHAPTERS\n{lines}"


def test_passes_when_chapters_repeat_location_beyond_prose_budget():
    """Prose mentions the location exactly 2× (within budget) but the
    embedded 📍 CHAPTERS block repeats it several more times. The chapter
    block is a separate functional block — the gate must PASS."""
    desc = (
        f"Highlights from 30m 0s of driving through {LOCATION}.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
        f"{_chapter_block(LOCATION, LOCATION, LOCATION)}"
    )
    # Sanity: the FULL description mentions the location 5× (2 prose + 3
    # chapter), which would breach the 2× budget if counted naively.
    assert desc.count(LOCATION) == 5
    result = check_description_redundancy(_meta(desc))
    assert result.passed, (
        f"chapter block place names must not count as prose redundancy: {result.message}"
    )


def test_fails_when_prose_exceeds_budget_despite_chapter_exclusion():
    """Counter-test: excluding the chapter block does NOT weaken the
    gate — prose that itself breaches the 2× location budget still FAILS,
    even when a chapter block is present."""
    desc = (
        f"POV drive through {LOCATION}.\n"
        f"\n"
        f"A 9m 48s drive through {LOCATION}.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
        f"{_chapter_block('Dingle', 'Sneem')}"
    )
    result = check_description_redundancy(_meta(desc))
    assert not result.passed
    assert "location" in result.message.lower()


def test_body_line_check_still_runs_on_full_description_with_chapters():
    """The body-line check (#5) still operates on the full description.
    A pure-restatement body still FAILS even when a chapter block follows."""
    desc = (
        f"Highlights from 30m 0s of driving through {LOCATION}.\n"
        f"\n"
        f"A 9m 48s drive through the area.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
        f"{_chapter_block('Dingle', 'Sneem')}"
    )
    # No chapter_towns / named_routes / terrain threaded → body is pure
    # restatement by construction.
    result = check_description_redundancy(_meta(desc))
    assert not result.passed
    assert "body" in result.message.lower() or "restatement" in result.message.lower()


# ─────────────────────────────────────────────────────────────────────────
# SEO search line is a sanctioned surface — not prose redundancy
# ─────────────────────────────────────────────────────────────────────────

VIA_LOCATION = "Kilgarvan to Ballylickey via the Famine Tunnel"


def test_passes_when_seo_search_line_repeats_region_less_via_label():
    """A GPS-less ``A to B via Road`` label (no region component) makes the
    SEO search line reconstruct the FULL location verbatim
    (``"{arc} via {road} — 4K drive."``). With the hook + 📍 Location
    that is 3× exact — but the SEO search line is a sanctioned above-the-fold
    surface (like the 📍 CHAPTERS block), not prose redundancy. Gate PASSES.
    """
    desc = (
        f"Highlights from 30m 0s of driving through {VIA_LOCATION}.\n"
        f"{VIA_LOCATION} — 4K drive.\n"
        f"\n"
        f"📍 Location: {VIA_LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    # Sanity: the full description names the location 3× (hook + SEO line + 📍).
    assert desc.count(VIA_LOCATION) == 3
    result = check_description_redundancy(_meta(desc, location=VIA_LOCATION))
    assert result.passed, f"SEO search line must not count as prose redundancy: {result.message}"


def test_seo_line_exclusion_does_not_weaken_location_budget():
    """Counter-test: excluding the SEO search line does NOT weaken the gate —
    a body restatement (a 4th occurrence with no ``4K drive`` marker)
    still breaches the 2× location budget and FAILS."""
    desc = (
        f"Highlights from 30m 0s of driving through {VIA_LOCATION}.\n"
        f"{VIA_LOCATION} — 4K drive.\n"
        f"\n"
        f"A 9m 48s drive through {VIA_LOCATION}.\n"
        f"\n"
        f"📍 Location: {VIA_LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    result = check_description_redundancy(_meta(desc, location=VIA_LOCATION))
    assert not result.passed
    assert "location" in result.message.lower()


# ─────────────────────────────────────────────────────────────────────────
# Phase D wiring — integration test
# ─────────────────────────────────────────────────────────────────────────


def _install_fake_youtube_uploader(monkeypatch):
    fake_mod = ModuleType("gomoveshift.pipeline.drive_phases.youtube_uploader")
    uploader_instance = MagicMock()
    uploader_instance.upload_video.return_value = "vid_01"
    uploader_instance.set_thumbnail.return_value = True

    def _upload_if_needed(*, state, lf_idx, video_path, **kwargs):
        return None

    uploader_instance.upload_if_needed.side_effect = _upload_if_needed

    class YouTubeUploader:
        def __new__(cls, *a, **kw):
            return uploader_instance

    fake_mod.YouTubeUploader = YouTubeUploader
    monkeypatch.setitem(
        sys.modules,
        "gomoveshift.pipeline.drive_phases.youtube_uploader",
        fake_mod,
    )
    return uploader_instance


def _build_state(tmp_path, num_lfs=1):
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source_drive"
    source.mkdir()

    regular_lfs = []
    lf_paths = []
    shorts_per_lf: dict[int, list] = {}
    for i in range(num_lfs):
        seg = LFSegment(
            source_file_index=i,
            source_path=tmp_path / f"ep_{i:03d}_FINAL.mp4",
            start_time=0.0,
            end_time=600.0,
            original_start_time=0.0,
            original_end_time=600.0,
            composite_score=0.7,
            gps_center_lat=0.0,
            gps_center_lon=0.0,
            terrain_type="countryside",
            contiguous_with_previous=False,
        )
        regular_lfs.append(
            RegularLFPlan(
                index=i + 1,
                segments=[seg],
                montage_peaks=[seg],
                shorts_target=1,
                region="ireland",
                total_duration=600.0,
            )
        )
        lf_path = output / f"lf_{i + 1:02d}.mp4"
        lf_path.write_bytes(b"x" * (2 * 1024 * 1024))
        lf_paths.append(lf_path)
        shorts_per_lf[i + 1] = [MagicMock()]

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=600.0 * num_lfs,
        total_highlight_duration=600.0 * num_lfs,
    )

    from _state_helpers import make_state

    return make_state(
        source_dir=source,
        output_base=output,
        region="ireland",
        publish_day="wed",
        location=LOCATION,
        skip_upload=True,
        skip_csv=True,
        youtube_upload=False,
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        shorts_per_lf=shorts_per_lf,
        per_lf_recording_date={i + 1: RECORDING_DATE for i in range(num_lfs)},
        # BC→D boundary contract — every rendered LF must appear here.
        per_lf_locations={i + 1: LOCATION for i in range(num_lfs)},
        per_lf_trim_ratio={i + 1: 0.7 for i in range(num_lfs)},
    )


def _phase_d_patches():
    return [
        patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.ReleaseSchedule"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.UploadManifest"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineValidator"),
    ]


def _setup_mocks_redundant(mocks, num_lfs):
    """Stage a meta_gen that returns a REDUNDANT description so the gate fires."""
    (meta_cls, sched_cls, rs_cls, social_cls, upload_cls, manifest_cls, val_cls) = mocks

    redundant_desc = (
        f"POV drive through {LOCATION} on {RECORDING_DATE}.\n"
        f"\n"
        f"A 9m 48s drive through {LOCATION}.\n"
        f"\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )

    meta = MagicMock()

    def _meta_side_effect(**kwargs):
        return {
            "title": f"Highlights from {LOCATION}",
            "description": redundant_desc,
            "location": LOCATION,
            "recording_date": kwargs.get("recording_date"),
            "video_duration": kwargs.get("video_duration"),
            "source_duration": kwargs.get("source_duration"),
            "tags": ["driving"],
            "category": "Travel",
        }

    meta.generate.side_effect = _meta_side_effect
    meta_cls.return_value = meta

    from datetime import datetime as _dt
    from datetime import timedelta as _td

    wed_dates = [(_dt(2026, 4, 1) + _td(days=7 * i)).strftime("%Y-%m-%d") for i in range(num_lfs)]
    lf_slots = [
        MagicMock(
            date=wed_dates[i],
            time="11:00:00",
            datetime_est=f"{wed_dates[i]}T11:00:00-04:00",
            day_of_week="Wednesday",
        )
        for i in range(num_lfs)
    ]
    drive_schedule = MagicMock()
    drive_schedule.lf_slots = lf_slots
    drive_schedule.shorts_slots = []
    drive_schedule.to_dict.return_value = {"region": "ireland"}
    sched = MagicMock()
    sched.generate_drive_schedule.return_value = drive_schedule
    sched_cls.return_value = sched
    sched_cls.from_config.return_value = sched

    social = MagicMock()
    social.generate_all.return_value = []
    social_cls.return_value = social

    uploader = MagicMock()
    upload_cls.return_value = uploader

    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok", gates=[])
    val_cls.return_value = val


def test_phase_d_aborts_lf_write_on_redundancy_failure(tmp_path, monkeypatch):
    """Wired integration test: with a redundant LF description staged, Phase
    D must raise (RuntimeError) and the per-LF youtube.json must NOT be
    written to disk."""
    _install_fake_youtube_uploader(monkeypatch)
    state = _build_state(tmp_path, num_lfs=1)

    output_base = state["output_base"]
    project_base = state["source_dir"].stem.replace(" ", "_").lower()
    expected_meta_path = output_base / f"{project_base}_lf01_youtube.json"

    patches = _phase_d_patches()
    with (
        patches[0] as m0,
        patches[1] as m1,
        patches[2] as m2,
        patches[3] as m3,
        patches[4] as m4,
        patches[5] as m5,
        patches[6] as m6,
    ):
        _setup_mocks_redundant((m0, m1, m2, m3, m4, m5, m6), num_lfs=1)
        with pytest.raises(RuntimeError, match=r"description redundancy"):
            PhaseD(get_config()).run(state)

    assert not expected_meta_path.exists(), (
        f"redundant LF metadata JSON should NOT be on disk: {expected_meta_path}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Regression: the LF's own label must not be restated on the "Through:" line
# ─────────────────────────────────────────────────────────────────────────
#
# A long-form video failed Phase D with
#   location 'Sullivan County, Tennessee' appears 3x (allowed: 2x)
# The three surfaces were the trim-banded hook, the "Location:" line, and
# the "Through:" regions line — which listed the LF's OWN region. The fix
# is at the generator (longform_metadata._format_regions_line filters an
# exact match), NOT a third exemption in this counter.


_SELF = "Sullivan County, Tennessee"


def test_passes_when_regions_line_omits_the_lf_own_label():
    """A single-county LF: hook + Location line = 2x, within budget."""
    desc = (
        f"Highlights from 3h 17m of driving through {_SELF}.\n"
        f"{_SELF} — 4K drive.\n"
        f"\n"
        f"📍 Location: {_SELF}\n"
        f"🎬 Final cut: 1h 51m (edited from 3h 17m of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    assert check_description_redundancy(_meta(desc, location=_SELF)).passed


def test_fails_when_regions_line_restates_the_lf_own_label():
    """The production shape, reproduced: a 'Through:' line naming the LF's
    own region is the 3rd occurrence and must still breach the budget."""
    desc = (
        f"Highlights from 3h 17m of driving through {_SELF}.\n"
        f"{_SELF} — 4K drive.\n"
        f"\n"
        f"📍 Location: {_SELF}\n"
        f"🗺️ Through: {_SELF}\n"
        f"🎬 Final cut: 1h 51m (edited from 3h 17m of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )
    result = check_description_redundancy(_meta(desc, location=_SELF))
    assert not result.passed
    assert "location" in result.message.lower()


# ─────────────────────────────────────────────────────────────────────────
# Distance/pace stat lines must be scoped — a full-drive (pre-cut)
# total needs an explicit "Full drive:" prefix; a kept-segment-derived stat
# (the default) needs none.
# ─────────────────────────────────────────────────────────────────────────


def _desc_with_distance_line(distance_line: str) -> str:
    return (
        f"Highlights from 30m 0s of driving through {LOCATION}.\n"
        f"\n"
        f"{distance_line}\n"
        f"📍 Location: {LOCATION}\n"
        f"🎬 Final cut: 9m 48s (edited from 30m 0s of footage)\n"
        f"📅 Filmed: {RECORDING_DATE}\n"
    )


def test_fails_when_full_drive_distance_lacks_prefix():
    """distance_km is flagged as a full-drive (pre-cut) total but the
    rendered line carries no 'Full drive:' scope prefix — hard failure
    naming the offending line."""
    desc = _desc_with_distance_line("\U0001f6e3️ Distance: 22 km (14 mi)")
    result = check_description_redundancy(_meta(desc, full_drive_stats=["distance"]))
    assert not result.passed
    assert "Distance: 22 km (14 mi)" in result.message


def test_passes_when_full_drive_distance_carries_prefix():
    """Same full-drive-flagged stat, but the line explicitly scopes itself
    with 'Full drive:' — passes."""
    desc = _desc_with_distance_line("\U0001f6e3️ Full drive: 22 km (14 mi)")
    result = check_description_redundancy(_meta(desc, full_drive_stats=["distance"]))
    assert result.passed, f"Full drive:-prefixed stat rejected: {result.message}"


def test_passes_when_distance_is_kept_segment_derived():
    """No full_drive_stats declared (the default) — a plain, unprefixed
    Distance line is kept-segment-derived and passes."""
    desc = _desc_with_distance_line("\U0001f6e3️ Distance: 6 km (4 mi)")
    result = check_description_redundancy(_meta(desc))
    assert result.passed, f"kept-segment stat rejected: {result.message}"


def test_passes_with_no_stat_lines():
    """No Distance/Pace line at all — the scope rule is a no-op."""
    result = check_description_redundancy(
        _meta(_clean_description(), full_drive_stats=["distance"])
    )
    assert result.passed, f"description with no stat lines should pass: {result.message}"
