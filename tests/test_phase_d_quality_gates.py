"""Tests for Phase D end-of-phase quality gates.

``quality_gates.assert_phase_d_quality`` is the Layer-2 gate that runs
after all per-LF loops in ``PhaseD.run``. It accumulates every semantic
violation it finds (title collisions, seo_degraded, publish_day drift,
drive-wide location leak into a per-LF, CSV row-count mismatch, phantom
video IDs in First Comment Text, schedule miscounts, wrong-day slots)
and raises ``PhaseQualityError`` once with the full list.

These tests pin the Layer-2 rules and two specific regressions —
all-identical LF titles and phantom video IDs in Metricool CSV — each
named after the failure so future regressions are easy to find.

See also:
  * ``gomoveshift/pipeline/quality_gates.py`` — the implementation
  * ``gomoveshift/pipeline/exceptions.py`` — ``PhaseQualityError``
  * ``tests/test_drive_cli.py`` — ``--allow-seo-degraded`` flag plumbing
"""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.csv_generator import GOLDEN_COLUMNS


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_drive_plan(num_lfs: int = 3):
    """Minimal drive_plan with N regular_lfs (indexed 1..N)."""
    regular_lfs = [SimpleNamespace(index=i, total_duration=1800.0) for i in range(1, num_lfs + 1)]
    return SimpleNamespace(regular_lfs=regular_lfs)


def _make_lf_slot(date: str, time: str = "11:00:00"):
    """Mimic ScheduledSlot (duck-typed: ``.date`` + ``.time``)."""
    return SimpleNamespace(date=date, time=time, datetime_est=f"{date}T{time}-04:00")


def _make_state(
    num_lfs: int = 3,
    titles: list[str] | None = None,
    per_lf_locations: dict[int, str] | None = None,
    seo_degraded_indices: set[int] | None = None,
    allow_seo_degraded: bool = False,
    publish_day: str = "wed",
    drive_location: str = "South Carolina to North Carolina",
    lf_slot_dates: list[str] | None = None,
    shorts_per_lf_counts: list[int] | None = None,
) -> dict:
    """Assemble a minimal Phase D output state for gate testing.

    All overrides default to the "happy path" so any single override
    produces a state that violates exactly one rule.
    """
    if titles is None:
        titles = [f"LF {i} Title via Route {i}, SC" for i in range(1, num_lfs + 1)]
    if per_lf_locations is None:
        per_lf_locations = {
            i: f"LF {i} Place to LF {i} Other Place via Road {i}, SC" for i in range(1, num_lfs + 1)
        }
    if seo_degraded_indices is None:
        seo_degraded_indices = set()
    if lf_slot_dates is None:
        # Four Wednesdays in April–May 2026
        lf_slot_dates = ["2026-04-29", "2026-05-06", "2026-05-13", "2026-05-20"][:num_lfs]
    if shorts_per_lf_counts is None:
        shorts_per_lf_counts = [3] * num_lfs

    metadata_per_lf: dict[int, dict] = {}
    for i in range(1, num_lfs + 1):
        metadata_per_lf[i] = {
            "title": titles[i - 1],
            "description": f"desc {i}",
            "tags": ["driving"],
            "seo_degraded": i in seo_degraded_indices,
        }

    shorts_per_lf: dict[int, list] = {}
    for i in range(1, num_lfs + 1):
        shorts_per_lf[i] = [object() for _ in range(shorts_per_lf_counts[i - 1])]

    lf_slots = [_make_lf_slot(d) for d in lf_slot_dates]
    drive_schedule = SimpleNamespace(lf_slots=lf_slots, shorts_slots=[])

    return {
        "publish_day": publish_day,
        "location": drive_location,
        "metadata_per_lf": metadata_per_lf,
        "per_lf_locations": per_lf_locations,
        "shorts_per_lf": shorts_per_lf,
        "drive_schedule": drive_schedule,
        "allow_seo_degraded": allow_seo_degraded,
        "youtube_uploads": {},
        # Default to skip_upload=True so legacy rule 1-8 tests don't have to
        # populate upload_manifests just to satisfy rule 9. Tests
        # that exercise rule 9 set skip_upload=False explicitly.
        "skip_upload": True,
        "upload_manifests": {},
    }


def _write_shorts_csv(
    path: Path,
    rows: list[dict],
) -> Path:
    """Write a shorts CSV with GOLDEN_COLUMNS header. Rows are filled-in per input dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    full_rows = []
    for r in rows:
        row = {col: "" for col in GOLDEN_COLUMNS}
        row.update(r)
        full_rows.append(row)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=GOLDEN_COLUMNS)
        writer.writeheader()
        writer.writerows(full_rows)
    return path


# ---------------------------------------------------------------------------
# Rule 1 — LF titles all distinct
# ---------------------------------------------------------------------------


class TestRule1TitlesDistinct:
    # Rules 1+2 moved UPSTREAM of the upload: they are enforced by
    # ``assert_phase_d_title_quality`` before publish, not by the end-of-Phase-D
    # ``assert_phase_d_quality``. These tests pin the upstream gate directly.
    def test_distinct_titles_pass(self):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

        state = _make_state(num_lfs=3)
        assert_phase_d_title_quality(
            state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
        )

    def test_duplicate_titles_raise(self):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

        state = _make_state(
            num_lfs=3,
            titles=["Same Title", "Same Title", "Same Title"],
        )

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_title_quality(
                state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
            )

        msg = str(exc_info.value)
        assert "titles" in msg.lower()
        assert "Same Title" in msg


# ---------------------------------------------------------------------------
# Rule 2 — No seo_degraded=True unless --allow-seo-degraded
# ---------------------------------------------------------------------------


class TestRule2SeoDegraded:
    # See TestRule1TitlesDistinct: Rules 1+2 run in the upstream title gate.
    def test_no_seo_degraded_passes(self):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

        state = _make_state(num_lfs=2)
        assert_phase_d_title_quality(
            state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
        )

    def test_seo_degraded_false_flag_raises(self):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

        state = _make_state(
            num_lfs=2,
            seo_degraded_indices={1},
            allow_seo_degraded=False,
        )

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_title_quality(
                state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
            )

        msg = str(exc_info.value)
        assert "seo_degraded" in msg
        assert "1" in msg  # the LF index

    def test_seo_degraded_with_flag_passes(self):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

        state = _make_state(
            num_lfs=2,
            seo_degraded_indices={1, 2},
            allow_seo_degraded=True,
        )
        assert_phase_d_title_quality(
            state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
        )


# ---------------------------------------------------------------------------
# Rule 3 — publish_day in LONG_FORM_DAYS
# ---------------------------------------------------------------------------


class TestRule3PublishDayInLongFormDays:
    def test_wed_passes(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=2, publish_day="wed")
        state["shorts_per_lf"] = {1: [], 2: []}
        drive_plan = _make_drive_plan(num_lfs=2)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_sat_raises_when_long_form_days_is_wed_only(self, tmp_path):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=2, publish_day="sat")
        state["shorts_per_lf"] = {1: [], 2: []}
        drive_plan = _make_drive_plan(num_lfs=2)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "publish_day" in msg
        assert "sat" in msg


# ---------------------------------------------------------------------------
# Rule 4 — DELETED.
#
# ``check_per_lf_location_not_drive_wide`` was removed because the
# drive-wide fallback path in ``derive_per_lf_locations`` was deleted at
# the source. By the robust-by-construction principle, the failure class
# this gate caught is now unreachable from a legitimate CLI invocation.
# The contract assertion at OST-burn entry (``day_pipeline.py``) is the
# only surviving guard, and it fires only on construction-logic
# regressions — not on operator input.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rule 5 — CSV row count == Σ len(shorts_per_lf[i])
# ---------------------------------------------------------------------------


class TestRule5CsvRowCount:
    def test_matching_count_passes(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=2, shorts_per_lf_counts=[3, 2])
        drive_plan = _make_drive_plan(num_lfs=2)
        rows = [{"Text": f"caption {i}", "First Comment Text": ""} for i in range(5)]
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", rows)

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_mismatched_count_raises(self, tmp_path):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        # Expect 5 shorts (3 + 2) but CSV has only 2 rows.
        state = _make_state(num_lfs=2, shorts_per_lf_counts=[3, 2])
        drive_plan = _make_drive_plan(num_lfs=2)
        rows = [
            {"Text": "c1", "First Comment Text": ""},
            {"Text": "c2", "First Comment Text": ""},
        ]
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", rows)

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "CSV row count" in msg or "csv row count" in msg.lower()
        assert "5" in msg
        assert "2" in msg


# ---------------------------------------------------------------------------
# Rule 6 — First Comment Text matches URL regex OR placeholder (never phantom ID)
# ---------------------------------------------------------------------------


class TestRule6FirstCommentText:
    def test_empty_first_comment_passes(self, tmp_path):
        """Placeholder-pattern: empty string is OK (pre-upload / skip-upload)."""
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=1, shorts_per_lf_counts=[2])
        drive_plan = _make_drive_plan(num_lfs=1)
        rows = [
            {"Text": "c1", "First Comment Text": ""},
            {"Text": "c2", "First Comment Text": ""},
        ]
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", rows)

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_youtube_url_first_comment_passes(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=1, shorts_per_lf_counts=[2])
        drive_plan = _make_drive_plan(num_lfs=1)
        rows = [
            {
                "Text": "c1",
                "First Comment Text": "Watch the full drive here: https://youtu.be/abc12345678",
            },
            {
                "Text": "c2",
                "First Comment Text": "Watch the full drive here: https://youtube.com/watch?v=xyzDEF12345",
            },
        ]
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", rows)

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_hashtag_placeholder_passes(self, tmp_path):
        """Non-URL comment (e.g. ``#shorts``) is a valid placeholder."""
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=1, shorts_per_lf_counts=[1])
        drive_plan = _make_drive_plan(num_lfs=1)
        rows = [{"Text": "c1", "First Comment Text": "#shorts #driving"}]
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", rows)

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_phantom_bare_video_id_raises(self, tmp_path):
        """Bare 11-char video ID (no URL wrapping) is a phantom → raise."""
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=1, shorts_per_lf_counts=[2])
        drive_plan = _make_drive_plan(num_lfs=1)
        rows = [
            {
                "Text": "c1",
                "First Comment Text": "abc12345678",  # pragma: allowlist secret
            },  # phantom
            {"Text": "c2", "First Comment Text": ""},
        ]
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", rows)

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "phantom" in msg.lower() or "First Comment" in msg
        assert "abc12345678" in msg  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# Rules 5/6 — CSV expected-but-absent must error
#
# A non-None csv_path that does NOT exist on disk used to pass vacuously,
# conflating "CSV never written (a bug)" with "skip_csv (intentional)".
# ``expect_csv=True`` (threaded from ``not skip_csv``) flips the missing
# case to a hard error; ``csv_path is None`` is still a clean intentional
# skip regardless of the flag.
# ---------------------------------------------------------------------------


class TestCsvExpectedButAbsent:
    def test_row_count_missing_when_expected_errors(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import check_csv_row_count

        state = _make_state(num_lfs=1, shorts_per_lf_counts=[2])
        missing = tmp_path / "never_written.csv"
        assert not missing.exists()

        errors = check_csv_row_count(state, missing, expect_csv=True)

        assert len(errors) == 1
        assert str(missing) in errors[0]

    def test_row_count_none_path_passes(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import check_csv_row_count

        state = _make_state(num_lfs=1, shorts_per_lf_counts=[2])
        # Intentional skip — csv_path None passes even with expect_csv=True.
        assert check_csv_row_count(state, None, expect_csv=True) == []

    def test_row_count_missing_not_expected_passes(self, tmp_path):
        """Default expect_csv=False preserves the current vacuous-pass for
        existing callers."""
        from gomoveshift.pipeline.quality_gates import check_csv_row_count

        state = _make_state(num_lfs=1, shorts_per_lf_counts=[2])
        missing = tmp_path / "never_written.csv"
        assert check_csv_row_count(state, missing) == []

    def test_first_comment_missing_when_expected_errors(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import check_csv_first_comment_text

        missing = tmp_path / "never_written.csv"
        assert not missing.exists()

        errors = check_csv_first_comment_text(missing, expect_csv=True)

        assert len(errors) == 1
        assert str(missing) in errors[0]

    def test_first_comment_none_path_passes(self):
        from gomoveshift.pipeline.quality_gates import check_csv_first_comment_text

        assert check_csv_first_comment_text(None, expect_csv=True) == []


# ---------------------------------------------------------------------------
# Rule 7 — drive_schedule.lf_slots count == len(drive_plan.regular_lfs)
# ---------------------------------------------------------------------------


class TestRule7ScheduleCount:
    def test_matching_count_passes(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=3)
        state["shorts_per_lf"] = {1: [], 2: [], 3: []}
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_mismatched_count_raises(self, tmp_path):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(
            num_lfs=3,
            lf_slot_dates=["2026-04-29", "2026-05-06"],  # only 2 slots, plan has 3
        )
        state["shorts_per_lf"] = {1: [], 2: [], 3: []}
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "lf_slots" in msg or "lf slots" in msg.lower()
        assert "3" in msg  # plan count
        assert "2" in msg  # slot count


# ---------------------------------------------------------------------------
# Rule 8 — Every lf_slot.date weekday in LONG_FORM_DAYS
# ---------------------------------------------------------------------------


class TestRule8SlotDayInLongFormDays:
    def test_wed_dates_pass(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        # 2026-04-29 is a Wednesday; 2026-05-06 is too.
        state = _make_state(num_lfs=2, lf_slot_dates=["2026-04-29", "2026-05-06"])
        state["shorts_per_lf"] = {1: [], 2: []}
        drive_plan = _make_drive_plan(num_lfs=2)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_saturday_date_raises(self, tmp_path):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        # 2026-05-02 is Saturday; LONG_FORM_DAYS={2} (Wed only) rejects it.
        state = _make_state(num_lfs=2, lf_slot_dates=["2026-04-29", "2026-05-02"])
        state["shorts_per_lf"] = {1: [], 2: []}
        drive_plan = _make_drive_plan(num_lfs=2)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "2026-05-02" in msg
        assert "LONG_FORM_DAYS" in msg or "long_form_days" in msg.lower()


# ---------------------------------------------------------------------------
# Rule 9 — shorts upload completeness
# ---------------------------------------------------------------------------
#
# A run once reported all phases passing while every Drive
# upload silently failed (FileNotFoundError caught + WARN'd in phase_d).
# Every shorts CSV row had empty Picture Url 1 → Metricool import was empty.
# This rule fails the gate when ``upload_manifests`` is missing entries for
# LFs in ``drive_plan.regular_lfs`` AND the run did NOT skip uploads.


class TestRule9ShortsUploadComplete:
    def test_all_manifests_present_passes(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=3)
        state["shorts_per_lf"] = {1: [], 2: [], 3: []}
        # Simulate phase_d having populated upload_manifests for every LF.
        state["upload_manifests"] = {
            1: SimpleNamespace(files={"short_01.mp4": "https://drive.google.com/x"}),
            2: SimpleNamespace(files={"short_02.mp4": "https://drive.google.com/y"}),
            3: SimpleNamespace(files={"short_03.mp4": "https://drive.google.com/z"}),
        }
        state["skip_upload"] = False
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_missing_manifests_raise_naming_lfs(self, tmp_path):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        # All seven Wednesdays starting 2026-04-29 (weekly cadence).
        wed_dates = [
            "2026-04-29",
            "2026-05-06",
            "2026-05-13",
            "2026-05-20",
            "2026-05-27",
            "2026-06-03",
            "2026-06-10",
        ]
        state = _make_state(num_lfs=7, lf_slot_dates=wed_dates)
        # The failure shape: every LF HAS shorts but the upload
        # failed, leaving upload_manifests empty. (A zero-shorts LF needs no
        # manifest — Rule 9 only expects one for LFs that produced shorts.)
        state["shorts_per_lf"] = {i: [object()] for i in range(1, 8)}
        state["upload_manifests"] = {}
        state["skip_upload"] = False
        drive_plan = _make_drive_plan(num_lfs=7)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [{} for _ in range(7)])

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "shorts" in msg.lower() and "upload" in msg.lower()
        # All seven LF indices must be named in the error.
        for i in range(1, 8):
            assert str(i) in msg

    def test_partial_failure_raises_naming_only_missing(self, tmp_path):
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=3)
        # All three LFs have shorts; LF 2's manifest is missing — the gate must
        # call out "2" and not the others.
        state["shorts_per_lf"] = {1: [object()], 2: [object()], 3: [object()]}
        state["upload_manifests"] = {
            1: SimpleNamespace(files={"short_01.mp4": "https://drive.google.com/x"}),
            3: SimpleNamespace(files={"short_03.mp4": "https://drive.google.com/z"}),
        }
        state["skip_upload"] = False
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [{} for _ in range(3)])

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "shorts" in msg.lower() and "upload" in msg.lower()
        assert "2" in msg

    def test_skip_upload_bypasses_check(self, tmp_path):
        """When the operator passed --skip-upload, missing manifests is expected."""
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _make_state(num_lfs=3)
        state["shorts_per_lf"] = {1: [], 2: [], 3: []}
        state["upload_manifests"] = {}
        state["skip_upload"] = True
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)


# ---------------------------------------------------------------------------
# Rule 10 — YouTube LF upload completeness (silent-drop catch)
# ---------------------------------------------------------------------------
#
# ``external_systems_coordinator._upload_lf_to_youtube`` catches a transient
# ``_HttpError`` / bare ``Exception`` and returns ``None``; the caller loop
# does ``if record is None: continue`` — so a failed per-LF YouTube upload is
# silently skipped, leaving ``state["youtube_uploads"]`` short of a
# ``video_id`` for that LF. Without this rule the drive reports success on a
# state where an LF's YouTube video never uploaded. Rule 10 hard-fails the
# gate naming the missing LF indices. No-op when YouTube upload is disabled
# (``youtube_upload`` falsy) or skipped (``skip_youtube_upload`` /
# ``skip_upload``), mirroring Rule 9's skip semantics.


def _yt_enabled_state(num_lfs: int = 3, **kwargs) -> dict:
    """Happy-path state with YouTube upload ON and a full youtube_uploads map."""
    state = _make_state(num_lfs=num_lfs, **kwargs)
    state["youtube_upload"] = True
    state["skip_upload"] = False
    state["skip_youtube_upload"] = False
    state["youtube_uploads"] = {i: {"video_id": f"vid{i:08d}"} for i in range(1, num_lfs + 1)}
    # Satisfy Rule 9 (Drive shorts manifests) so only Rule 10 can fire.
    state["shorts_per_lf"] = {i: [] for i in range(1, num_lfs + 1)}
    return state


class TestRule10YouTubeUploadComplete:
    def test_all_video_ids_present_passes(self, tmp_path):
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _yt_enabled_state(num_lfs=3)
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_missing_upload_raises_naming_lf(self, tmp_path):
        """LF 2's upload failed transiently (returned None) → no entry."""
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _yt_enabled_state(num_lfs=3)
        del state["youtube_uploads"][2]  # LF 2 silently dropped
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "youtube" in msg.lower() and "upload" in msg.lower()
        assert "2" in msg

    def test_entry_without_video_id_raises(self, tmp_path):
        """A record with a falsy/missing video_id is as bad as no entry."""
        from gomoveshift.pipeline.exceptions import PhaseQualityError
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _yt_enabled_state(num_lfs=2)
        state["youtube_uploads"][2] = {"video_id": ""}  # empty id
        drive_plan = _make_drive_plan(num_lfs=2)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        with pytest.raises(PhaseQualityError) as exc_info:
            assert_phase_d_quality(state, drive_plan, csv_path)

        msg = str(exc_info.value)
        assert "youtube" in msg.lower()
        assert "2" in msg

    def test_str_keyed_uploads_pass(self, tmp_path):
        """state JSON round-trip stringifies keys — int/str tolerance."""
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _yt_enabled_state(num_lfs=2)
        state["youtube_uploads"] = {
            "1": {"video_id": "vidaaa"},
            "2": {"video_id": "vidbbb"},
        }
        drive_plan = _make_drive_plan(num_lfs=2)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_youtube_disabled_bypasses_check(self, tmp_path):
        """youtube_upload falsy (no --youtube-upload/--publish) → no-op."""
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _yt_enabled_state(num_lfs=3)
        state["youtube_upload"] = False  # upload never enabled this run
        state["youtube_uploads"] = {}
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_skip_youtube_upload_bypasses_check(self, tmp_path):
        """--skip-youtube-upload with uploads enabled → no-op."""
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _yt_enabled_state(num_lfs=3)
        state["skip_youtube_upload"] = True
        state["youtube_uploads"] = {}
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)

    def test_skip_upload_alias_bypasses_check(self, tmp_path):
        """--skip-upload back-compat alias also disables Rule 10."""
        from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

        state = _yt_enabled_state(num_lfs=3)
        state["skip_upload"] = True
        state["youtube_uploads"] = {}
        drive_plan = _make_drive_plan(num_lfs=3)
        csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

        assert_phase_d_quality(state, drive_plan, csv_path)


# ---------------------------------------------------------------------------
# Accumulate-and-report pattern
# ---------------------------------------------------------------------------


def test_gate_accumulates_errors_before_raising(tmp_path):
    """A single invocation that violates multiple rules must report all at once.

    Specifically: wrong publish_day (Rule 3) AND a schedule slot-count
    mismatch (Rule 7) trigger the same raise, and the message references
    both. (Title distinctness moved to the upstream title gate,
    so this accumulation pin now pairs two rules that remain in this gate.)
    """
    from gomoveshift.pipeline.exceptions import PhaseQualityError
    from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

    state = _make_state(
        num_lfs=3,
        publish_day="sat",  # Rule 3 fail (Sat not in LONG_FORM_DAYS={Wed})
    )
    state["shorts_per_lf"] = {1: [], 2: [], 3: []}
    # Rule 7 fail: 2 lf_slots for a 3-LF plan (both Wednesdays so Rule 8 stays clean).
    state["drive_schedule"] = SimpleNamespace(
        lf_slots=[_make_lf_slot("2026-04-29"), _make_lf_slot("2026-05-06")],
        shorts_slots=[],
    )
    drive_plan = _make_drive_plan(num_lfs=3)
    csv_path = _write_shorts_csv(tmp_path / "shorts.csv", [])

    with pytest.raises(PhaseQualityError) as exc_info:
        assert_phase_d_quality(state, drive_plan, csv_path)

    msg = str(exc_info.value)
    # Rule 3 mentioned
    assert "publish_day" in msg or "sat" in msg
    # Rule 7 mentioned
    assert "lf_slots count mismatch" in msg


# ---------------------------------------------------------------------------
# Regression pins
# ---------------------------------------------------------------------------


def test_title_drive_wide_leak_detected(tmp_path):
    """Regression: every LF's title was the same drive-wide location
    (a drive-wide --location leaked).

    Rule 4 was deleted because the drive-wide fallback
    path in ``derive_per_lf_locations`` is gone — the title-leak class
    is now caught upstream by the post-discovery validation gate at
    PhaseBC entry, not by a Phase D gate. Rule 1 still fires on the
    duplicate-title symptom — now enforced by the upstream title gate,
    which is the surviving Phase D signal.
    """
    from gomoveshift.pipeline.exceptions import PhaseQualityError
    from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

    drive_location = "South Carolina to North Carolina"
    titles = [drive_location] * 3
    per_lf = {1: drive_location, 2: drive_location, 3: drive_location}
    state = _make_state(
        num_lfs=3,
        titles=titles,
        per_lf_locations=per_lf,
        drive_location=drive_location,
    )

    with pytest.raises(PhaseQualityError) as exc_info:
        assert_phase_d_title_quality(
            state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
        )

    msg = str(exc_info.value)
    # Duplicate-title signal (Rule 1) — surviving Phase D guard.
    assert "titles" in msg.lower() and "South Carolina to North Carolina" in msg


def test_phantom_video_id_in_csv_detected(tmp_path):
    """Regression: the Metricool CSV had bare 11-char video
    IDs in First Comment Text, surviving from the stale sidecar. Rule 6
    must reject these.
    """
    from gomoveshift.pipeline.exceptions import PhaseQualityError
    from gomoveshift.pipeline.quality_gates import assert_phase_d_quality

    state = _make_state(num_lfs=1, shorts_per_lf_counts=[3])
    drive_plan = _make_drive_plan(num_lfs=1)
    # Bare 11-char phantom — exact pattern seen in a stale-sidecar CSV
    rows = [
        {"Text": "c1", "First Comment Text": "AAAAAAAAAAA"},
        {"Text": "c2", "First Comment Text": "BBBBBBBBBBB"},
        {"Text": "c3", "First Comment Text": ""},
    ]
    csv_path = _write_shorts_csv(tmp_path / "shorts.csv", rows)

    with pytest.raises(PhaseQualityError) as exc_info:
        assert_phase_d_quality(state, drive_plan, csv_path)

    msg = str(exc_info.value)
    assert "phantom" in msg.lower() or "First Comment" in msg
    # Both phantom IDs appear in the report
    assert "AAAAAAAAAAA" in msg
    assert "BBBBBBBBBBB" in msg


def test_all_lf_titles_identical_detected(tmp_path):
    """Regression: all 7 LF titles were identical — Rule 1 must catch it.

    Now enforced by the upstream title gate, which sees only
    the metadata map (schedule/CSV rules are irrelevant here).
    """
    from gomoveshift.pipeline.exceptions import PhaseQualityError
    from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

    state = _make_state(
        num_lfs=7,
        titles=["Scenic Drive, SC"] * 7,
    )

    with pytest.raises(PhaseQualityError) as exc_info:
        assert_phase_d_title_quality(
            state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
        )

    msg = str(exc_info.value)
    assert "titles" in msg.lower()
    assert "Scenic Drive, SC" in msg


def test_seo_degraded_not_allowed_by_default(tmp_path):
    """seo_degraded=True with default (no flag) must raise (upstream title gate)."""
    from gomoveshift.pipeline.exceptions import PhaseQualityError
    from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

    state = _make_state(
        num_lfs=3,
        seo_degraded_indices={1, 2, 3},
        allow_seo_degraded=False,
    )

    with pytest.raises(PhaseQualityError) as exc_info:
        assert_phase_d_title_quality(
            state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
        )

    msg = str(exc_info.value)
    assert "seo_degraded" in msg


def test_seo_degraded_allowed_with_flag(tmp_path):
    """seo_degraded=True with allow_seo_degraded=True must pass the gate."""
    from gomoveshift.pipeline.quality_gates import assert_phase_d_title_quality

    state = _make_state(
        num_lfs=3,
        seo_degraded_indices={1, 2, 3},
        allow_seo_degraded=True,
    )
    assert_phase_d_title_quality(
        state["metadata_per_lf"], allow_seo_degraded=state["allow_seo_degraded"]
    )


# ---------------------------------------------------------------------------
# CLI flag plumbing — --allow-seo-degraded
# ---------------------------------------------------------------------------


class TestAllowSeoDegradedFlagWiresThrough:
    """End-to-end flag plumbing: CLI parser → cmd_drive → orchestrator → state.

    The flag is per-invocation (CLI-wins); ``DrivePipelineOrchestrator.run``
    accepts it and deposits it on state, and Phase D's gate reads
    ``state["allow_seo_degraded"]``.
    """

    def test_parser_recognizes_flag(self, tmp_path):
        from gomoveshift.cli import _build_parser

        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--allow-seo-degraded",
            ]
        )
        assert getattr(args, "allow_seo_degraded") is True

    def test_parser_default_is_none(self, tmp_path):
        """Argparse-sentinel pattern — the
        parser defaults to ``None`` (not supplied) so the orchestrator
        can distinguish "operator did not pass the flag" from "operator
        explicitly said False". See parser.py --allow-seo-degraded for
        the sentinel rationale.
        """
        from gomoveshift.cli import _build_parser

        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
            ]
        )
        assert getattr(args, "allow_seo_degraded") is None

    def test_orchestrator_run_accepts_allow_seo_degraded(self):
        """Sentinel default is ``None`` — see
        parser.py for rationale."""
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        params = inspect.signature(DrivePipelineOrchestrator.run).parameters
        assert "allow_seo_degraded" in params, (
            "allow_seo_degraded missing from DrivePipelineOrchestrator.run; "
            "cmd_drive would silently drop the CLI flag."
        )
        assert params["allow_seo_degraded"].default is None

    def test_cmd_drive_passes_allow_seo_degraded_to_orchestrator(self, tmp_path, monkeypatch):
        from gomoveshift.cli import _build_parser
        from gomoveshift.cli.handlers.drive import cmd_drive
        from gomoveshift.config import get_config

        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--allow-seo-degraded",
                "--lf-day",
                "mon",
            ]
        )
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        captured_kwargs: dict = {}

        class FakeResult:
            success = True
            drive_plan = None
            errors: list = []

        class FakeOrch:
            def __init__(self, config):
                self.config = config

            def run(self, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeResult()

        with patch(
            "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator",
            FakeOrch,
        ):
            cmd_drive(args, get_config())

        assert captured_kwargs.get("allow_seo_degraded") is True
