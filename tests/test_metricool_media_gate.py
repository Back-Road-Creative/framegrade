"""Metricool shorts-CSV media gate.

``Picture Url 1`` is the Drive media URL Metricool fetches to attach each
short's video. A ``--skip-upload`` run can ship a CSV with every ``Picture Url 1`` blank,
so Metricool has nothing to post and silently drops every short. ``validate_csv`` only flagged a non-empty non-URL, so a fully
blank column passed every check.

These tests pin the hardening: ``build_consolidated_shorts_csv`` raises
``METRICOOL-CSV-NO-MEDIA`` (default ``require_media=True``) when any row lacks
a media URL, and tolerates blanks only when the caller explicitly opts out
(``require_media=False`` — the deferred audit path).
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv


def _make_post(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        caption=f"Short {short_index} caption",
        first_comment="",
        youtube_title=f"Short {short_index}",
        youtube_tags="",
        tiktok_title=f"Short {short_index}",
        pinterest_board="",
    )


def _make_slot(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        date="2026-04-29",
        time="16:00:00",
        datetime_est="2026-04-29T16:00:00-04:00",
        day_of_week="Wednesday",
    )


def _build(tmp_path: Path, *, drive_links, require_media=True):
    return build_consolidated_shorts_csv(
        output_path=tmp_path / "shorts.csv",
        shorts_by_lf={1: [_make_post(1), _make_post(2)]},
        slots_by_short_index={1: _make_slot(1), 2: _make_slot(2)},
        drive_links_by_short_index=drive_links,
        parent_lf_video_id_by_short_index={1: "vid1", 2: "vid1"},
        parent_recording_date_by_short_index={1: "2025-09-14", 2: "2025-09-14"},
        require_media=require_media,
    )


class TestMetricoolMediaGate:
    def test_all_blank_media_raises(self, tmp_path: Path):
        # No Drive links at all.
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-MEDIA"):
            _build(tmp_path, drive_links={})
        assert not (tmp_path / "shorts.csv").exists()  # dead CSV never written

    def test_partial_blank_media_raises_and_names_rows(self, tmp_path: Path):
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-MEDIA") as exc:
            _build(tmp_path, drive_links={1: "https://drive.example/1"})
        # short_index 2 has no link → its row (index 1) is named.
        assert "[1]" in str(exc.value)

    def test_full_media_passes(self, tmp_path: Path):
        out = _build(
            tmp_path,
            drive_links={1: "https://drive.example/1", 2: "https://drive.example/2"},
        )
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert [r["Picture Url 1"] for r in rows] == [
            "https://drive.example/1",
            "https://drive.example/2",
        ]

    def test_require_media_false_tolerates_blanks(self, tmp_path: Path):
        # The deferred-audit opt-out: write copy-review CSV without media.
        out = _build(tmp_path, drive_links={}, require_media=False)
        assert out.exists()
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert all(r["Picture Url 1"] == "" for r in rows)


class TestEmitDeferralPath:
    """Phase D's ``_emit_consolidated_shorts_csv`` must DEFER (skip the write)
    when shorts weren't Drive-uploaded under ``skip_drive_upload``, and let the
    media gate RAISE on a real upload run that produced no links.
    """

    def _inputs(self):
        from gomoveshift.pipeline.drive_phases import phase_d

        drive_plan = SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)])
        social_posts_per_lf = {1: [_make_post(1), _make_post(2)]}
        drive_schedule = SimpleNamespace(shorts_slots=[_make_slot(1), _make_slot(2)])
        return phase_d, dict(
            drive_plan=drive_plan,
            social_posts_per_lf=social_posts_per_lf,
            drive_schedule=drive_schedule,
            youtube_uploads={1: {"video_id": "vid1"}},
            per_lf_trim_ratio={1: 1.0},
            per_lf_recording_date={1: "2025-09-14"},
        )

    def test_skip_upload_no_links_defers(self, tmp_path: Path):
        phase_d, kw = self._inputs()
        out = phase_d._emit_consolidated_shorts_csv(
            output_base=tmp_path,
            project_base="sc",
            upload_manifests={},  # nothing uploaded
            skip_drive_upload=True,
            **kw,
        )
        assert out is None
        assert not (tmp_path / "sc_shorts_metricool.csv").exists()

    def test_real_run_no_links_raises(self, tmp_path: Path):
        phase_d, kw = self._inputs()
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-MEDIA"):
            phase_d._emit_consolidated_shorts_csv(
                output_base=tmp_path,
                project_base="sc",
                upload_manifests={},  # upload was expected but produced nothing
                skip_drive_upload=False,
                **kw,
            )
