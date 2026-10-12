"""Unit tests for Phase D reconstruction + CSV-emit helpers.

Extracts two helpers from ``PhaseD.run`` for testability:

* ``_reconstruct_drive_plan`` — JSON-resumed dict → ``DrivePlan``,
  with pass-through for non-dict and graceful None on parse failure.
* ``_emit_consolidated_shorts_csv`` — builds the per-short maps
  (drive links, slots, parent video ids + trim ratios, recording
  dates) and delegates to ``build_consolidated_shorts_csv``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    DrivePlanReconstructionError,
    _emit_consolidated_shorts_csv,
    _reconstruct_drive_plan,
)


# ─── _reconstruct_drive_plan ────────────────────────────────────────────


class TestReconstructDrivePlan:
    def test_none_passes_through(self):
        assert _reconstruct_drive_plan(None) is None

    def test_already_typed_passes_through(self):
        sentinel = SimpleNamespace(regular_lfs=[], source_files=[])
        assert _reconstruct_drive_plan(sentinel) is sentinel

    def test_minimal_dict_reconstructs(self):
        plan = _reconstruct_drive_plan(
            {
                "source_files": ["/x/a.mp4"],
                "processed_files": ["/x/b.mp4"],
                "regular_lfs": [],
                "score_threshold": 0.5,
                "total_source_duration": 3600.0,
                "total_highlight_duration": 1800.0,
                "best_moments_filtered": True,
            }
        )
        from gomoveshift.models import DrivePlan

        assert isinstance(plan, DrivePlan)
        assert plan.source_files == [Path("/x/a.mp4")]
        assert plan.processed_files == [Path("/x/b.mp4")]
        assert plan.regular_lfs == []
        assert plan.score_threshold == 0.5
        assert plan.total_source_duration == 3600.0
        assert plan.total_highlight_duration == 1800.0
        assert plan.best_moments_filtered is True

    def test_defaults_when_keys_missing(self):
        plan = _reconstruct_drive_plan({})
        from gomoveshift.models import DrivePlan

        assert isinstance(plan, DrivePlan)
        assert plan.source_files == []
        assert plan.regular_lfs == []
        assert plan.score_threshold == 0.0
        assert plan.best_moments_filtered is False

    def test_parse_failure_raises_and_logs(self, caplog):
        # An int in source_files makes Path(...) succeed (Path(0) is a
        # valid Path("0")), so to force failure we hand a non-dict to
        # the regular_lfs list which RegularLFPlan.from_dict rejects.
        #
        # A failed reconstruction must be distinguishable from a
        # legitimately empty plan — raising instead of returning a
        # bare None lets the caller fail loudly instead of silently
        # wiping persisted scheduling state.
        with caplog.at_level("ERROR"):
            with pytest.raises(DrivePlanReconstructionError):
                _reconstruct_drive_plan({"regular_lfs": ["not-a-dict"]})
        assert any("Failed to reconstruct DrivePlan" in r.getMessage() for r in caplog.records)


# ─── _emit_consolidated_shorts_csv ──────────────────────────────────────


class TestEmitConsolidatedShortsCsv:
    def _mk_post(self, *, short_index):
        return SimpleNamespace(short_index=short_index)

    def _mk_drive_plan(self, lf_indices):
        return SimpleNamespace(regular_lfs=[SimpleNamespace(index=i) for i in lf_indices])

    def test_delegates_to_builder_with_expected_path(self, tmp_path):
        drive_plan = self._mk_drive_plan([1])
        posts_lf1 = [self._mk_post(short_index=10), self._mk_post(short_index=11)]
        social_posts_per_lf = {1: posts_lf1}
        upload_manifests = {}
        drive_schedule = SimpleNamespace(shorts_slots=[])
        youtube_uploads = {1: {"video_id": "vid-1"}}
        per_lf_trim_ratio = {1: 0.4}
        per_lf_recording_date = {1: "2026-03-15"}

        with patch("gomoveshift.pipeline.csv_generator.build_consolidated_shorts_csv") as builder:
            csv_path = _emit_consolidated_shorts_csv(
                output_base=tmp_path,
                project_base="myproj",
                drive_plan=drive_plan,
                social_posts_per_lf=social_posts_per_lf,
                upload_manifests=upload_manifests,
                drive_schedule=drive_schedule,
                youtube_uploads=youtube_uploads,
                per_lf_trim_ratio=per_lf_trim_ratio,
                per_lf_recording_date=per_lf_recording_date,
            )
        assert csv_path == tmp_path / "myproj_shorts_metricool.csv"
        builder.assert_called_once()
        kwargs = builder.call_args.kwargs
        assert kwargs["output_path"] == csv_path
        # shorts_by_lf bucketed by lf.index
        assert kwargs["shorts_by_lf"] == {1: posts_lf1}
        # parent video id propagated to every short under that LF
        assert kwargs["parent_lf_video_id_by_short_index"] == {10: "vid-1", 11: "vid-1"}
        # trim ratio propagated as float to every short under that LF
        assert kwargs["parent_trim_ratio_by_short_index"] == {10: 0.4, 11: 0.4}

    def test_missing_youtube_upload_yields_blank_video_id(self, tmp_path):
        drive_plan = self._mk_drive_plan([1, 2])
        social_posts_per_lf = {
            1: [self._mk_post(short_index=10)],
            2: [self._mk_post(short_index=20)],
        }
        with patch("gomoveshift.pipeline.csv_generator.build_consolidated_shorts_csv") as builder:
            _emit_consolidated_shorts_csv(
                output_base=tmp_path,
                project_base="p",
                drive_plan=drive_plan,
                social_posts_per_lf=social_posts_per_lf,
                upload_manifests={},
                drive_schedule=SimpleNamespace(shorts_slots=[]),
                youtube_uploads={1: {"video_id": "vid-1"}},
                per_lf_trim_ratio={1: 0.5, 2: 0.5},
                per_lf_recording_date={1: "2026-03-15", 2: "2026-03-16"},
            )
        kwargs = builder.call_args.kwargs
        assert kwargs["parent_lf_video_id_by_short_index"] == {10: "vid-1", 20: ""}

    def test_non_numeric_trim_ratio_omitted_from_map(self, tmp_path):
        drive_plan = self._mk_drive_plan([1])
        social_posts_per_lf = {1: [self._mk_post(short_index=10)]}
        with patch("gomoveshift.pipeline.csv_generator.build_consolidated_shorts_csv") as builder:
            _emit_consolidated_shorts_csv(
                output_base=tmp_path,
                project_base="p",
                drive_plan=drive_plan,
                social_posts_per_lf=social_posts_per_lf,
                upload_manifests={},
                drive_schedule=SimpleNamespace(shorts_slots=[]),
                youtube_uploads={},
                per_lf_trim_ratio={1: None},  # type: ignore[dict-item]
                per_lf_recording_date={1: "2026-03-15"},
            )
        kwargs = builder.call_args.kwargs
        assert kwargs["parent_trim_ratio_by_short_index"] == {}

    def test_slots_by_short_index_keyed_from_schedule(self, tmp_path):
        slot_a = SimpleNamespace(short_index=10, datetime_est="2026-03-15T08:00:00")
        slot_b = SimpleNamespace(short_index=11, datetime_est="2026-03-16T08:00:00")
        drive_schedule = SimpleNamespace(shorts_slots=[slot_a, slot_b])
        drive_plan = self._mk_drive_plan([1])
        social_posts_per_lf = {1: [self._mk_post(short_index=10), self._mk_post(short_index=11)]}
        with patch("gomoveshift.pipeline.csv_generator.build_consolidated_shorts_csv") as builder:
            _emit_consolidated_shorts_csv(
                output_base=tmp_path,
                project_base="p",
                drive_plan=drive_plan,
                social_posts_per_lf=social_posts_per_lf,
                upload_manifests={},
                drive_schedule=drive_schedule,
                youtube_uploads={},
                per_lf_trim_ratio={},
                per_lf_recording_date={1: "2026-03-15"},
            )
        kwargs = builder.call_args.kwargs
        assert kwargs["slots_by_short_index"] == {10: slot_a, 11: slot_b}

    def test_missing_recording_date_for_lf_with_shorts_raises(self, tmp_path):
        # _build_recording_dates_by_short_index enforces an invariant when
        # a LF has shorts but no date. This is the contract that flows
        # through the helper unchanged.
        drive_plan = self._mk_drive_plan([1])
        social_posts_per_lf = {1: [self._mk_post(short_index=10)]}
        with patch("gomoveshift.pipeline.csv_generator.build_consolidated_shorts_csv"):
            with pytest.raises(ValueError, match="per_lf_recording_date"):
                _emit_consolidated_shorts_csv(
                    output_base=tmp_path,
                    project_base="p",
                    drive_plan=drive_plan,
                    social_posts_per_lf=social_posts_per_lf,
                    upload_manifests={},
                    drive_schedule=SimpleNamespace(shorts_slots=[]),
                    youtube_uploads={},
                    per_lf_trim_ratio={},
                    per_lf_recording_date={},  # missing!
                )

    def test_first_party_route_forces_matching_toggle_false_in_platforms_kwarg(self, tmp_path):
        # Threading config.drive through must suppress the Metricool toggle
        # for whichever platform is on the first_party route -- otherwise
        # Phase D's publish-plan sidecar (first-party) and this CSV
        # (Metricool) both post the same short.
        from gomoveshift.config import DriveConfig

        drive_plan = self._mk_drive_plan([1])
        social_posts_per_lf = {1: [self._mk_post(short_index=10)]}
        config = SimpleNamespace(drive=DriveConfig(publishing_route_youtube="first_party"))
        with patch("gomoveshift.pipeline.csv_generator.build_consolidated_shorts_csv") as builder:
            _emit_consolidated_shorts_csv(
                output_base=tmp_path,
                project_base="p",
                drive_plan=drive_plan,
                social_posts_per_lf=social_posts_per_lf,
                upload_manifests={},
                drive_schedule=SimpleNamespace(shorts_slots=[]),
                youtube_uploads={},
                per_lf_trim_ratio={},
                per_lf_recording_date={1: "2026-03-15"},
                config=config,
            )
        kwargs = builder.call_args.kwargs
        assert kwargs["platforms"]["YouTube"] == "FALSE"
        assert kwargs["platforms"]["Instagram"] == "TRUE"

    def test_all_off_routes_leave_platforms_at_defaults(self, tmp_path):
        from gomoveshift.config import DriveConfig
        from gomoveshift.pipeline.csv_generator import DEFAULT_PLATFORMS

        drive_plan = self._mk_drive_plan([1])
        social_posts_per_lf = {1: [self._mk_post(short_index=10)]}
        config = SimpleNamespace(drive=DriveConfig())
        with patch("gomoveshift.pipeline.csv_generator.build_consolidated_shorts_csv") as builder:
            _emit_consolidated_shorts_csv(
                output_base=tmp_path,
                project_base="p",
                drive_plan=drive_plan,
                social_posts_per_lf=social_posts_per_lf,
                upload_manifests={},
                drive_schedule=SimpleNamespace(shorts_slots=[]),
                youtube_uploads={},
                per_lf_trim_ratio={},
                per_lf_recording_date={1: "2026-03-15"},
                config=config,
            )
        kwargs = builder.call_args.kwargs
        assert kwargs["platforms"] == DEFAULT_PLATFORMS
