"""Skip-upload / audit runs write an UNPUBLISHABLE Shorts payload preview.

The real ``<project>_publish_plan.json`` stays deferred on those runs (pinned in
``test_phase_d_publish_plan_sidecar.py``); the preview is a separate file no
consumer globs, carries a refusal marker, and flags every placeholder field.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    _emit_publish_plan_sidecar,
    _emit_publish_preview_sidecar,
)
from gomoveshift.publishing.plan_io import PublishPlanFormatError, load_publish_plan

sys.path.insert(0, str(Path(__file__).parent))

from test_phase_d_publish_plan_sidecar import FINGERPRINT, FP, _base_kwargs  # noqa: E402

PREVIEW_NAME = "p_publish_preview.json"
LF_PUBLISH = "2026-03-14T08:00:00-05:00"


def _kwargs(tmp_path, **over):
    kw = _base_kwargs(tmp_path, publishing_route_youtube="first_party")
    kw.update(over)
    return kw


def _no_id():
    return {1: {}}


def _write(tmp_path, **over):
    kw = _kwargs(tmp_path, **over)
    drive_cfg = kw["config"].drive
    return _emit_publish_preview_sidecar(
        output_base=tmp_path,
        project_base=kw["project_base"],
        route_selected=("youtube",),
        drive_cfg=drive_cfg,
        drive_plan=kw["drive_plan"],
        social_posts_per_lf=kw["social_posts_per_lf"],
        drive_schedule=kw["drive_schedule"],
        youtube_uploads=kw["youtube_uploads"],
        per_lf_recording_date=kw["per_lf_recording_date"],
        shorts_paths_by_lf=kw["shorts_paths_by_lf"],
    )


class TestIsolation:
    def test_filename_matches_no_consumer_glob_or_convention(self):
        for pattern in ("*_publish_plan.json", "p_publish_plan.json"):
            assert not fnmatch.fnmatch(PREVIEW_NAME, pattern)
        assert "_publish_plan" not in PREVIEW_NAME

    def test_audit_pass_writes_preview_not_plan(self, tmp_path):
        with patch(FINGERPRINT, return_value=FP):
            result = _emit_publish_plan_sidecar(
                output_base=tmp_path, **_kwargs(tmp_path, publish_pass=False)
            )
        assert result is None  # still deferred
        assert (tmp_path / PREVIEW_NAME).exists()
        assert not (tmp_path / "p_publish_plan.json").exists()
        assert list(tmp_path.glob("*_publish_plan.json")) == []

    def test_skip_upload_publish_pass_writes_preview_not_plan(self, tmp_path):
        with patch(FINGERPRINT, return_value=FP):
            result = _emit_publish_plan_sidecar(
                output_base=tmp_path, **_kwargs(tmp_path, youtube_uploads=_no_id())
            )
        assert result is None
        assert (tmp_path / PREVIEW_NAME).exists()
        assert not (tmp_path / "p_publish_plan.json").exists()

    def test_real_publish_pass_writes_no_preview(self, tmp_path):
        with patch(FINGERPRINT, return_value=FP):
            result = _emit_publish_plan_sidecar(output_base=tmp_path, **_kwargs(tmp_path))
        assert result == tmp_path / "p_publish_plan.json"
        assert not (tmp_path / PREVIEW_NAME).exists()

    def test_all_metricool_run_writes_no_preview(self, tmp_path):
        kw = _base_kwargs(tmp_path)  # every route off
        kw["publish_pass"] = False
        with patch(FINGERPRINT) as fp:
            _emit_publish_plan_sidecar(output_base=tmp_path, **kw)
        fp.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    def test_preview_writer_returns_no_plan_path(self, tmp_path):
        assert _write(tmp_path, publish_pass=False) is None

    def test_load_publish_plan_refuses_preview(self, tmp_path):
        _write(tmp_path, publish_pass=False)
        with pytest.raises(PublishPlanFormatError, match="publish_preview"):
            load_publish_plan(tmp_path / PREVIEW_NAME)

    def test_marker_fields(self, tmp_path):
        _write(tmp_path, publish_pass=False)
        raw = json.loads((tmp_path / PREVIEW_NAME).read_text())
        assert raw["kind"] == "publish_preview"
        assert raw["not_for_publish"] is True


class TestParityWithRealPlan:
    def test_payloads_equal_real_plan_except_parent_fields(self, tmp_path):
        real_dir, prev_dir = tmp_path / "real", tmp_path / "prev"
        real_dir.mkdir()
        prev_dir.mkdir()
        with patch(FINGERPRINT, return_value=FP):
            real_path = _emit_publish_plan_sidecar(
                output_base=real_dir,
                **_kwargs(
                    real_dir, youtube_uploads={1: {"video_id": "vid-1", "publish_at": LF_PUBLISH}}
                ),
            )
        _write(prev_dir, publish_pass=False, youtube_uploads=_no_id())
        real = json.loads(real_path.read_text())["shorts"][0]
        prev = json.loads((prev_dir / PREVIEW_NAME).read_text())["shorts"][0]
        prev.pop("placeholders")
        for rec in (real, prev):
            for key in ("parent_lf_video_id", "parent_lf_publish_at"):
                rec.pop(key)
            rec["payloads"]["youtube"].pop("description")
            rec.pop("media")
        assert prev == real

    def test_real_description_body_kept_with_synthetic_link(self, tmp_path):
        _write(tmp_path, publish_pass=False, youtube_uploads=_no_id())
        rec = json.loads((tmp_path / PREVIEW_NAME).read_text())["shorts"][0]
        assert rec["parent_lf_video_id"] == "PREVIEW-NO-LF-ID-1"
        assert "PREVIEW-NO-LF-ID-1" in rec["payloads"]["youtube"]["description"]


class TestPlaceholders:
    def test_missing_id_and_no_lf_slot_lists_both_placeholders(self, tmp_path):
        _write(tmp_path, publish_pass=False, youtube_uploads=_no_id())
        rec = json.loads((tmp_path / PREVIEW_NAME).read_text())["shorts"][0]
        assert rec["placeholders"] == [
            "parent_lf_video_id",
            "parent_lf_publish_at",
            "payloads.youtube.description",
            "media",
        ]

    def test_missing_publish_at_uses_scheduled_lf_slot(self, tmp_path):
        from types import SimpleNamespace

        sched = _kwargs(tmp_path)["drive_schedule"]
        sched.lf_slots = [SimpleNamespace(datetime_est="2026-03-13T08:00:00-05:00")]
        _write(
            tmp_path,
            publish_pass=False,
            drive_schedule=sched,
            youtube_uploads={1: {"video_id": "vid-1"}},
        )
        rec = json.loads((tmp_path / PREVIEW_NAME).read_text())["shorts"][0]
        assert rec["parent_lf_publish_at"].startswith("2026-03-13T13:00:00")
        assert rec["placeholders"] == ["parent_lf_publish_at", "media"]

    def test_real_ids_yield_no_placeholders(self, tmp_path):
        _write(tmp_path, publish_pass=False)
        rec = json.loads((tmp_path / PREVIEW_NAME).read_text())["shorts"][0]
        assert rec["placeholders"] == ["media"]


class TestFailureIsNonFatal:
    def test_build_error_warns_loudly_and_writes_nothing(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            with patch(
                "gomoveshift.publishing.plan_builder.build_publish_plan",
                side_effect=ValueError("title gate boom"),
            ):
                result = _write(tmp_path, publish_pass=False)
        assert result is None
        assert not (tmp_path / PREVIEW_NAME).exists()
        assert "PREVIEW" in caplog.text and "title gate boom" in caplog.text

    def test_preview_never_fingerprints_media(self, tmp_path):
        with patch(FINGERPRINT) as fp:
            _emit_publish_plan_sidecar(
                output_base=tmp_path, **_kwargs(tmp_path, publish_pass=False)
            )
        fp.assert_not_called()
        rec = json.loads((tmp_path / PREVIEW_NAME).read_text())["shorts"][0]
        assert rec["media"]["sha256"] == "PREVIEW-NOT-HASHED"

    def test_log_says_preview(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            _write(tmp_path, publish_pass=False)
        assert "D-publish-preview" in caplog.text
