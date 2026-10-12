"""Unit tests for Phase D upload-loop helpers.

Smoke tests pinning the extracted helpers' contracts:

* ``_assert_shorts_quality_gate`` — 1080x1920 enforcement gate.
* ``_upload_lfs_to_youtube`` — per-LF YouTube upload loop with
  schedule-date selection and quota error handling.
* ``_generate_lf_thumbnails`` — generation + canonical-path promotion
  (runs upstream of every upload).
* ``_attach_lf_thumbnails`` — conditional YouTube attach (post-upload).
* ``_upload_lf_thumbnails_to_drive`` — Drive upload of canonical thumb
  JPGs, idempotent across resumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import phase_d
from gomoveshift.pipeline.drive_phases.phase_d import (
    _assert_shorts_quality_gate,
    _attach_lf_thumbnails,
    _generate_lf_thumbnails,
    _upload_lf_thumbnails_to_drive,
    _upload_lfs_to_youtube,
)
from gomoveshift.pipeline.exceptions import YouTubeQuotaError

from _state_helpers import make_state


def _lf(idx, *, total_duration=1800.0):
    return SimpleNamespace(index=idx, total_duration=total_duration)


# ─── _assert_shorts_quality_gate ────────────────────────────────────────


class TestAssertShortsQualityGate:
    def test_no_shorts_state_is_noop(self):
        _assert_shorts_quality_gate(make_state(), validator_cls=MagicMock)

    def test_skip_flag_bypasses(self):
        validator_cls = MagicMock()
        state = make_state(shorts_per_lf={1: ["/x/s.mp4"]})
        state.skip_shorts_quality_gate = True
        _assert_shorts_quality_gate(state, validator_cls=validator_cls)
        validator_cls.assert_not_called()

    def test_pass_path_runs_validator_and_logs(self, caplog):
        validator = MagicMock()
        validator.validate_short_files.return_value = SimpleNamespace(
            passed=True, summary=lambda: ""
        )
        validator_cls = MagicMock(return_value=validator)
        with caplog.at_level("INFO"):
            _assert_shorts_quality_gate(
                make_state(shorts_per_lf={1: ["/x/s.mp4"]}),
                validator_cls=validator_cls,
            )
        validator.validate_short_files.assert_called_once()
        assert any("ALL 1 shorts pass" in r.getMessage() for r in caplog.records)

    def test_fail_path_raises_runtime_error(self):
        validator = MagicMock()
        validator.validate_short_files.return_value = SimpleNamespace(
            passed=False, summary=lambda: "16:9 detected"
        )
        validator_cls = MagicMock(return_value=validator)
        with pytest.raises(RuntimeError, match="Shorts quality gate FAILED"):
            _assert_shorts_quality_gate(
                make_state(shorts_per_lf={1: ["/x/s.mp4"]}),
                validator_cls=validator_cls,
            )

    def test_populates_short_durations_per_lf_after_validation(self):
        """The gate ffprobes each validated short and records its
        duration into ``state.short_durations_per_lf`` so preflight stops
        falling back to the 30s default."""
        validator = MagicMock()
        validator.validate_short_files.return_value = SimpleNamespace(
            passed=True, summary=lambda: ""
        )
        validator_cls = MagicMock(return_value=validator)
        durations = {"/x/s1.mp4": 31.5, "/x/s2.mp4": 28.0, "/y/s3.mp4": 42.0}
        state = make_state(shorts_per_lf={1: ["/x/s1.mp4", "/x/s2.mp4"], 2: ["/y/s3.mp4"]})
        _assert_shorts_quality_gate(
            state,
            validator_cls=validator_cls,
            probe_duration_fn=lambda p: durations[str(p)],
        )
        assert set(state.short_durations_per_lf.keys()) == {1, 2}
        assert state.short_durations_per_lf[1] == [31.5, 28.0]
        assert state.short_durations_per_lf[2] == [42.0]

    def test_skips_unprobeable_short_without_failing(self):
        """A None ffprobe result (missing/corrupt file) is dropped from the
        per-LF list rather than poisoning the estimate or raising."""
        validator = MagicMock()
        validator.validate_short_files.return_value = SimpleNamespace(
            passed=True, summary=lambda: ""
        )
        validator_cls = MagicMock(return_value=validator)
        state = make_state(shorts_per_lf={1: ["/x/ok.mp4", "/x/bad.mp4"]})
        _assert_shorts_quality_gate(
            state,
            validator_cls=validator_cls,
            probe_duration_fn=lambda p: 30.0 if str(p).endswith("ok.mp4") else None,
        )
        assert state.short_durations_per_lf == {1: [30.0]}

    def test_no_durations_recorded_when_gate_skipped(self):
        validator_cls = MagicMock()
        state = make_state(shorts_per_lf={1: ["/x/s.mp4"]})
        state.skip_shorts_quality_gate = True
        _assert_shorts_quality_gate(state, validator_cls=validator_cls)
        assert not getattr(state, "short_durations_per_lf", None)


# ─── _upload_lfs_to_youtube ─────────────────────────────────────────────


class TestUploadLfsToYoutube:
    def _mk_inputs(self, *, lf_count=2, lf_slots=None):
        drive_plan = SimpleNamespace(regular_lfs=[_lf(i) for i in range(1, lf_count + 1)])
        regular_lf_paths = [f"/v/lf{i}.mp4" for i in range(1, lf_count + 1)]
        metadata_per_lf = {
            i: {"title": f"T{i}", "description": "d", "tags": []} for i in range(1, lf_count + 1)
        }
        drive_schedule = SimpleNamespace(
            lf_slots=[SimpleNamespace(datetime_est="2026-03-15T11:00:00") for _ in range(lf_count)]
            if lf_slots is None
            else lf_slots
        )
        return drive_plan, regular_lf_paths, metadata_per_lf, drive_schedule

    def test_skip_flag_returns_none_uploader(self, tmp_path):
        drive_plan, paths, meta, sched = self._mk_inputs()
        yt, uploads = _upload_lfs_to_youtube(
            drive_plan=drive_plan,
            regular_lf_paths=paths,
            metadata_per_lf=meta,
            drive_schedule=sched,
            youtube_uploads={},
            state=make_state(),
            output_base=tmp_path,
            config=MagicMock(),
            save_state=lambda s: None,
            skip_youtube_upload=True,
            youtube_upload_enabled=True,
        )
        assert yt is None
        assert uploads == {}

    def test_disabled_returns_none_uploader(self, tmp_path):
        drive_plan, paths, meta, sched = self._mk_inputs()
        yt, _ = _upload_lfs_to_youtube(
            drive_plan=drive_plan,
            regular_lf_paths=paths,
            metadata_per_lf=meta,
            drive_schedule=sched,
            youtube_uploads={},
            state=make_state(),
            output_base=tmp_path,
            config=MagicMock(),
            save_state=lambda s: None,
            skip_youtube_upload=False,
            youtube_upload_enabled=False,
        )
        assert yt is None

    def test_success_path_records_uploads_and_invokes_save_state(self, tmp_path):
        drive_plan, paths, meta, sched = self._mk_inputs(lf_count=1)
        save_calls = []
        yt_mock = MagicMock()
        with (
            patch.object(
                esc,
                "_resolve_youtube_uploader_class",
                return_value=MagicMock(return_value=yt_mock),
            ),
            patch.object(esc, "_upload_lf_to_youtube", return_value={"video_id": "abc"}),
        ):
            yt, uploads = _upload_lfs_to_youtube(
                drive_plan=drive_plan,
                regular_lf_paths=paths,
                metadata_per_lf=meta,
                drive_schedule=sched,
                youtube_uploads={},
                state=make_state(),
                output_base=tmp_path,
                config=MagicMock(),
                save_state=lambda s: save_calls.append(s),
                skip_youtube_upload=False,
                youtube_upload_enabled=True,
            )
        assert yt is yt_mock
        assert uploads == {1: {"video_id": "abc"}}
        assert save_calls  # save_state called at least once
        yt_mock.set_operator_checklist_path.assert_called_once_with(
            tmp_path / "operator_checklist.txt"
        )

    def test_quota_error_persists_state_and_reraises(self, tmp_path):
        drive_plan, paths, meta, sched = self._mk_inputs(lf_count=2)
        save_calls = []
        yt_mock = MagicMock()
        quota = YouTubeQuotaError("daily quota exceeded")
        with (
            patch.object(
                esc,
                "_resolve_youtube_uploader_class",
                return_value=MagicMock(return_value=yt_mock),
            ),
            patch.object(
                esc,
                "_upload_lf_to_youtube",
                side_effect=[{"video_id": "ok"}, quota],
            ),
        ):
            state = make_state()
            with pytest.raises(YouTubeQuotaError):
                _upload_lfs_to_youtube(
                    drive_plan=drive_plan,
                    regular_lf_paths=paths,
                    metadata_per_lf=meta,
                    drive_schedule=sched,
                    youtube_uploads={},
                    state=state,
                    output_base=tmp_path,
                    config=MagicMock(),
                    save_state=lambda s: save_calls.append(s),
                    skip_youtube_upload=False,
                    youtube_upload_enabled=True,
                )
        assert state["youtube_uploads"] == {1: {"video_id": "ok"}}
        assert save_calls

    def test_record_none_skips_assignment(self, tmp_path):
        drive_plan, paths, meta, sched = self._mk_inputs(lf_count=2)
        yt_mock = MagicMock()
        with (
            patch.object(
                esc,
                "_resolve_youtube_uploader_class",
                return_value=MagicMock(return_value=yt_mock),
            ),
            patch.object(
                esc,
                "_upload_lf_to_youtube",
                side_effect=[None, {"video_id": "ok"}],
            ),
        ):
            _, uploads = _upload_lfs_to_youtube(
                drive_plan=drive_plan,
                regular_lf_paths=paths,
                metadata_per_lf=meta,
                drive_schedule=sched,
                youtube_uploads={},
                state=make_state(),
                output_base=tmp_path,
                config=MagicMock(),
                save_state=lambda s: None,
                skip_youtube_upload=False,
                youtube_upload_enabled=True,
            )
        # LF 1 returned None → not recorded. LF 2 succeeded.
        assert uploads == {2: {"video_id": "ok"}}


# ─── _generate_lf_thumbnails ────────────────────────────────────────────


class TestGenerateLfThumbnails:
    def test_empty_plan_returns_silently(self, tmp_path):
        _generate_lf_thumbnails(
            drive_plan=SimpleNamespace(regular_lfs=[]),
            regular_lf_paths=[],
            metadata_per_lf={},
            state=make_state(),
            output_base=tmp_path,
        )
        assert not (tmp_path / "thumbnails").exists()

    def test_canonical_path_written(self, tmp_path):
        # Helper cleans thumb_subdir before generation, so the mock must
        # write the variant file inside it on call (the real generator
        # creates output_dir as part of its work).
        def _fake_gen(*, output_dir, **kw):
            output_dir.mkdir(parents=True, exist_ok=True)
            variant = output_dir / "v.jpg"
            variant.write_bytes(b"jpegbytes")
            return variant

        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=_fake_gen,
        ):
            _generate_lf_thumbnails(
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=["/v/lf1.mp4"],
                metadata_per_lf={1: {"title": "T1"}},
                state=make_state(per_lf_ost_text={1: "Canonical OST"}),
                output_base=tmp_path,
            )
        canonical = tmp_path / "thumbnails" / "lf_01.jpg"
        assert canonical.exists()
        assert canonical.read_bytes() == b"jpegbytes"


# ─── _attach_lf_thumbnails ──────────────────────────────────────────────


class TestAttachLfThumbnails:
    def _attach(self, tmp_path, uploads, state=None):
        canonical = tmp_path / "thumbnails" / "lf_01.jpg"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"jpegbytes")
        yt_mock = MagicMock()
        _attach_lf_thumbnails(
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            output_base=tmp_path,
            yt=yt_mock,
            youtube_uploads=uploads,
            state=state,
        )
        return yt_mock, canonical

    def test_attach_when_upload_present(self, tmp_path):
        uploads = {1: {"video_id": "vid-1"}}
        yt_mock, canonical = self._attach(tmp_path, uploads)
        yt_mock.set_thumbnail.assert_called_once_with("vid-1", str(canonical))
        assert uploads[1]["thumbnail_path"] == str(canonical)

    def test_no_attach_when_no_upload_record(self, tmp_path):
        yt_mock, _ = self._attach(tmp_path, {})
        yt_mock.set_thumbnail.assert_not_called()


class TestAttachLfThumbnailsMasterIdentity:
    """Same cut-from-master identity + attestation cross-check as the
    LF provenance sidecar, on the thumbnail upload entry. Best-effort,
    never gating — absent ``state`` (legacy callers) degrades to
    null-with-reason rather than raising."""

    def _write_attestation(self, tmp_path, date, master_fingerprint="123:456"):
        day_dir = tmp_path / "_stages" / f"day_{date}"
        day_dir.mkdir(parents=True)
        payload = {"schema": "gms.privacy_attestation/v1", "date": date}
        if master_fingerprint is not None:
            payload["master_fingerprint"] = master_fingerprint
        (day_dir / f"day_{date}_privacy_attestation.json").write_text(json.dumps(payload))

    def _run(self, tmp_path, *, state=None):
        canonical = tmp_path / "thumbnails" / "lf_01.jpg"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"jpegbytes")
        uploads = {1: {"video_id": "vid-1"}}
        _attach_lf_thumbnails(
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            output_base=tmp_path,
            yt=MagicMock(),
            youtube_uploads=uploads,
            state=state,
        )
        return uploads[1]

    def test_matching_fingerprint_verifies_true(self, tmp_path):
        self._write_attestation(tmp_path, "2026-03-01", master_fingerprint="123:456")
        state = make_state(
            per_lf_recording_date={1: "2026-03-01"},
            per_day_completed={"2026-03-01": {"master_fingerprint": "123:456"}},
        )
        entry = self._run(tmp_path, state=state)
        assert entry["cut_from_master_fingerprint"] == "123:456"
        assert entry["master_attestation_verified"] is True

    def test_mismatched_fingerprint_verifies_false(self, tmp_path):
        self._write_attestation(tmp_path, "2026-03-01", master_fingerprint="999:999")
        state = make_state(
            per_lf_recording_date={1: "2026-03-01"},
            per_day_completed={"2026-03-01": {"master_fingerprint": "123:456"}},
        )
        entry = self._run(tmp_path, state=state)
        assert entry["master_attestation_verified"] is False
        assert entry["null_reasons"].get("master_attestation_verified")

    def test_no_attestation_on_disk_is_none(self, tmp_path):
        state = make_state(
            per_lf_recording_date={1: "2026-03-01"},
            per_day_completed={"2026-03-01": {"master_fingerprint": "123:456"}},
        )
        entry = self._run(tmp_path, state=state)
        assert entry["master_attestation_verified"] is None
        assert entry["null_reasons"].get("master_attestation_verified")

    def test_no_state_is_none_with_reason(self, tmp_path):
        entry = self._run(tmp_path, state=None)
        assert entry["cut_from_master_fingerprint"] is None
        assert entry["null_reasons"].get("cut_from_master_fingerprint")


# ─── _upload_lf_thumbnails_to_drive ─────────────────────────────────────


class TestUploadLfThumbnailsToDrive:
    def test_skip_flag_persists_existing_urls(self, tmp_path):
        state = make_state(lf_thumbnail_drive_urls={1: "existing"})
        uploader = MagicMock()
        out = _upload_lf_thumbnails_to_drive(
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=state,
            output_base=tmp_path,
            project_base="p",
            uploader=uploader,
            uploader_factory=MagicMock(),
            skip_drive_upload=True,
        )
        assert state["lf_thumbnail_drive_urls"] == {1: "existing"}
        assert out is uploader

    def test_uploads_missing_idempotently(self, tmp_path):
        thumbs = tmp_path / "thumbnails"
        thumbs.mkdir()
        (thumbs / "lf_01.jpg").write_bytes(b"j")
        uploader = MagicMock()
        with patch.object(esc, "_resolve_drive_subfolder_id", return_value="folder-p_lf01"):
            uploader._upload_file.return_value = "https://drive/thumb1"
            state = make_state()
            out = _upload_lf_thumbnails_to_drive(
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                state=state,
                output_base=tmp_path,
                project_base="p",
                uploader=uploader,
                uploader_factory=MagicMock(),
                skip_drive_upload=False,
            )
        assert state["lf_thumbnail_drive_urls"] == {1: "https://drive/thumb1"}
        assert out is uploader

    def test_skips_lfs_with_existing_url(self, tmp_path):
        thumbs = tmp_path / "thumbnails"
        thumbs.mkdir()
        (thumbs / "lf_01.jpg").write_bytes(b"j")
        uploader = MagicMock()
        state = make_state(lf_thumbnail_drive_urls={1: "already"})
        with patch.object(esc, "_resolve_drive_subfolder_id", return_value="folder"):
            _upload_lf_thumbnails_to_drive(
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                state=state,
                output_base=tmp_path,
                project_base="p",
                uploader=uploader,
                uploader_factory=MagicMock(),
                skip_drive_upload=False,
            )
        uploader._upload_file.assert_not_called()
        assert state["lf_thumbnail_drive_urls"] == {1: "already"}

    def test_missing_canonical_thumb_logs_and_continues(self, tmp_path, caplog):
        thumbs = tmp_path / "thumbnails"
        thumbs.mkdir()
        # lf_01.jpg intentionally missing
        uploader = MagicMock()
        state = make_state()
        with caplog.at_level("WARNING"):
            _upload_lf_thumbnails_to_drive(
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                state=state,
                output_base=tmp_path,
                project_base="p",
                uploader=uploader,
                uploader_factory=MagicMock(),
                skip_drive_upload=False,
            )
        uploader._upload_file.assert_not_called()
        assert state["lf_thumbnail_drive_urls"] == {}
        assert any("canonical thumbnail missing" in r.getMessage() for r in caplog.records)

    def test_uploader_factory_invoked_when_uploader_none(self, tmp_path):
        thumbs = tmp_path / "thumbnails"
        thumbs.mkdir()
        (thumbs / "lf_01.jpg").write_bytes(b"j")
        new_uploader = MagicMock()
        new_uploader._upload_file.return_value = "https://drive/thumb"
        factory = MagicMock(return_value=new_uploader)
        with patch.object(esc, "_resolve_drive_subfolder_id", return_value="folder"):
            out = _upload_lf_thumbnails_to_drive(
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                state=make_state(),
                output_base=tmp_path,
                project_base="p",
                uploader=None,
                uploader_factory=factory,
                skip_drive_upload=False,
            )
        factory.assert_called_once()
        assert out is new_uploader
