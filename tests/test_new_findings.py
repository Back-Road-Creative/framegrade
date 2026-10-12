"""Tests for YouTube upload checkpointing,
NVENC detection in OST applier, day-first prereqs, wrap_text
duplicate-word correctness, Drive uploader chunksize/timeout,
and silent OST failure warning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config

from _state_helpers import make_state
from ost_subprocess_harness import monkeypatch_ost_run


# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class _FakeLF:
    index: int
    total_duration: float = 600.0


def _mock_ffprobe_response(
    width: int = 3840,
    height: int = 2160,
    duration: float = 600.0,
    fps: float = 29.97,
    bitrate_kbps: int = 53000,
):
    """Return fake ffprobe JSON stdout."""
    import json

    data = {
        "streams": [
            {
                "width": width,
                "height": height,
                "r_frame_rate": f"{int(fps * 1000)}/1000",
            }
        ],
        "format": {
            "duration": str(duration),
            "bit_rate": str(bitrate_kbps * 1000),
        },
    }
    return json.dumps(data).encode("utf-8")


# ── _wrap_text handles duplicate words correctly ─────────


class TestWrapTextDuplicateWords:
    """Verify that _wrap_text uses positional indexing (enumerate) so
    repeated words don't confuse the line-breaking algorithm."""

    def test_repeated_words_all_preserved(self):
        """Text like 'the road the road' keeps every word."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        text = "the road the road the road"
        lines = proc._wrap_text(text, fontsize=48, max_width_px=2000)
        rejoined = " ".join(lines)
        assert rejoined == text

    def test_repeated_words_wrap_correctly(self):
        """When wrapping, repeated words end up in the correct lines."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        # Force narrow width so wrapping is required
        text = "County Clare County Clare County Clare"
        lines = proc._wrap_text(text, fontsize=48, max_width_px=400, max_lines=3)
        # All words must be present
        rejoined = " ".join(lines)
        assert rejoined == text
        # Should actually wrap (more than 1 line at 400px)
        assert len(lines) >= 2

    def test_single_word_repeated(self):
        """Edge case: same word many times."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        text = "Ireland Ireland Ireland Ireland Ireland"
        lines = proc._wrap_text(text, fontsize=48, max_width_px=2000)
        assert " ".join(lines) == text


# ── Day-first prereq allows --phase phase_d ──────────────


class TestDayFirstPrereqAllowsPhaseD:
    """Verify _get_prereqs() returns phase_bc_day_first as prereq for phase_d.

    Legacy phase_b / phase_c / per_day paths were deleted; the
    surviving manifest is phase_a → phase_bc_day_first → phase_d.
    """

    def test_day_first_prereqs_include_phase_bc_day_first(self):
        from gomoveshift.pipeline.drive_orchestrator import (
            DrivePipelineOrchestrator,
        )

        orch = DrivePipelineOrchestrator(Config())
        prereqs = orch._get_prereqs()
        assert "phase_bc_day_first" in prereqs
        assert "phase_bc_day_first" in prereqs["phase_d"]

    def test_day_first_prereqs_do_not_require_phase_b_or_c(self):
        from gomoveshift.pipeline.drive_orchestrator import (
            DrivePipelineOrchestrator,
        )

        orch = DrivePipelineOrchestrator(Config())
        prereqs = orch._get_prereqs()
        phase_d_prereqs = prereqs["phase_d"]
        assert "phase_b" not in phase_d_prereqs
        assert "phase_c" not in phase_d_prereqs


# ── OST applier detects NVENC availability ──────────────


class TestOSTApplierDetectsNVENC:
    """Verify apply_ost() calls detect_nvenc() and passes the result
    to get_encoding_args so NVENC is only used when available."""

    def test_nvenc_detected_and_passed_to_encoder(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        nvenc_calls = []
        encoding_args_calls = []

        def fake_detect_nvenc():
            nvenc_calls.append(True)
            return True

        def fake_get_encoding_args(**kwargs):
            encoding_args_calls.append(kwargs)
            return ["-c:v", "hevc_nvenc"]

        def fake_run(cmd, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response()
            else:
                result.stdout = b""
                # Follow the real cmd target: apply_ost writes to a temp
                # path and promotes it only after validation passes. A
                # trailing "-" is ffmpeg's stdout, not a file to create.
                if cmd[-1] != "-":
                    Path(cmd[-1]).touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        # Patch at the color module level since ost_applier does a local
        # `from .color import detect_nvenc, get_encoding_args` inside the
        # function body.
        with (
            patch(
                "gomoveshift.pipeline.color.detect_nvenc",
                fake_detect_nvenc,
            ),
            patch(
                "gomoveshift.pipeline.color.get_encoding_args",
                fake_get_encoding_args,
            ),
        ):
            apply_ost(
                input_path=input_path,
                output_path=output_path,
                location="County Kerry, Ireland",
                layout="regular_lf",
            )

        assert len(nvenc_calls) == 1, "detect_nvenc() must be called"
        assert len(encoding_args_calls) == 1, "get_encoding_args() must be called"
        assert encoding_args_calls[0]["has_nvenc"] is True

    def test_nvenc_unavailable_raises_no_fallback(self, tmp_path, monkeypatch):
        """detect_nvenc() raises propagate; no libx265 fallback.

        apply_ost() re-raises the NVENC error rather than producing a
        CPU-encoded output.
        """
        from gomoveshift.pipeline.color_filters import NVENCUnavailableError
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        def fake_detect_nvenc():
            raise NVENCUnavailableError("simulated probe failure")

        def fake_run(cmd, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response()
            else:
                result.stdout = b""
                # Follow the real cmd target: apply_ost writes to a temp
                # path and promotes it only after validation passes. A
                # trailing "-" is ffmpeg's stdout, not a file to create.
                if cmd[-1] != "-":
                    Path(cmd[-1]).touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        with patch(
            "gomoveshift.pipeline.color.detect_nvenc",
            fake_detect_nvenc,
        ):
            with pytest.raises(NVENCUnavailableError):
                apply_ost(
                    input_path=input_path,
                    output_path=output_path,
                    location="County Kerry, Ireland",
                    layout="regular_lf",
                    prefer_nvenc=False,
                )


# ── YouTube upload checkpoints state after each LF ──────


class TestYouTubeUploadCheckpointsState:
    """Verify that PhaseD._save_state checkpoints state and that the
    upload loop in phase_d.py calls it after each successful upload.

    We test the save_callback mechanism directly (unit) and verify
    via source inspection that the upload loop calls _save_state.
    """

    def test_save_callback_persists_state(self):
        """PhaseD._save_state invokes the save_callback with the state."""
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        save_calls = []
        config = Config()
        phase = PhaseD(config=config, save_callback=lambda s: save_calls.append(dict(s)))

        state = make_state(youtube_uploads={1: {"video_id": "abc123"}})
        phase._save_state(state)

        assert len(save_calls) == 1
        assert save_calls[0]["youtube_uploads"][1]["video_id"] == "abc123"

    def test_save_callback_none_does_not_raise(self):
        """When no save_callback is set, _save_state is a no-op."""
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        config = Config()
        phase = PhaseD(config=config, save_callback=None)
        # Should not raise
        phase._save_state({"youtube_uploads": {}})

    def test_upload_loop_calls_save_state(self):
        """Verify via source inspection that the state save is called inside
        the YouTube upload loop (after ``youtube_uploads[lf.index] = record``).

        The upload loop lives in a module-level helper in
        ``external_systems_coordinator`` that receives a ``save_state``
        callback, so the assignment + save live in the helper, not on
        ``PhaseD.run``; inspect that module to pin the assign→save ordering.
        """
        from pathlib import Path

        source = (
            Path(__file__).parent.parent
            / "gomoveshift"
            / "pipeline"
            / "drive_phases"
            / "external_systems_coordinator.py"
        ).read_text()
        assert "youtube_uploads[lf.index] = record" in source
        # Accept both shapes: ``self._save_state(state)`` (orchestrator
        # method) or ``save_state(state)`` (helper callback).
        assert "self._save_state(state)" in source or "save_state(state)" in source, (
            "phase_d must call the state-save hook in the YouTube upload loop"
        )

        # Verify the happy-path save-after-assignment ordering is preserved.
        # There is an additional save call inside the ``except YouTubeQuotaError`` branch; ``rindex``
        # anchors on the last (post-assignment) call regardless of shape.
        assign_pos = source.index("youtube_uploads[lf.index] = record")
        if "save_state(state)" in source[assign_pos:]:
            save_pos = source.rindex("save_state(state)")
        else:
            save_pos = source.rindex("self._save_state(state)")
        assert save_pos > assign_pos, "state save must be called AFTER recording the upload result"


# ── Silent OST failure warning in day_pipeline ──────────


class TestSilentOstFailureWarning:
    """Verify day_pipeline logs a warning when apply_ost returns None for shorts."""

    def test_ost_failure_warning_in_source(self):
        """day_pipeline.py must contain the short-render failure warning message.

        Originally the day-first shorts path was a two-step
        ``extract_slice`` + ``apply_ost(layout="short")`` pipeline whose OST
        step could return None and silently drop a short. That path was
        replaced with a single
        ``render_short_ffmpeg`` invocation that center-crops to 9:16,
        scales, and composites the short OST in one ffmpeg pass — so the
        old "OST application failed for short rank" warning string no
        longer exists. The equivalent safety net is the
        ``render_short_ffmpeg failed for short rank`` warning logged when
        the renderer returns False; that's the surface this test guards.
        """
        source_path = (
            Path(__file__).resolve().parent.parent / "gomoveshift" / "pipeline" / "day_pipeline.py"
        )
        source = source_path.read_text()
        assert "render_short_ffmpeg failed for short rank" in source


# ── Drive uploader explicit chunksize ───────────────────


class TestDriveUploaderChunksize:
    """Verify Drive uploader uses explicit 256MB chunksize, not default -1."""

    def test_upload_file_uses_explicit_chunksize(self):
        """_upload_file must create MediaFileUpload with chunksize=256*1024*1024."""
        source_path = (
            Path(__file__).resolve().parent.parent / "gomoveshift" / "pipeline" / "uploader.py"
        )
        source = source_path.read_text()
        # Find the _upload_file method and verify chunksize is set
        assert "chunksize=256 * 1024 * 1024" in source
        # Must NOT contain chunksize=-1
        assert "chunksize=-1" not in source


# ── Drive uploader per-chunk timeout ────────────────────


class TestDriveUploaderChunkTimeout:
    """Verify Drive uploader has per-chunk timeout protection."""

    def test_next_chunk_with_timeout_method_exists(self):
        """PipelineUploader must have _next_chunk_with_timeout."""
        from gomoveshift.pipeline.uploader import PipelineUploader

        assert hasattr(PipelineUploader, "_next_chunk_with_timeout")

    def test_next_chunk_with_timeout_enforces_deadline(self):
        """_next_chunk_with_timeout raises on a stuck next_chunk call."""
        from gomoveshift.pipeline.uploader import PipelineUploader

        uploader = PipelineUploader.__new__(PipelineUploader)
        mock_request = MagicMock()

        def slow_chunk():
            time.sleep(10)
            return None, {"id": "test"}

        mock_request.next_chunk = slow_chunk

        with pytest.raises(Exception):
            uploader._next_chunk_with_timeout(mock_request, timeout=0.1)

    def test_upload_file_calls_next_chunk_with_timeout(self):
        """_upload_file must use _next_chunk_with_timeout, not raw next_chunk."""
        source_path = (
            Path(__file__).resolve().parent.parent / "gomoveshift" / "pipeline" / "uploader.py"
        )
        source = source_path.read_text()
        # Inside _upload_file, the loop should call self._next_chunk_with_timeout
        assert "self._next_chunk_with_timeout(request)" in source
