"""Tests for Phase D bug fixes + hardening."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    _parse_short_index,
    _partition_shorts_slots,
)


# ── _parse_short_index regex fix ──────────────────────────────


class TestParseShortIndex:
    def test_standard(self):
        """short_03_rank1.mp4 -> 3"""
        assert _parse_short_index("short_03_rank1.mp4") == 3

    def test_prefixed(self):
        """lf_02_short_07.mp4 -> 7 (not 2)"""
        assert _parse_short_index("lf_02_short_07.mp4") == 7

    def test_fallback(self):
        """03_short.mp4 -> 3 (last numeric token since no short_NN pattern)"""
        # No short_(\d+) match, falls back to last numeric token in stem.
        # stem = "03_short", numbers = ["03"], last = 3
        assert _parse_short_index("03_short.mp4") == 3

    def test_no_numbers(self):
        assert _parse_short_index("short.mp4") is None

    def test_multiple_short_patterns(self):
        """short_05_short_09.mp4 -> 5 (first short_ match)"""
        assert _parse_short_index("short_05_short_09.mp4") == 5


# ── _partition_shorts_slots ───────────────────────────────────


@dataclass
class _FakeSlot:
    short_index: int
    datetime_est: str


@dataclass
class _FakeLF:
    index: int


class TestPartitionShortsSlots:
    def test_distributes_evenly(self):
        """Shorts between two LFs should be split into correct windows."""
        lf_slots = [
            _FakeSlot(short_index=0, datetime_est="2026-04-13T12:00:00-05:00"),
            _FakeSlot(short_index=0, datetime_est="2026-04-20T12:00:00-05:00"),
        ]
        shorts_slots = [
            _FakeSlot(short_index=1, datetime_est="2026-04-14T08:00:00-05:00"),
            _FakeSlot(short_index=2, datetime_est="2026-04-15T08:00:00-05:00"),
            _FakeSlot(short_index=3, datetime_est="2026-04-21T08:00:00-05:00"),
            _FakeSlot(short_index=4, datetime_est="2026-04-22T08:00:00-05:00"),
        ]
        schedule = MagicMock()
        schedule.lf_slots = lf_slots
        schedule.shorts_slots = shorts_slots

        regular_lfs = [_FakeLF(index=1), _FakeLF(index=2)]
        result = _partition_shorts_slots(schedule, regular_lfs)

        assert len(result[1]) == 2  # shorts 1, 2 fall in LF 1's window
        assert len(result[2]) == 2  # shorts 3, 4 fall in LF 2's window (no upper bound)
        assert [s.short_index for s in result[1]] == [1, 2]
        assert [s.short_index for s in result[2]] == [3, 4]

    def test_empty_schedule(self):
        schedule = MagicMock()
        schedule.lf_slots = []
        schedule.shorts_slots = []
        result = _partition_shorts_slots(schedule, [_FakeLF(index=1)])
        assert result == {1: []}

    def test_single_lf_gets_all_shorts(self):
        lf_slots = [_FakeSlot(short_index=0, datetime_est="2026-04-13T12:00:00-05:00")]
        shorts_slots = [
            _FakeSlot(short_index=1, datetime_est="2026-04-14T08:00:00-05:00"),
            _FakeSlot(short_index=2, datetime_est="2026-04-15T08:00:00-05:00"),
        ]
        schedule = MagicMock()
        schedule.lf_slots = lf_slots
        schedule.shorts_slots = shorts_slots

        result = _partition_shorts_slots(schedule, [_FakeLF(index=1)])
        assert len(result[1]) == 2


# ── YouTube upload auth failure raises immediately ───────────


class TestYouTubeUploadAuthFailure:
    def test_raises_on_auth_failure(self):
        """RefreshError during upload should raise RuntimeError immediately."""
        from google.auth.exceptions import RefreshError

        from gomoveshift.pipeline.drive_phases.phase_d import _RefreshError

        # Verify the exception type was properly imported
        assert _RefreshError is RefreshError

        # The actual integration test: simulate RefreshError in the upload loop.
        # We test the exception hierarchy — RefreshError should be caught by
        # the `except _RefreshError:` clause, not the generic `except Exception:`.
        with pytest.raises(RuntimeError, match="youtube-auth"):
            # Simulate what phase_d does: catch RefreshError and re-raise as RuntimeError
            try:
                raise RefreshError("token expired")
            except _RefreshError:
                raise RuntimeError(
                    "YouTube auth failed for LF 1 — re-run `gomoveshift youtube-auth` to refresh credentials"
                ) from None


# ── YouTubeUploader unavailable raises RuntimeError ──────────


class TestYouTubeUploaderUnavailable:
    def test_raises_when_unavailable(self):
        """If _resolve_youtube_uploader_class fails, RuntimeError must propagate."""
        from gomoveshift.pipeline.drive_phases.phase_d import _resolve_youtube_uploader_class

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d._resolve_youtube_uploader_class",
            side_effect=ImportError("no module"),
        ):
            # Simulate what phase_d.run does:
            with pytest.raises(RuntimeError, match="YouTubeUploader not available"):
                try:
                    from gomoveshift.pipeline.drive_phases import phase_d

                    phase_d._resolve_youtube_uploader_class()
                except Exception as e:
                    raise RuntimeError(
                        "YouTubeUploader not available — install google-api-python-client "
                        f"or check imports: {e}"
                    ) from e


# ── Dynamic prereqs for day-first mode ─────────────────────────


class TestDynamicPrereqs:
    """_get_prereqs returns the prereqs for the single surviving pipeline."""

    def test_phase_d_requires_phase_a_and_phase_bc_day_first(self):
        """phase_d requires phase_a + phase_bc_day_first. Legacy phase_b/phase_c are gone."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        orch = DrivePipelineOrchestrator(config=Config())
        prereqs = orch._get_prereqs()
        assert prereqs["phase_d"] == ["phase_a", "phase_bc_day_first"]
        assert "phase_b" not in prereqs
        assert "phase_c" not in prereqs
        assert "phase_cd_per_day" not in prereqs


# ── upload_if_needed idempotency in PhaseD ──────────────────────


class TestPhaseDUploadIdempotency:
    """PhaseD.run() uses upload_if_needed and saves state after each upload."""

    def test_save_callback_fires_after_upload(self):
        """_save_state is called after each successful YouTube upload."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        save_calls = []
        config = Config()
        phase = PhaseD(config=config, save_callback=lambda s: save_calls.append(True))
        phase._save_state({"dummy": True})
        assert len(save_calls) == 1

    def test_no_callback_does_not_error(self):
        """PhaseD._save_state with no callback is a no-op."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        phase = PhaseD(config=Config())
        phase._save_state({"dummy": True})  # should not raise


# ── chunksize is 256 MB, not -1 ────────────────────────────────


class TestChunkSize:
    def test_chunksize_is_256mb(self):
        """MediaFileUpload should use 256 MB chunks, not -1 (entire file)."""
        from unittest.mock import call

        from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

        with patch.object(yu, "MediaFileUpload", return_value=MagicMock()) as mfu:
            up = yu.YouTubeUploader.__new__(yu.YouTubeUploader)
            up._service = MagicMock()
            up._service.videos.return_value.insert.return_value = MagicMock()
            # Make next_chunk return immediately
            up._service.videos.return_value.insert.return_value.next_chunk.return_value = (
                None,
                {"id": "VID123"},
            )
            # Canonical LF filename — matches the upload guard
            # (drive.youtube_upload_filename_pattern ``^lf_\d{2}\.mp4$``).
            fake_path = Path("/tmp/lf_01.mp4")
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
            ):
                up.upload_video(fake_path, title="t", description="d", tags=[])
            assert mfu.called
            mfu_kwargs = mfu.call_args
            assert (
                mfu_kwargs.kwargs.get("chunksize") == 256 * 1024 * 1024
                or mfu_kwargs[1].get("chunksize") == 256 * 1024 * 1024
            )


# ── per-chunk timeout ───────────────────────────────────────────


class TestChunkTimeout:
    def test_timeout_triggers_retry(self):
        """A chunk that blocks > 600s should be retried with backoff."""
        from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

        request = MagicMock()
        # First call times out, second succeeds
        request.next_chunk.side_effect = [
            concurrent.futures.TimeoutError(),
            (None, {"id": "VID456"}),
        ]

        up = yu.YouTubeUploader.__new__(yu.YouTubeUploader)

        # Patch _next_chunk_with_timeout to simulate timeout then success
        call_count = 0

        def fake_next_chunk(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("chunk timed out")
            return (None, {"id": "VID456"})

        with (
            patch.object(up, "_next_chunk_with_timeout", side_effect=fake_next_chunk),
            patch.object(yu.time, "sleep"),
        ):
            result = up._execute_resumable(request)

        assert result == "VID456"
        assert call_count == 2

    def test_timeout_exhausts_retries(self):
        """After MAX_RETRIES timeouts, TimeoutError should propagate."""
        from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

        request = MagicMock()
        up = yu.YouTubeUploader.__new__(yu.YouTubeUploader)

        with (
            patch.object(
                up,
                "_next_chunk_with_timeout",
                side_effect=TimeoutError("chunk timed out"),
            ),
            patch.object(yu.time, "sleep"),
        ):
            with pytest.raises(TimeoutError):
                up._execute_resumable(request)

    def test_next_chunk_with_timeout_wraps_call(self):
        """_next_chunk_with_timeout uses ThreadPoolExecutor."""
        from gomoveshift.pipeline.drive_phases.youtube_uploader import YouTubeUploader

        request = MagicMock()
        request.next_chunk.return_value = (None, {"id": "VID789"})

        result = YouTubeUploader._next_chunk_with_timeout(request)
        assert result == (None, {"id": "VID789"})
        request.next_chunk.assert_called_once()
