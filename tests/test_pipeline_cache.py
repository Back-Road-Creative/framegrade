"""
Tests for pipeline cache — ensure expensive operations are skipped when outputs are fresh.

mtime ordering is forced via ``os.utime`` rather than relying on ``time.sleep``
because the dev workstation (WSL2) clock can regress backwards by several
seconds under load; a ``sleep(0.05)`` pattern produced an output_mtime
< input_mtime in about 1% of runs, flaking the cache tests.
"""

import os
import time

import numpy as np


# Offset (in nanoseconds) used to space input/output mtimes deterministically.
# 100ms is well above ext4/tmpfs filesystem resolution and large enough that
# bit-by-bit comparison logic in ``is_output_fresh`` (which uses ``>``, not ``>=``)
# never sees a tie. Direction is set explicitly per test via ``_force_input_older``
# / ``_force_input_newer`` so the assertion never depends on wall-clock direction.
_MTIME_GAP_NS = 100_000_000


def _force_input_older(input_file, output_file):
    """Stamp input.mtime = output.mtime - 100ms so input is unambiguously older."""
    out_ns = output_file.stat().st_mtime_ns
    os.utime(input_file, ns=(out_ns - _MTIME_GAP_NS, out_ns - _MTIME_GAP_NS))


def _force_input_newer(input_file, output_file):
    """Stamp input.mtime = output.mtime + 100ms so input is unambiguously newer."""
    out_ns = output_file.stat().st_mtime_ns
    os.utime(input_file, ns=(out_ns + _MTIME_GAP_NS, out_ns + _MTIME_GAP_NS))


class TestCacheUtilities:
    """Test core cache functions."""

    def test_output_fresh_when_newer(self, tmp_path):
        """Output newer than input should be considered fresh."""
        from gomoveshift.pipeline.cache import is_output_fresh

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"x" * 2048)
        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"y" * 2048)
        _force_input_older(input_file, output_file)

        assert is_output_fresh(input_file, output_file) is True

    def test_output_stale_when_older(self, tmp_path):
        """Output older than input should not be considered fresh."""
        from gomoveshift.pipeline.cache import is_output_fresh

        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"y" * 2048)
        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"x" * 2048)
        _force_input_newer(input_file, output_file)

        assert is_output_fresh(input_file, output_file) is False

    def test_output_missing_not_fresh(self, tmp_path):
        """Missing output should not be considered fresh."""
        from gomoveshift.pipeline.cache import is_output_fresh

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"x" * 2048)

        assert is_output_fresh(input_file, tmp_path / "nonexistent.mp4") is False

    def test_output_too_small_not_fresh(self, tmp_path):
        """Output below min_size_bytes should not be considered fresh."""
        from gomoveshift.pipeline.cache import is_output_fresh

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"x" * 2048)

        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"y" * 100)  # Too small

        assert is_output_fresh(input_file, output_file, min_size_bytes=1024) is False

    def test_are_outputs_fresh_all_must_exist(self, tmp_path):
        """All outputs must be fresh for are_outputs_fresh to return True."""
        from gomoveshift.pipeline.cache import are_outputs_fresh

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"x" * 2048)
        out1 = tmp_path / "out1.mp4"
        out1.write_bytes(b"y" * 2048)
        out2 = tmp_path / "out2.mp4"
        out2.write_bytes(b"y" * 2048)
        # Force input older than the EARLIER of the two outputs so both look fresh.
        oldest_out_ns = min(out1.stat().st_mtime_ns, out2.stat().st_mtime_ns)
        os.utime(
            input_file,
            ns=(oldest_out_ns - _MTIME_GAP_NS, oldest_out_ns - _MTIME_GAP_NS),
        )

        assert are_outputs_fresh(input_file, [out1, out2]) is True

        # One missing → False
        assert are_outputs_fresh(input_file, [out1, tmp_path / "missing.mp4"]) is False

    def test_analysis_cache_roundtrip(self, tmp_path):
        """Should save and load analysis cache JSON."""
        from gomoveshift.pipeline.cache import load_cached_json, save_cache_json

        data = {"segments": [{"black_point": 15}], "is_adaptive": False}
        cache_path = tmp_path / "cache.json"
        save_cache_json(cache_path, data)

        loaded = load_cached_json(cache_path)
        assert loaded["segments"][0]["black_point"] == 15
        assert loaded["is_adaptive"] is False

    def test_file_fingerprint_changes_with_content(self, tmp_path):
        """Fingerprint should change when file content changes."""
        from gomoveshift.pipeline.cache import file_fingerprint

        f = tmp_path / "test.mp4"
        f.write_bytes(b"original")
        fp1 = file_fingerprint(f)

        time.sleep(0.05)
        f.write_bytes(b"modified content that is different")
        fp2 = file_fingerprint(f)

        assert fp1 != fp2


class TestStopProcessorCache:
    """Test that StopProcessor skips when output is fresh."""

    def test_skips_when_output_fresh(self, tmp_path):
        """A fresh output WITH its provenance sidecar is reused as-is — the
        real cache-skip path (no re-probe of the input)."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.stop_processor import StopProcessor

        config = Config()
        proc = StopProcessor(config)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"x" * 2048)
        output_file = tmp_path / "no_stops.mp4"
        output_file.write_bytes(b"y" * (2 * 1024 * 1024))  # 2MB
        _force_input_older(input_file, output_file)
        # Every completed run persists this sidecar beside the cut;
        # without it a fresh output is re-derived (which probes the input).
        proc._save_stops_result(
            output_file,
            {
                "stops_found": 0,
                "stops": [],
                "total_removed_duration": 0.0,
                "output_path": str(output_file),
            },
        )

        result = proc.process(input_file, output_file)

        assert result.get("cached") is True
        assert result["output_path"] == str(output_file)

    def test_fresh_output_without_sidecar_is_rederived(self, tmp_path):
        """A fresh output lacking the sidecar must NOT be trusted blindly:
        process re-derives so a resumed run still switches to the cut file and
        runs STOP-VERIFY. detect_stops is stubbed to avoid probing the fixture."""
        from unittest.mock import patch

        from gomoveshift.config import Config
        from gomoveshift.pipeline.stop_processor import StopProcessor

        config = Config()
        proc = StopProcessor(config)

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"x" * 2048)
        output_file = tmp_path / "no_stops.mp4"
        output_file.write_bytes(b"y" * (2 * 1024 * 1024))  # 2MB
        _force_input_older(input_file, output_file)

        with patch.object(StopProcessor, "detect_stops", return_value=[]) as mock_detect:
            result = proc.process(input_file, output_file)

        mock_detect.assert_called_once()  # re-derived, not blindly skipped
        assert result["stops_found"] == 0


# ``TestAudioProcessorCache`` was removed with its subject, the dead
# ``AudioProcessor`` (since deleted). The live
# freshness-skip is ``cache.is_output_fresh``, covered by ``TestCacheUtilities``
# above and the shorts/stop caches below.


class TestShortsRenderCache:
    """Test that shorts renderer skips already-rendered shorts."""

    def test_skips_existing_short(self, tmp_path):
        """render_short_ffmpeg should be skipped for existing output files."""
        from gomoveshift.pipeline.cache import is_output_fresh

        source = tmp_path / "source.mp4"
        source.write_bytes(b"x" * 2048)
        short = tmp_path / "short_01.mp4"
        short.write_bytes(b"y" * (200 * 1024))  # 200KB
        _force_input_older(source, short)

        assert is_output_fresh(source, short, min_size_bytes=100 * 1024) is True

    def test_does_not_skip_missing_short(self, tmp_path):
        """Missing short should not be skipped."""
        from gomoveshift.pipeline.cache import is_output_fresh

        source = tmp_path / "source.mp4"
        source.write_bytes(b"x" * 2048)

        assert is_output_fresh(source, tmp_path / "missing.mp4") is False
