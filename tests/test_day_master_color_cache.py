"""Tests for day master color analysis caching.

When encode_day_master creates a raw concat file, SinglePassProcessor's
_analyze_color runs expensive frame sampling (~35 min for a day's footage).
On pipeline restart, the raw concat is recreated (new mtime), invalidating
the mtime-based color cache that _analyze_color maintains.

The fix: encode_day_master maintains a stable color cache keyed on source
episode fingerprints (size + mtime). Before calling processor.process(),
if the stable cache is valid, it pre-seeds the color_analysis.json file
that _analyze_color expects alongside the raw concat. This way _analyze_color
finds a fresh cache and skips re-analysis.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.cache import TRANSIENT_COLOR_CACHE_SCHEMA_VERSION
from gomoveshift.pipeline.day_master_encoder import COLOR_CACHE_SCHEMA_VERSION


@pytest.fixture(autouse=True)
def _blur_off(monkeypatch):
    """Color-cache MECHANICS, orthogonal to the privacy blur now default-ON in
    ``encode_day_master`` (its coverage lives in ``test_day_master_blur.py``). Force
    blur off so the mocked encode isn't gated by the day-master mask build / marker."""
    monkeypatch.setattr("gomoveshift.pipeline.blur_gate.blur_required", lambda cfg: False)


def _make_episode(index: int, concat_path: Path, duration_min: float = 30.0):
    from datetime import datetime, timezone

    ep = MagicMock()
    ep.index = index
    ep.creation_time = datetime(2026, 2, 20, 10 + index, 0, tzinfo=timezone.utc)
    ep.total_duration = duration_min * 60
    ep.total_duration_min = duration_min
    ep.concat_path = concat_path
    ep.gpx_path = None
    return ep


def _make_day_batch(date: str, episode_paths: list):
    from gomoveshift.pipeline.day_grouper import DayBatch
    from datetime import datetime

    episodes = []
    for i, p in enumerate(episode_paths, start=1):
        episodes.append(_make_episode(i, p, duration_min=30.0))
    return DayBatch(date=date, episodes=episodes)


def _patch_subprocess_and_singlepass(monkeypatch, sp_calls=None):
    """Common patches: subprocess.run succeeds, SinglePassProcessor records calls.

    The fake_process also writes a transient color_analysis.json alongside
    the input video, simulating what SinglePassProcessor._analyze_color does
    during a real encode. This is necessary for the stable cache save logic
    to find the transient data.
    """
    monkeypatch.setattr(
        "gomoveshift.pipeline.day_master_encoder.subprocess.run",
        lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
    )

    if sp_calls is None:
        sp_calls = []

    def fake_process(self, video_path, output_path, location, **kwargs):
        sp_calls.append(
            {
                "input": video_path,
                "output": output_path,
                "location": location,
                "kwargs": kwargs,
            }
        )
        # Simulate _analyze_color writing the transient color cache: it stores
        # merged-span ``corrections`` + ``span_durations`` rather than a
        # derived ``filter_str``.
        transient = video_path.parent / f"{video_path.stem}_color_analysis.json"
        if not transient.exists():
            transient.parent.mkdir(parents=True, exist_ok=True)
            transient.write_text(
                json.dumps(
                    {
                        "version": TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
                        "corrections": [
                            {
                                "shadow_lift": 5.0,
                                "highlight_pull": 5.0,
                                "saturation_mult": 1.05,
                                "contrast_curve": 0.0,
                                "white_balance_r": 0.0,
                                "white_balance_b": 0.0,
                                "gamma": 1.0,
                                "denoise_strength": 0.0,
                                "sharpen_amount": 0.0,
                            }
                        ],
                        "method": "auto_global",
                        "span_durations": None,
                    }
                )
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return MagicMock(success=True, output_path=output_path)

    monkeypatch.setattr(
        "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
        fake_process,
    )
    return sp_calls


class TestDayMasterColorCacheWrite:
    """Color stats are cached to disk after first analysis."""

    def test_stable_cache_written_after_encode(self, tmp_path, monkeypatch):
        """After encode_day_master succeeds, a stable color cache JSON
        exists in the output directory keyed on the day date."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-02-20", [ep])

        _patch_subprocess_and_singlepass(monkeypatch)

        output_dir = tmp_path / "days_out"
        encode_day_master(day_batch=batch, output_dir=output_dir, config=Config())

        cache_path = output_dir / "day_2026-02-20_color_cache.json"
        assert cache_path.exists(), "Stable color cache was not written"

        data = json.loads(cache_path.read_text())
        assert "source_fingerprints" in data
        assert data.get("version") == COLOR_CACHE_SCHEMA_VERSION
        # The stable cache stamps the grade-algorithm fingerprint so a grade
        # change auto-invalidates the cache.
        assert data.get("grade_algo_version")
        assert "corrections" in data
        assert "method" in data

    def test_stable_cache_contains_source_fingerprints(self, tmp_path, monkeypatch):
        """The stable cache includes fingerprints of all source episode files
        so we can detect when sources change."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        ep1.write_bytes(b"x" * 1024)
        ep2.write_bytes(b"y" * 2048)
        batch = _make_day_batch("2026-02-20", [ep1, ep2])

        _patch_subprocess_and_singlepass(monkeypatch)

        output_dir = tmp_path / "days_out"
        encode_day_master(day_batch=batch, output_dir=output_dir, config=Config())

        cache_path = output_dir / "day_2026-02-20_color_cache.json"
        data = json.loads(cache_path.read_text())
        fps = data["source_fingerprints"]
        assert len(fps) == 2
        # Each fingerprint is "size:mtime_ns"
        for fp in fps:
            assert ":" in fp


class TestDayMasterColorCacheLoad:
    """Cached stats loaded on second call (no re-analysis)."""

    def test_second_encode_preseeds_color_cache(self, tmp_path, monkeypatch):
        """On a second run (pipeline restart), encode_day_master should
        pre-seed the transient color_analysis.json so _analyze_color
        finds a cache hit. We verify by checking the cache file is
        placed alongside the raw concat path before process() is called."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-02-20", [ep])
        output_dir = tmp_path / "days_out"

        # First run: creates the stable cache
        sp_calls_1 = _patch_subprocess_and_singlepass(monkeypatch)
        encode_day_master(day_batch=batch, output_dir=output_dir, config=Config())
        assert len(sp_calls_1) == 1

        # Verify the stable cache was written
        stable_cache = output_dir / "day_2026-02-20_color_cache.json"
        assert stable_cache.exists()

        # Patch out the resume-skip check so we actually re-run the encode path
        # (in real pipeline, master file would be missing/incomplete on restart)
        master = output_dir / "day_2026-02-20_master.mp4"
        if master.exists():
            master.unlink()

        # Second run: should pre-seed the transient cache
        preseeded_paths = []
        original_process = None

        def tracking_process(self, video_path, output_path, location, **kwargs):
            # Check if the transient color cache was pre-seeded
            transient_cache = video_path.parent / f"{video_path.stem}_color_analysis.json"
            if transient_cache.exists():
                preseeded_paths.append(transient_cache)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            tracking_process,
        )

        encode_day_master(day_batch=batch, output_dir=output_dir, config=Config())
        assert len(preseeded_paths) == 1, (
            "Transient color cache was not pre-seeded from stable cache on restart"
        )
        # Existence alone masks the bug: the preseed must stamp the SAME
        # schema version the _analyze_color reader accepts, or resume re-runs the
        # 35–90 min analysis anyway. Assert reader/preseed version parity.
        from gomoveshift.pipeline.cache import TRANSIENT_COLOR_CACHE_SCHEMA_VERSION

        preseeded = json.loads(preseeded_paths[0].read_text())
        assert preseeded["version"] == TRANSIENT_COLOR_CACHE_SCHEMA_VERSION


class TestPreseedReaderParity:
    """The real preseed output must be honored by the _analyze_color reader."""

    def test_preseed_transient_cache_accepted_by_analyze_color(self, tmp_path):
        """_preseed_transient_color_cache must write the SAME schema version
        SinglePassProcessor._analyze_color accepts, so a resume gets a cache
        hit and the 35–90 min 4K color analysis is skipped. A version drift
        silently forces the re-analysis on every restart."""
        from gomoveshift.pipeline.day_master_encoder import _preseed_transient_color_cache
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        raw_concat = tmp_path / "day_2026-02-20_raw.mp4"
        raw_concat.write_bytes(b"x" * 4096)

        stable_color = {
            "corrections": [
                {
                    "shadow_lift": 5.0,
                    "highlight_pull": 5.0,
                    "saturation_mult": 1.05,
                    "contrast_curve": 0.0,
                    "white_balance_r": 0.0,
                    "white_balance_b": 0.0,
                    "gamma": 1.0,
                    "denoise_strength": 0.0,
                    "sharpen_amount": 0.0,
                }
            ],
            "method": "auto_global",
            "span_durations": None,
        }

        # Run the REAL preseed (not a hand-seeded cache file).
        _preseed_transient_color_cache(raw_concat, stable_color)

        # If _analyze_color reaches _get_video_info, the preseeded cache was
        # rejected and it fell through to the expensive re-analysis — fail loudly.
        # It re-derives the filter from the cached corrections via
        # _corrections_to_filter, so the stub borrows the real method.
        class _NoReanalysis:
            _corrections_to_filter = SinglePassProcessor._corrections_to_filter
            # The reader compares the cached sharpen/denoise flags against the
            # run's --no-sharpen/--no-denoise; the preseed writer records the
            # defaults, so the parity stub carries the same defaults.
            config = SimpleNamespace(drive=SimpleNamespace(no_sharpen=False, no_denoise=False))

            def _get_video_info(self, *a, **k):
                raise AssertionError(
                    "preseeded color cache REJECTED — _analyze_color fell "
                    "through to re-analysis (reader/preseed schema drift)"
                )

        filter_str, method, corrections, lut_dir = SinglePassProcessor._analyze_color(
            _NoReanalysis(), raw_concat
        )
        assert method == "auto_global"
        assert len(corrections) == 1


class TestDayMasterColorCacheInvalidation:
    """Cache invalidation when source files change."""

    def test_cache_invalidated_when_source_changes(self, tmp_path, monkeypatch):
        """If a source episode file changes (different size/mtime), the
        stable cache must NOT be used — fingerprints won't match."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-02-20", [ep])
        output_dir = tmp_path / "days_out"

        # First run
        _patch_subprocess_and_singlepass(monkeypatch)
        encode_day_master(day_batch=batch, output_dir=output_dir, config=Config())

        stable_cache = output_dir / "day_2026-02-20_color_cache.json"
        assert stable_cache.exists()
        original_data = json.loads(stable_cache.read_text())

        # Remove master so we go through the encode path again
        master = output_dir / "day_2026-02-20_master.mp4"
        if master.exists():
            master.unlink()

        # Clean up any transient color cache from the first run so we
        # can detect whether the second run pre-seeds a fresh one
        for f in output_dir.glob("*_color_analysis.json"):
            f.unlink()

        # Modify source file (simulates re-ingest with different content)
        time.sleep(0.05)  # Ensure mtime differs
        ep.write_bytes(b"y" * 2048)  # Different size AND mtime

        # Rebuild batch with the changed file
        batch2 = _make_day_batch("2026-02-20", [ep])

        # Track whether the transient cache was pre-seeded
        preseeded = []

        def tracking_process(self, video_path, output_path, location, **kwargs):
            transient_cache = video_path.parent / f"{video_path.stem}_color_analysis.json"
            if transient_cache.exists():
                preseeded.append(True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            tracking_process,
        )

        encode_day_master(day_batch=batch2, output_dir=output_dir, config=Config())

        # The transient cache should NOT have been pre-seeded because
        # the source fingerprints changed
        assert len(preseeded) == 0, "Stale color cache was pre-seeded despite source file change"

    def test_cache_invalidated_when_episode_count_changes(self, tmp_path, monkeypatch):
        """Adding or removing episodes from a day batch invalidates the cache."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep1.write_bytes(b"x" * 1024)
        batch1 = _make_day_batch("2026-02-20", [ep1])
        output_dir = tmp_path / "days_out"

        # First run with 1 episode
        _patch_subprocess_and_singlepass(monkeypatch)
        encode_day_master(day_batch=batch1, output_dir=output_dir, config=Config())

        stable_cache = output_dir / "day_2026-02-20_color_cache.json"
        data1 = json.loads(stable_cache.read_text())
        assert len(data1["source_fingerprints"]) == 1

        # Remove master and transient caches for second run
        master = output_dir / "day_2026-02-20_master.mp4"
        if master.exists():
            master.unlink()
        for f in output_dir.glob("*_color_analysis.json"):
            f.unlink()

        # Second run with 2 episodes
        ep2 = tmp_path / "ep_002.mp4"
        ep2.write_bytes(b"y" * 2048)
        batch2 = _make_day_batch("2026-02-20", [ep1, ep2])

        preseeded = []

        def tracking_process(self, video_path, output_path, location, **kwargs):
            transient_cache = video_path.parent / f"{video_path.stem}_color_analysis.json"
            if transient_cache.exists():
                preseeded.append(True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            tracking_process,
        )

        encode_day_master(day_batch=batch2, output_dir=output_dir, config=Config())

        assert len(preseeded) == 0, "Color cache should be invalidated when episode count changes"
