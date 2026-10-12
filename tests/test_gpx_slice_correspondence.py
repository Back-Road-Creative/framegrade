"""Footage↔GPS anchoring: slice-tuned correspondence gate + coverage window.

A GPMF sub-slice's GPX is time-windowed out of the parent's own merged GPMF, so
it can never be a *swapped* track — but a gross anchoring/slicing error can
still hand a slice a window that doesn't match its video. The slice-tuned gate
catches that (start-time + one-sided over-span; no embedded-coord) while treating
under-coverage (lock-delay prefix / dropout) as expected. ``gps_coverage_window``
is read-only metadata exposing an unlocked GPS prefix in video time.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.pipeline.gopro_gps import (
    GpxCorrespondenceError,
    assert_slice_gpx_correspondence,
    gps_coverage_window,
)
from gomoveshift.pipeline.ingest.concatenator import (
    _split_episode_at_points,
    extract_and_merge_gpx,
)
from gomoveshift.pipeline.ingest.gopro_parser import Episode, GoProChapter, GoProRecording

_CREATION = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
_T0 = _CREATION.timestamp()

# The adoption gate now parses the pre-set GPX and requires ≥1
# trackpoint (bare `<gpx/>` no longer passes — that's what a crash-truncated
# write looks like), so fixtures standing in for a legitimate pre-set GPX
# need a real trackpoint.
_VALID_GPX = (
    '<?xml version="1.0"?><gpx version="1.1" creator="test">'
    '<trk><trkseg><trkpt lat="45.0" lon="-63.0">'
    "<time>2026-05-01T12:00:00Z</time></trkpt></trkseg></trk></gpx>"
)


def _gps(start_unix: float, span_sec: float, lat: float = 32.42, lon: float = -80.46):
    """Two-point GPSData spanning ``span_sec`` from ``start_unix``."""
    return GPSData(
        trackpoints=[
            GPSTrackpoint(lat, lon, None, start_unix),
            GPSTrackpoint(lat + 0.01, lon + 0.01, None, start_unix + span_sec),
        ]
    )


# --- Slice gate: well-corresponded / expected-partial slices PASS ---


def test_well_corresponded_slice_passes():
    assert_slice_gpx_correspondence(_gps(_T0, 600), _CREATION, video_duration=600.0)


def test_lock_delay_prefix_under_coverage_passes():
    # 34-min GPS-lock delay + under-coverage of the 3600s video: start delta
    # < 1h tolerance, and a shorter GPX is legitimate partial coverage.
    assert_slice_gpx_correspondence(_gps(_T0 + 34 * 60, 1560), _CREATION, video_duration=3600.0)
    # Aligned start, GPX spans only half the video (mid-clip dropout) — no raise.
    assert_slice_gpx_correspondence(_gps(_T0, 300), _CREATION, video_duration=600.0)


# --- Slice gate: a genuinely MIS-corresponded slice is caught ---


def test_mis_corresponded_wrong_start_time_raises():
    with pytest.raises(GpxCorrespondenceError, match="start-time"):
        assert_slice_gpx_correspondence(_gps(_T0 + 2 * 3600, 600), _CREATION, video_duration=600.0)


def test_mis_corresponded_over_span_raises():
    # GPX spans 30 min for a 10 min slice — window grabbed points beyond extent.
    with pytest.raises(GpxCorrespondenceError, match="over-covers"):
        assert_slice_gpx_correspondence(_gps(_T0, 1800), _CREATION, video_duration=600.0)


def test_slice_gate_degrades_on_missing_data():
    assert_slice_gpx_correspondence(GPSData(trackpoints=[]), _CREATION, video_duration=600.0)
    assert_slice_gpx_correspondence(_gps(_T0, 600), None, video_duration=600.0)
    assert_slice_gpx_correspondence(None, _CREATION, video_duration=600.0)


# --- gps_coverage_window: read-only metadata (in video time) ---


def test_coverage_window_healthy_and_prefix():
    first, last = gps_coverage_window(_gps(_T0, 600), _CREATION)
    assert abs(first) < 0.01 and abs(last - 600.0) < 0.01
    # 34-min unlocked prefix surfaces as first_fix_offset ≈ 2040s.
    first, _ = gps_coverage_window(_gps(_T0 + 34 * 60, 1560), _CREATION)
    assert abs(first - 34 * 60) < 0.01


def test_coverage_window_skips_null_island_prefix():
    # Leading null-island (0,0) fixes are skipped — window opens at first real fix.
    gps = GPSData(
        trackpoints=[
            GPSTrackpoint(0.0, 0.0, None, _T0),
            GPSTrackpoint(0.0, 0.0, None, _T0 + 30),
            GPSTrackpoint(32.42, -80.46, None, _T0 + 300),
            GPSTrackpoint(32.43, -80.45, None, _T0 + 900),
        ]
    )
    first, last = gps_coverage_window(gps, _CREATION)
    assert abs(first - 300.0) < 0.01 and abs(last - 900.0) < 0.01


def test_coverage_window_degrades_to_none():
    assert gps_coverage_window(None, _CREATION) is None
    assert gps_coverage_window(_gps(_T0, 600), None) is None
    assert gps_coverage_window(GPSData(trackpoints=[]), _CREATION) is None


# --- Provenance: a sub-slice is tagged ``gpmf_slice`` (both ingest paths) ---


def _ffmpeg_success(cmd, **kwargs):
    Path(cmd[-1]).write_bytes(b"\x00" * 512)
    return MagicMock(returncode=0, stderr="")


def test_splitter_tags_sub_slices_gpmf_slice(tmp_path):
    rec = GoProRecording(recording_num=1)
    rec.chapters = [GoProChapter(Path("/fake/GX010001.MP4"), 1, 1, 13200.0, _CREATION)]
    concat_file = tmp_path / "episode_04.mp4"
    concat_file.write_bytes(b"\x00" * 1024)
    ep = Episode(index=4, recordings=[rec], date="2026-03-04", concat_path=concat_file)

    with patch(
        "gomoveshift.pipeline.ingest.concatenator.subprocess.run", side_effect=_ffmpeg_success
    ):
        sub_eps = _split_episode_at_points(
            ep, split_points=[2880.0, 5760.0], output_dir=tmp_path, stages_dir=tmp_path / "_s"
        )

    assert sub_eps
    assert all(s.gpx_source == "gpmf_slice" for s in sub_eps)


def test_resume_preset_sub_slice_tagged_gpmf_slice(tmp_path):
    # Sub-slice re-entering extract_and_merge_gpx on resume (recordings=[],
    # _explicit_total_duration set, gpx_path pre-set) → tagged gpmf_slice.
    gpx_file = tmp_path / "episode_04_a.gpx"
    gpx_file.write_text(_VALID_GPX, encoding="utf-8")
    sub_ep = Episode(
        index=401,
        recordings=[],
        date="2026-03-04",
        gpx_path=gpx_file,
        _explicit_total_duration=2880.0,
    )
    assert extract_and_merge_gpx(sub_ep, tmp_path) == gpx_file
    assert sub_ep.gpx_source == "gpmf_slice"


def test_resume_preset_external_sidecar_stays_external(tmp_path):
    # A pre-set GPX on a non-slice (operator sidecar) stays ``external``.
    gpx_file = tmp_path / "operator.gpx"
    gpx_file.write_text(_VALID_GPX, encoding="utf-8")
    ep = Episode(index=1, recordings=[], date="2026-03-04", gpx_path=gpx_file)
    assert extract_and_merge_gpx(ep, tmp_path) == gpx_file
    assert ep.gpx_source == "external"
