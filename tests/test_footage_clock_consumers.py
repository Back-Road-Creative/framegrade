"""Resume, longform and the legacy shorts path read the footage clock.

Every GPX fix is stamped with its footage second and ``effective_offset``
returns footage second 0. These consumers still anchored at the raw first fix or
stretched the GPS span over the video, so on a stamped GPX they stayed late by
the lock delay and drifted across recording gaps.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gpxpy
import gpxpy.gpx
import numpy as np

from gomoveshift.analysis.gps_parser import parse_gpx_file
from gomoveshift.analysis.shorts_extractor import ShortsExtractor
from gomoveshift.analysis.shorts_models import effective_offset, footage_origin
from gomoveshift.analysis.shorts_time_mapping import VideoToGpsTimeMapper
from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.gps_enrichment import rebuild_gps_per_episode
from gomoveshift.pipeline.longform_assembler import LongFormAssembler
from gomoveshift.pipeline.shorts_extractor import _derive_short_location_map
from test_gps_media_time_anchor import T0, _samples, _write


def _stamped_gap_gpx(tmp_path):
    # Rec 1: first fix 90 s after record start, footage 90..699. Rec 2 follows
    # immediately in the concat (footage 700..799) but 30 min later on the wall.
    rec1 = _samples(90, 610, T0 + timedelta(seconds=90), lat=32.0)
    rec2 = _samples(700, 100, T0 + timedelta(seconds=700 + 1800), lat=40.0)
    return _write(tmp_path / "ep.gpx", rec1 + rec2)


def _plain_gpx(path, pts):
    gpx = gpxpy.gpx.GPX()
    seg = gpxpy.gpx.GPXTrackSegment()
    gpx.tracks.append(gpxpy.gpx.GPXTrack())
    gpx.tracks[0].segments.append(seg)
    for i, (lat, lon) in enumerate(pts):
        seg.points.append(
            gpxpy.gpx.GPXTrackPoint(
                lat, lon, elevation=5.0, time=datetime(2026, 5, 1, 12, 0, i, tzinfo=timezone.utc)
            )
        )
    Path(path).write_text(gpx.to_xml(), encoding="utf-8")
    return Path(path)


# 1. resume parity -----------------------------------------------------------


def test_resume_offset_matches_fresh_ingest_stamped(tmp_path):
    p = _stamped_gap_gpx(tmp_path)
    data = parse_gpx_file(p)
    got = rebuild_gps_per_episode([str(p)])[0][1]
    assert got == effective_offset(data, data.start_time)
    assert got == footage_origin(data)
    assert got == data.start_time - 90


def test_resume_offset_matches_fresh_ingest_null_island_lead(tmp_path):
    pts = [(0.0, 0.0)] * 5 + [(32.0 + i * 1e-4, -80.0) for i in range(10)]
    p = _plain_gpx(tmp_path / "n.gpx", pts)
    data = parse_gpx_file(p)
    got = rebuild_gps_per_episode([str(p)])[0][1]
    assert got == effective_offset(data, data.start_time)
    assert got != data.start_time


def test_resume_offset_plain_unstamped_unchanged(tmp_path):
    p = _plain_gpx(tmp_path / "c.gpx", [(32.0 + i * 1e-4, -80.0) for i in range(10)])
    data = parse_gpx_file(p)
    assert rebuild_gps_per_episode([str(p)])[0][1] == data.start_time


# 2. longform ----------------------------------------------------------------


def _lf_center(gps, offset):
    n = 800
    scores = np.zeros((n, 9), dtype=np.float64)
    scores[:, 0] = np.arange(n, dtype=np.float64)
    scores[:, 1] = 0.1
    scores[700:760, 1] = 0.8
    scores[:, 2] = 0.05
    asm = LongFormAssembler(DriveConfig(min_segment_duration=60.0, max_segment_duration=0.0))
    segs = asm.find_segments_above_threshold(
        scores, threshold=0.5, file_boundaries=[(0.0, 800.0)], gps_per_episode=[(gps, offset)]
    )
    assert len(segs) == 1
    return segs[0].gps_center_lat, segs[0].gps_center_lon


def test_longform_uses_footage_origin_for_stored_raw_offset(tmp_path):
    p = _stamped_gap_gpx(tmp_path)
    data = parse_gpx_file(p)
    lats = [tp.lat for tp in data.trackpoints if 700 <= tp.media_s <= 760]
    expected = round(sum(lats) / len(lats), 6)
    raw = _lf_center(data, data.start_time)
    origin = _lf_center(data, footage_origin(data))
    assert raw[0] == origin[0] == expected


def test_longform_dict_with_and_without_media_s(tmp_path):
    p = _stamped_gap_gpx(tmp_path)
    data = parse_gpx_file(p)
    full = {"trackpoints": [tp.__dict__ for tp in data.trackpoints]}
    stripped = {"trackpoints": [{**tp.__dict__, "media_s": None} for tp in data.trackpoints]}
    assert _lf_center(full, data.start_time)[0] == _lf_center(data, data.start_time)[0]
    # unstamped dict: offset used as given (no crash)
    _lf_center(stripped, data.start_time)


# 3. mapper ------------------------------------------------------------------


def test_mapper_footage_clock_never_scales():
    w0 = 1000.0
    for gd in (0.0, 5000.0):
        for od in (0.0, 300.0):
            m = VideoToGpsTimeMapper(
                gps_offset=w0,
                keep_segments=[(0, 100), (200, 300)],
                gps_duration=gd,
                original_duration=od,
                footage_clock=True,
            )
            assert m.video_to_gps_time(150) == w0 + 250
    m = VideoToGpsTimeMapper(gps_offset=w0, footage_clock=True)
    assert m.video_to_gps_time(42) == w0 + 42


def test_mapper_default_still_scales():
    m = VideoToGpsTimeMapper(
        gps_offset=1000.0,
        keep_segments=[(0, 100), (200, 300)],
        gps_duration=600.0,
        original_duration=300.0,
    )
    assert m.video_to_gps_time(150) == 1000.0 + 250 * 2.0


# 4. legacy shorts location map ---------------------------------------------


def test_derive_short_location_map_ignores_wall_clock_span(tmp_path):
    p = _stamped_gap_gpx(tmp_path)
    captured = {}

    def _ost(pts, short):
        captured["pts"] = pts
        return "label"

    cfg = MagicMock()
    cfg.stop_detection.crossfade_duration = 0.0
    short = SimpleNamespace(index=1, start_time=700.0, end_time=730.0)
    with (
        patch("gomoveshift.pipeline.gps_enrichment._build_shared_geocoder", return_value=object()),
        patch("gomoveshift.pipeline.gps_enrichment._make_short_ost_text_fn", return_value=_ost),
    ):
        out = _derive_short_location_map([short], p, [(0.0, 800.0)], 2600.0, 800.0, cfg)
    assert out == {1: "label"}
    assert captured["pts"]
    assert all(lat >= 40.0 for _, lat, _ in captured["pts"])
    assert len(captured["pts"]) == 31


# 5. analysis ShortsExtractor ------------------------------------------------


def test_set_gps_data_sets_footage_clock_and_survives_resync(tmp_path):
    data = parse_gpx_file(_stamped_gap_gpx(tmp_path))
    ex = ShortsExtractor()
    ex.set_gps_data(data, gps_offset=footage_origin(data))
    assert ex._time_mapper.footage_clock is True
    ex._selector._score_segment = lambda *a, **k: 0.0
    ex._time_mapper.footage_clock = False
    ex._score_segment(MagicMock(), 100.0)
    assert ex._time_mapper.footage_clock is True


def test_set_gps_data_unstamped_is_not_footage_clock(tmp_path):
    data = parse_gpx_file(_plain_gpx(tmp_path / "c.gpx", [(32.0, -80.0)] * 5))
    ex = ShortsExtractor()
    ex.set_gps_data(data, gps_offset=data.start_time)
    assert ex._time_mapper.footage_clock is False
