"""GPS fixes carry their footage second (``gms_media_s``).

Every footage-second -> GPS-time lookup used ``origin + t`` with origin = the
wall clock of the FIRST LOCKED FIX. That is late by the GPS lock delay, and
after a stop inside a multi-recording episode (concat skips the gap, wall
clock does not) it reads GPS from up to two hours earlier in the drive. Each
trackpoint now records its position in the episode concat timeline, and the
parser rebuilds a footage clock from it. None of these tests needs
``gopro_overlay`` (not installable in CI).
"""

import defusedxml.ElementTree as ET  # noqa: N817  drop-in for stdlib ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gpxpy
import gpxpy.gpx
import pytest

from gomoveshift.analysis.gps_parser import parse_gpx_file
from gomoveshift.analysis.shorts_models import (
    GPX_MEDIA_S_TAG,
    GPSData,
    GPSError,
    GPSTrackpoint,
    effective_offset,
    footage_origin,
)
from gomoveshift.pipeline.gopro_gps import build_stamped_gpx, iter_stamped_samples
from gomoveshift.pipeline.ingest.concatenator import (
    _extract_gpx_time_range,
    _merge_gpx_files,
    _split_episode_at_points,
    extract_and_merge_gpx,
)
from gomoveshift.pipeline.ingest.gopro_parser import Episode, GoProChapter, GoProRecording

T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
NS = "{http://www.topografix.com/GPX/1/1}"


def _samples(media_start, n, wall_start, lat=32.0, lon=-80.0):
    return [
        (media_start + i, wall_start + timedelta(seconds=i), lat + i * 1e-4, lon, 10.0)
        for i in range(n)
    ]


def _write(path, samples):
    Path(path).write_text(build_stamped_gpx(samples).to_xml(), encoding="utf-8")
    return Path(path)


def _media_values(path):
    root = ET.parse(str(path)).getroot()
    return [float(e.text) for e in root.iter(f"{NS}{GPX_MEDIA_S_TAG}")]


def _two_recording_gpx(tmp_path):
    # Rec 1: first fix arrives 90 s after record start. Rec 2 starts 30 min of
    # wall clock later but follows immediately in the concat video.
    rec1 = _samples(90, 60, T0 + timedelta(seconds=90))
    rec2 = _samples(150, 60, T0 + timedelta(seconds=150 + 1800))
    return _write(tmp_path / "ep.gpx", rec1 + rec2)


# 1. writer / builder -------------------------------------------------------


def test_build_stamped_gpx_stamps_each_point_and_round_trips(tmp_path):
    p = _write(tmp_path / "a.gpx", _samples(5, 3, T0))
    assert _media_values(p) == [5.0, 6.0, 7.0]
    data = parse_gpx_file(p)
    assert [tp.media_s for tp in data.trackpoints] == [5.0, 6.0, 7.0]


def test_build_stamped_gpx_leaves_wall_clock_time_untouched(tmp_path):
    p = _write(tmp_path / "a.gpx", _samples(5, 2, T0))
    pts = gpxpy.parse(p.read_text()).tracks[0].segments[0].points
    assert pts[0].time == T0 and pts[1].time == T0 + timedelta(seconds=1)


def _fake_fm(entries):
    return SimpleNamespace(items=lambda step=None: iter(entries))


def _entry(fix, ts_ms, sec):
    return SimpleNamespace(
        gpsfix=fix,
        timestamp=SimpleNamespace(magnitude=ts_ms),
        dt=T0 + timedelta(seconds=sec),
        point=SimpleNamespace(lat=1.0 + sec, lon=2.0),
        alt=SimpleNamespace(to=lambda unit: SimpleNamespace(magnitude=7.0)),
    )


def test_iter_stamped_samples_filters_locked_and_adds_offset():
    fm = _fake_fm([_entry(0, 0, 0), _entry(3, 1500, 1), _entry(3, 2500, 2)])
    out = list(iter_stamped_samples(fm, 100.0, lambda e: e.gpsfix == 3))
    assert [round(s[0], 3) for s in out] == [101.5, 102.5]
    assert out[0][1:] == (T0 + timedelta(seconds=1), 2.0, 2.0, 7.0)


def test_iter_stamped_samples_missing_timestamp_is_unstamped():
    e = _entry(3, 0, 0)
    e.timestamp = None
    out = list(iter_stamped_samples(_fake_fm([e]), 0.0, lambda e: True))
    assert out[0][0] is None


# 2. concat offsets ---------------------------------------------------------


def test_chapter_without_gpmf_still_advances_media_offset(tmp_path):
    chapters = [
        GoProChapter(Path(f"/fake/GX0{n}0001.MP4"), n, 1, dur, T0)
        for n, dur in ((1, 100.0), (2, 50.0), (3, 70.0))
    ]
    rec = GoProRecording(recording_num=1, chapters=chapters)
    ep = Episode(index=1, recordings=[rec], date="2026-05-01")
    offsets = {}

    def fake_extract(path, out_dir, media_offset_s=0.0):
        offsets[path.name] = media_offset_s
        return _write(out_dir / f"{path.stem}.gpx", _samples(media_offset_s, 3, T0))

    with (
        patch(
            "gomoveshift.pipeline.gopro_gps.detect_gpmf_stream",
            side_effect=lambda p: p.name != chapters[1].path.name,
        ),
        patch("gomoveshift.pipeline.gopro_gps.extract_gopro_gpx", side_effect=fake_extract),
    ):
        merged = extract_and_merge_gpx(ep, tmp_path)

    assert offsets[chapters[0].path.name] == 0.0
    assert offsets[chapters[2].path.name] == 150.0  # dur1 + dur2, dropped chapter counted
    assert _media_values(merged) == [0.0, 1.0, 2.0, 150.0, 151.0, 152.0]


def test_chapter_with_no_lock_still_advances_media_offset(tmp_path):
    chapters = [GoProChapter(Path(f"/fake/GX0{n}0001.MP4"), n, 1, 100.0, T0) for n in (1, 2, 3)]
    ep = Episode(index=1, recordings=[GoProRecording(recording_num=1, chapters=chapters)], date="d")
    offsets = {}

    def fake_extract(path, out_dir, media_offset_s=0.0):
        offsets[path.name] = media_offset_s
        if path.name == chapters[0].path.name:
            return None  # no GPS lock
        return _write(out_dir / f"{path.stem}.gpx", _samples(media_offset_s, 2, T0))

    with (
        patch("gomoveshift.pipeline.gopro_gps.detect_gpmf_stream", return_value=True),
        patch("gomoveshift.pipeline.gopro_gps.extract_gopro_gpx", side_effect=fake_extract),
    ):
        extract_and_merge_gpx(ep, tmp_path)

    assert offsets[chapters[1].path.name] == 100.0
    assert offsets[chapters[2].path.name] == 200.0


# 3. merge / slice ----------------------------------------------------------


def test_merge_keeps_media_extensions(tmp_path):
    a = _write(tmp_path / "a.gpx", _samples(0, 2, T0))
    b = _write(tmp_path / "b.gpx", _samples(300, 2, T0 + timedelta(hours=1)))
    ep = Episode(index=1, date="d")
    merged = _merge_gpx_files([a, b], tmp_path, ep)
    assert _media_values(merged) == [0.0, 1.0, 300.0, 301.0]


def test_slice_uses_media_s_half_open_and_rebases_to_zero(tmp_path):
    # Wall clock deliberately unrelated to media_s (30 min gap in the middle).
    samples = _samples(0, 50, T0) + _samples(50, 50, T0 + timedelta(seconds=50 + 1800))
    gpx = _write(tmp_path / "p.gpx", samples)
    out = _extract_gpx_time_range(gpx, 40.0, 60.0, tmp_path, "s.gpx", anchor_time=T0)
    assert out is not None
    assert _media_values(out) == [float(i) for i in range(20)]
    pts = gpxpy.parse(out.read_text()).tracks[0].segments[0].points
    assert len(pts) == 20
    assert pts[-1].latitude == pytest.approx(32.0 + 9 * 1e-4)


# 4. parser / footage clock -------------------------------------------------


def test_parse_builds_footage_clock_across_a_wall_clock_gap(tmp_path):
    data = parse_gpx_file(_two_recording_gpx(tmp_path))
    origin = footage_origin(data)
    assert origin == pytest.approx((T0 + timedelta(seconds=90)).timestamp() - 90)
    assert data.footage_origin == origin
    for tp in data.trackpoints:
        assert tp.timestamp == pytest.approx(origin + tp.media_s)
    post_gap = next(tp for tp in data.trackpoints if tp.media_s == 150.0)
    assert post_gap.timestamp == pytest.approx(T0.timestamp() + 150)  # not 30 min late


def test_parse_first_point_timestamp_equals_its_wall_clock(tmp_path):
    data = parse_gpx_file(_two_recording_gpx(tmp_path))
    assert data.start_time == pytest.approx((T0 + timedelta(seconds=90)).timestamp())


def test_footage_origin_dict_form_and_unstamped():
    stamped = {"trackpoints": [{"lat": 1, "lon": 1, "timestamp": 1000.0, "media_s": 30.0}]}
    assert footage_origin(stamped) == 970.0
    assert footage_origin({"trackpoints": [{"lat": 1, "lon": 1, "timestamp": 1.0}]}) is None
    assert footage_origin(GPSData(trackpoints=[GPSTrackpoint(1, 1, None, 1.0)])) is None
    assert footage_origin(None) is None


# 5. effective_offset -------------------------------------------------------


def test_effective_offset_stamped_returns_origin_despite_leading_null_island():
    tps = [GPSTrackpoint(0.0, 0.0, None, 1000.0 + i, media_s=10.0 + i) for i in range(3)] + [
        GPSTrackpoint(32.0, -80.0, None, 1003.0 + i, media_s=13.0 + i) for i in range(3)
    ]
    assert effective_offset(GPSData(trackpoints=tps), 1000.0) == 990.0


def test_effective_offset_stamped_dict_form():
    tps = [{"lat": 32.0, "lon": -80.0, "timestamp": 500.0, "media_s": 20.0}]
    assert effective_offset({"trackpoints": tps}, 0.0) == 480.0


def test_effective_offset_unstamped_unchanged():
    tps = [GPSTrackpoint(0.0, 0.0, None, 1000.0), GPSTrackpoint(32.0, -80.0, None, 1005.0)]
    assert effective_offset(GPSData(trackpoints=tps), 1000.0) == 1005.0


# 6. partial stamping -------------------------------------------------------


def test_partially_stamped_gpx_raises(tmp_path):
    stamped = _samples(0, 2, T0)
    unstamped = [(None, T0 + timedelta(seconds=10), 32.0, -80.0, 1.0)]
    p = _write(tmp_path / "mix.gpx", stamped + unstamped)
    with pytest.raises(GPSError):
        parse_gpx_file(p)


def test_unstamped_gpx_parses_as_before(tmp_path):
    p = _write(
        tmp_path / "u.gpx", [(None, T0 + timedelta(seconds=i), 32.0, -80.0, 1.0) for i in range(3)]
    )
    data = parse_gpx_file(p)
    assert footage_origin(data) is None
    assert data.start_time == T0.timestamp()


# 7. split ------------------------------------------------------------------


def test_split_uses_first_fix_media_s_as_lock_delay_not_creation_time(tmp_path):
    # creation_time says the first fix is 600 s in; the stamp says 120 s.
    samples = _samples(120, 140, T0 + timedelta(seconds=600))
    gpx = _write(tmp_path / "episode_04.gpx", samples)
    rec = GoProRecording(recording_num=1)
    rec.chapters = [GoProChapter(Path("/fake/GX010001.MP4"), 1, 1, 8400.0, T0)]
    concat = tmp_path / "episode_04_d.mp4"
    concat.write_bytes(b"\x00" * 1024)
    ep = Episode(index=4, recordings=[rec], date="d", concat_path=concat, gpx_path=gpx)
    (tmp_path / "_s").mkdir()
    cmds = []

    def run(cmd, **kw):
        if not cmd or cmd[0] != "ffmpeg":
            return MagicMock(returncode=1, stdout=b"", stderr="")
        Path(cmd[-1]).write_bytes(b"\x00")
        cmds.append(cmd)
        return MagicMock(returncode=0, stderr="")

    with patch("gomoveshift.pipeline.ingest.concatenator.subprocess.run", side_effect=run):
        _split_episode_at_points(ep, [3600.0], tmp_path, tmp_path / "_s")

    ss = [float(c[c.index("-ss") + 1]) for c in cmds]
    assert 3720.0 in ss, ss
