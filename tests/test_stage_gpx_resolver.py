"""Standalone stages must not adopt an arbitrary ``*.gpx``.

``chapters``/``shorts``/``metadata`` historically resolved GPS via
``glob("*.gpx")[0]`` — picking a sidecar by filesystem order, unvalidated.
A wrong-drive GPX left in the folder then silently drove chapters, the
title overwrite, per-clip shorts labels and the recording date. The drive
path already gates external GPX with ``assert_gpx_video_correspondence``;
``resolve_corresponding_gpx`` reuses that gate for the standalone stages.

Accept: an explicit ``--gps`` path, or an exact ``<video-stem>.gpx``
sidecar (trusted by name), or an arbitrary ``*.gpx`` that passes the
correspondence gate. Refuse: an arbitrary ``*.gpx`` that does not.
"""

from datetime import datetime, timezone

import pytest

from gomoveshift.pipeline import gopro_gps
from gomoveshift.pipeline.gopro_gps import resolve_corresponding_gpx

CREATION = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _write_gpx(path, start_unix, span_sec, lat=32.42, lon=-80.46):
    """Write a two-point GPX spanning ``span_sec`` from ``start_unix``."""
    t0 = datetime.fromtimestamp(start_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t1 = datetime.fromtimestamp(start_unix + span_sec, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">\n'
        "<trk><trkseg>\n"
        f'<trkpt lat="{lat}" lon="{lon}"><time>{t0}</time></trkpt>\n'
        f'<trkpt lat="{lat + 0.01}" lon="{lon + 0.01}"><time>{t1}</time></trkpt>\n'
        "</trkseg></trk></gpx>\n"
    )


@pytest.fixture
def stub_probe(monkeypatch):
    """Stub the ffprobe-backed video extractors so tests need no real MP4."""
    monkeypatch.setattr(gopro_gps, "extract_recording_date", lambda _p: CREATION)
    monkeypatch.setattr(gopro_gps, "extract_location_coords", lambda _p: None)


def test_arbitrary_nonmatching_gpx_refused(tmp_path, stub_probe):
    """A wrong-stem GPX from a different drive is refused, not adopted."""
    video = tmp_path / "myclip.mp4"
    video.touch()
    # GPX starts 2h after the video's creation_time — no correspondence.
    _write_gpx(tmp_path / "someotherdrive.gpx", CREATION.timestamp() + 2 * 3600, 600)
    assert resolve_corresponding_gpx(video, video_duration=600.0) is None


def test_exact_stem_sidecar_adopted(tmp_path, stub_probe):
    """An exact ``<stem>.gpx`` sidecar is trusted by name, no gate needed."""
    video = tmp_path / "myclip.mp4"
    video.touch()
    sidecar = tmp_path / "myclip.gpx"
    # Even a temporally-off sidecar is trusted because the name matches.
    _write_gpx(sidecar, CREATION.timestamp() + 2 * 3600, 600)
    assert resolve_corresponding_gpx(video, video_duration=600.0) == sidecar


def test_explicit_gps_adopted(tmp_path, stub_probe):
    """An explicit operator ``--gps`` path is authoritative, no gate needed."""
    video = tmp_path / "myclip.mp4"
    video.touch()
    explicit = tmp_path / "operator_supplied.gpx"
    _write_gpx(explicit, CREATION.timestamp() + 2 * 3600, 600)
    assert resolve_corresponding_gpx(video, explicit_gpx=explicit, video_duration=600.0) == explicit


def test_arbitrary_matching_gpx_adopted(tmp_path, stub_probe):
    """A wrong-stem GPX that DOES correspond passes the gate and is adopted."""
    video = tmp_path / "myclip.mp4"
    video.touch()
    match = tmp_path / "randomname.gpx"
    _write_gpx(match, CREATION.timestamp() + 5, 600)  # +5s start, 600s span
    assert resolve_corresponding_gpx(video, video_duration=600.0) == match
