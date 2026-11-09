"""GoPro ingest takes the day-GPX sidecars when its own GPMF has no lock.

A GoPro that never locked (``nctopa-2026-03-05-rec0063``: 13 chapters, zero
fixes) has no GPS at all; ``--day-gpx`` slices a phone track into one
``<chapter stem>.gpx`` beside every chapter. ``extract_and_merge_gpx`` must
read those for an episode whose GPMF does not cover every chapter, tagged
``external`` so Phase A runs the correspondence gate; GPMF that covers the
whole episode still wins (frame-aligned, and a footage-clock track cannot be
mixed with wall-clock sidecar points).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.ingest.concatenator import extract_and_merge_gpx
from gomoveshift.pipeline.ingest.day_gpx_slicer import MARKER
from gomoveshift.pipeline.ingest.gopro_parser import Episode, GoProChapter, GoProRecording

UTC = timezone.utc
START = datetime(2026, 3, 5, 17, 0, 13, tzinfo=UTC)


def _sidecar(path: Path, start: datetime, seconds: int, creator: str = MARKER) -> Path:
    pts = "".join(
        f'<trkpt lat="39.{i % 1000:03d}" lon="-77.{i % 1000:03d}">'
        f"<time>{(start + timedelta(seconds=i)).strftime('%Y-%m-%dT%H:%M:%SZ')}</time></trkpt>"
        for i in range(seconds + 1)  # inclusive end: last point is start + seconds
    )
    path.write_text(
        f'<?xml version="1.0"?><gpx version="1.1" creator="{creator}">'
        f"<trk><trkseg>{pts}</trkseg></trk></gpx>"
    )
    return path


def _episode(src: Path, durations=(1025.0, 1025.0, 671.0)) -> Episode:
    chapters = []
    for i, dur in enumerate(durations, start=1):
        p = src / f"GX0{i}0063.MP4"
        p.write_bytes(b"x")
        chapters.append(
            GoProChapter(
                path=p,
                chapter_num=i,
                recording_num=63,
                duration=dur,
                creation_time=START if i == 1 else None,
            )
        )
    return Episode(
        index=1,
        recordings=[GoProRecording(recording_num=63, chapters=chapters)],
        date="2026-03-05",
    )


def _times(gpx: Path) -> list[datetime]:
    import gpxpy

    with open(gpx, encoding="utf-8") as f:
        g = gpxpy.parse(f)
    return [pt.time for tr in g.tracks for seg in tr.segments for pt in seg.points]


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    return d


def test_no_lock_episode_takes_the_sidecars_as_external(src, tmp_path):
    ep = _episode(src)
    # Sidecars as the slicer writes them: padded, overlapping at the chapter joins.
    _sidecar(src / "GX010063.gpx", START - timedelta(seconds=15), 15 + 1025 + 120)
    _sidecar(src / "GX020063.gpx", START + timedelta(seconds=1025 - 15), 15 + 1025 + 120)
    _sidecar(src / "GX030063.gpx", START + timedelta(seconds=2050 - 15), 15 + 671 + 120)
    out = tmp_path / "out"
    with patch("gomoveshift.pipeline.gopro_gps.detect_gpmf_stream", return_value=False):
        merged = extract_and_merge_gpx(ep, out)
    assert merged is not None and merged == ep.gpx_path and merged.parent == out
    assert ep.gpx_source == "external"
    times = _times(merged)
    assert times == sorted(times) and len(times) == len(set(times)), (
        "one monotone track, no join duplicates"
    )
    assert times[0] == START - timedelta(seconds=15)
    assert times[-1] == START + timedelta(seconds=2050 + 671 + 120)


def test_partial_lock_episode_takes_the_sidecars(src, tmp_path):
    """Chapter 1 locked, chapters 2-3 never did (rec0064's shape): sidecars for all."""
    ep = _episode(src)
    for i, (lead, dur) in enumerate(((0, 1025), (1025, 1025), (2050, 671)), start=1):
        _sidecar(src / f"GX0{i}0063.gpx", START + timedelta(seconds=lead - 15), 15 + dur + 120)
    out = tmp_path / "out"
    out.mkdir()

    def fake_extract(video_path, output_dir, media_offset_s=0.0):
        if video_path.name == "GX010063.MP4":
            return _sidecar(output_dir / "GX010063_gpmf.gpx", START, 1025)
        return None

    with (
        patch("gomoveshift.pipeline.gopro_gps.detect_gpmf_stream", return_value=True),
        patch("gomoveshift.pipeline.gopro_gps.extract_gopro_gpx", side_effect=fake_extract),
    ):
        merged = extract_and_merge_gpx(ep, out)
    assert ep.gpx_source == "external"
    assert _times(merged)[-1] == START + timedelta(seconds=2050 + 671 + 120)
    assert not (out / "GX010063_gpmf.gpx").exists(), "the discarded GPMF part is cleaned up"


def test_full_lock_episode_keeps_gpmf_over_the_sidecars(src, tmp_path):
    ep = _episode(src, durations=(1025.0,))
    _sidecar(src / "GX010063.gpx", START - timedelta(seconds=15), 15 + 1025 + 120)
    out = tmp_path / "out"
    out.mkdir()

    def fake_extract(video_path, output_dir, media_offset_s=0.0):
        return _sidecar(output_dir / "part.gpx", START, 1025, creator="gpmf")

    with (
        patch("gomoveshift.pipeline.gopro_gps.detect_gpmf_stream", return_value=True),
        patch("gomoveshift.pipeline.gopro_gps.extract_gopro_gpx", side_effect=fake_extract),
    ):
        merged = extract_and_merge_gpx(ep, out)
    assert ep.gpx_source == "gpmf"
    assert len(_times(merged)) == 1026


def test_sidecars_for_only_some_chapters_refused(src, tmp_path):
    ep = _episode(src)
    _sidecar(src / "GX010063.gpx", START - timedelta(seconds=15), 15 + 1025 + 120)
    with patch("gomoveshift.pipeline.gopro_gps.detect_gpmf_stream", return_value=False):
        with pytest.raises(RuntimeError, match="GX020063.gpx"):
            extract_and_merge_gpx(ep, tmp_path / "out")


def test_no_lock_and_no_sidecars_stays_gps_less(src, tmp_path):
    ep = _episode(src)
    with patch("gomoveshift.pipeline.gopro_gps.detect_gpmf_stream", return_value=False):
        assert extract_and_merge_gpx(ep, tmp_path / "out") is None
    assert ep.gpx_path is None and ep.gpx_source == "gpmf"
