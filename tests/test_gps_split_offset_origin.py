"""Single-origin GPS split offsets and sanitized slice windows.

Split offsets: split points from ``find_gps_split_points`` are anchored to the GPX
FIRST FIX, but ``-ss``/``-t`` index the concat video whose t=0 is record
start (``creation_time``). Applying first-fix offsets as footage-t0 offsets
lands every cut the GPS lock-delay ``D`` seconds early, and mis-windows each
sub-episode's sliced GPX relative to its video bytes.

Slice windows: ``_extract_gpx_time_range`` derived its window origin from
``min(p.time ...)`` over the RAW points — a leading invalid-coord / stale-clock
timestamp becomes the base and shifts (or empties) every window. Inclusive
upper bounds also duplicated a boundary fix into both adjacent slices.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import gpxpy
import gpxpy.gpx

from gomoveshift.pipeline.ingest.concatenator import (
    _extract_gpx_time_range,
    _split_episode_at_points,
)
from gomoveshift.pipeline.ingest.gopro_parser import Episode, GoProChapter, GoProRecording

CREATION = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _write_gpx(path, start_dt, n, step=60, lat=32.42, lon=-80.46, leading=None):
    g = gpxpy.gpx.GPX()
    tr = gpxpy.gpx.GPXTrack()
    g.tracks.append(tr)
    seg = gpxpy.gpx.GPXTrackSegment()
    tr.segments.append(seg)
    if leading is not None:
        llat, llon, ldt = leading
        seg.points.append(gpxpy.gpx.GPXTrackPoint(llat, llon, time=ldt))
    for i in range(n):
        seg.points.append(
            gpxpy.gpx.GPXTrackPoint(
                lat + i * 0.001, lon + i * 0.001, time=start_dt + timedelta(seconds=i * step)
            )
        )
    Path(path).write_text(g.to_xml())
    return Path(path)


def _make_parent(tmp_path, duration, creation, gpx_path):
    rec = GoProRecording(recording_num=1)
    rec.chapters = [GoProChapter(Path("/fake/GX010001.MP4"), 1, 1, duration, creation)]
    concat = tmp_path / "episode_04_2026-05-01.mp4"
    concat.write_bytes(b"\x00" * 1024)
    ep = Episode(index=4, recordings=[rec], date="2026-05-01", concat_path=concat)
    ep.gpx_path = gpx_path
    return ep


def _ffmpeg_capture(store):
    """Capture the ffmpeg CUT invocations, and only those.

    The split path also probes the source's keyframes so each cut lands on one
    (``concatenator._align_splits_to_keyframes``). ``concatenator.subprocess``
    IS the stdlib module object, so patching ``concatenator.subprocess.run``
    patches it for every importer — keycut's ffprobe arrives here too. Recording
    it as though it were a cut puts a command with no ``-ss``/``-t`` at the head
    of the list; writing to ``cmd[-1]`` would clobber the source file.

    Refusing the probe (rc=1) leaves the keyframes unknown, so the split points
    stay exactly where these tests put them — the offset-origin arithmetic below
    is what they are about, not the alignment.
    """

    def run(cmd, **kw):
        if not cmd or cmd[0] != "ffmpeg":
            return MagicMock(returncode=1, stdout=b"", stderr="")
        Path(cmd[-1]).write_bytes(b"\x00" * 512)
        store.append(cmd)
        return MagicMock(returncode=0, stderr="")

    return run


def _offsets(gpx_file, anchor):
    pts = gpxpy.parse(Path(gpx_file).read_text()).tracks[0].segments[0].points
    return [(p.time - anchor).total_seconds() for p in pts]


# --- the GPS lock-delay D is folded into the ffmpeg cut offsets ---


def test_split_offsets_include_lock_delay(tmp_path):
    first_fix = CREATION + timedelta(seconds=120)  # D = 120s lock delay
    gpx = _write_gpx(tmp_path / "episode_04.gpx", first_fix, n=140, step=60)
    ep = _make_parent(tmp_path, duration=8400.0, creation=CREATION, gpx_path=gpx)
    (tmp_path / "_s").mkdir()
    cmds = []
    with patch(
        "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
        side_effect=_ffmpeg_capture(cmds),
    ):
        _split_episode_at_points(
            ep, split_points=[3600.0], output_dir=tmp_path, stages_dir=tmp_path / "_s"
        )
    ss_values = [float(c[c.index("-ss") + 1]) for c in cmds]
    # First-fix-anchored split X=3600 must map to video-time -ss 3600+120=3720.
    assert 3720.0 in ss_values, ss_values


def test_sub_gpx_window_stays_inside_its_video_slice(tmp_path):
    first_fix = CREATION + timedelta(seconds=120)
    gpx = _write_gpx(tmp_path / "episode_04.gpx", first_fix, n=141, step=60)
    ep = _make_parent(tmp_path, duration=8400.0, creation=CREATION, gpx_path=gpx)
    (tmp_path / "_s").mkdir()
    cmds = []
    with patch(
        "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
        side_effect=_ffmpeg_capture(cmds),
    ):
        sub = _split_episode_at_points(
            ep, split_points=[3600.0], output_dir=tmp_path, stages_dir=tmp_path / "_s"
        )
    assert len(sub) == 2
    # First slice: video window [ss, ss+t) in record-time seconds.
    first_cmd = cmds[0]
    ss = float(first_cmd[first_cmd.index("-ss") + 1])
    dur = float(first_cmd[first_cmd.index("-t") + 1])
    # The sliced GPX must cover the SAME record-time window as the video bytes.
    offs = _offsets(sub[0].gpx_path, CREATION)
    assert offs, "first sub-slice GPX is empty"
    assert all(ss <= o < ss + dur for o in offs), (ss, dur, min(offs), max(offs))


# --- window origin is the sanitized first fix, not raw min(); exclusive ---


def test_extract_gpx_ignores_leading_null_island_for_base_time(tmp_path):
    # Leading (0,0) null-island fix with a stale clock 1h in the past: raw
    # min() would base the window on it, shift real points out of range, and
    # leak the invalid point into the slice.
    leading = (0.0, 0.0, CREATION - timedelta(hours=1))
    gpx = _write_gpx(tmp_path / "m.gpx", CREATION, n=20, step=60, leading=leading)
    out = _extract_gpx_time_range(gpx, 0.0, 600.0, tmp_path, "slice.gpx")
    assert out is not None
    pts = gpxpy.parse(Path(out).read_text()).tracks[0].segments[0].points
    # Window [0,600) relative to the first VALID fix → offsets 0,60,...,540.
    assert len(pts) == 10, [round((p.time - CREATION).total_seconds()) for p in pts]
    assert all(not (abs(p.latitude) < 0.001 and abs(p.longitude) < 0.001) for p in pts)


def test_slice_window_upper_bound_exclusive(tmp_path):
    gpx = _write_gpx(tmp_path / "m.gpx", CREATION, n=20, step=60)
    a = _extract_gpx_time_range(gpx, 0.0, 600.0, tmp_path, "a.gpx")
    b = _extract_gpx_time_range(gpx, 600.0, 1200.0, tmp_path, "b.gpx")
    off_a = {round(o) for o in _offsets(a, CREATION)}
    off_b = {round(o) for o in _offsets(b, CREATION)}
    # The boundary fix at offset 600 belongs to exactly one slice (the upper).
    assert 600 in off_b
    assert 600 not in off_a


# --- Degrade: no GPX / no clock → no shift (record-start == first-fix frame) ---


def test_no_gpx_leaves_offsets_unshifted(tmp_path):
    ep = _make_parent(tmp_path, duration=8400.0, creation=CREATION, gpx_path=None)
    (tmp_path / "_s").mkdir()
    cmds = []
    with patch(
        "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
        side_effect=_ffmpeg_capture(cmds),
    ):
        _split_episode_at_points(
            ep, split_points=[3600.0], output_dir=tmp_path, stages_dir=tmp_path / "_s"
        )
    ss_values = [float(c[c.index("-ss") + 1]) for c in cmds]
    assert ss_values == [0.0, 3600.0]
