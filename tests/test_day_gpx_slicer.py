"""Day-GPX slicer: one whole-day GPX -> per-file ``<stem>.gpx`` sidecars."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.ingest import day_gpx_slicer as dgs
from gomoveshift.pipeline.ingest.day_gpx_slicer import (
    DayGpxSliceError,
    slice_day_gpx,
)

UTC = timezone.utc
# Camera clock ran at UTC-4: local 08:00:00 == 12:00:00Z.
A = "VID_20240101_080000_00_001.mp4"
B = "VID_20240101_090000_00_002.mp4"
SHORT = "VID_20240101_083000_00_003.mp4"
A_START = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
B_START = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)


def _day_gpx(path: Path, start: datetime, hours: int = 4) -> Path:
    pts = []
    for i in range(hours * 3600):  # 1 Hz
        t = start + timedelta(seconds=i)
        pts.append(
            f'<trkpt lat="51.{i % 1000:03d}" lon="-9.{i % 1000:03d}">'
            f"<time>{t.strftime('%Y-%m-%dT%H:%M:%SZ')}</time></trkpt>"
        )
    path.write_text(
        '<?xml version="1.0"?><gpx version="1.1"><trk><trkseg>'
        + "".join(pts)
        + "</trkseg></trk></gpx>"
    )
    return path


def _times(gpx: Path) -> list[datetime]:
    return [
        datetime.fromisoformat(m.replace("Z", "+00:00")) for m in dgs.TIME.findall(gpx.read_text())
    ]


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / A).write_bytes(b"x")
    (d / B).write_bytes(b"x")
    return d


@pytest.fixture
def day(tmp_path):
    return _day_gpx(tmp_path / "day.gpx", datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC))


@pytest.fixture(autouse=True)
def _durations():
    with patch.object(
        dgs, "get_clip_duration", side_effect=lambda p: {A: 600.0, B: 900.0, SHORT: 300.0}[p.name]
    ):
        yield


def test_each_sidecar_holds_only_its_padded_window(src, day):
    results = slice_day_gpx(day, src, utc_offset_hours=-4)
    assert {r.sidecar.name for r in results} == {Path(A).stem + ".gpx", Path(B).stem + ".gpx"}
    ta = _times(src / (Path(A).stem + ".gpx"))
    tb = _times(src / (Path(B).stem + ".gpx"))
    assert ta[0] == A_START - timedelta(seconds=15)
    assert ta[-1] == A_START + timedelta(seconds=600 + 120)
    assert tb[0] == B_START - timedelta(seconds=15)
    assert tb[-1] == B_START + timedelta(seconds=900 + 120)


def test_chapter_starts_where_the_previous_chapter_ends(tmp_path, day):
    # Insta360 splits one recording into chapters that share the recording's
    # HHMMSS; only the sequence number moves. Chapter 2 starts after chapter 1.
    d = tmp_path / "chap"
    d.mkdir()
    c1, c2 = "VID_20240101_080000_00_030.mp4", "VID_20240101_080000_00_031.mp4"
    for name in (c1, c2):
        (d / name).write_bytes(b"x")
    durs = {c1: 1800.0, c2: 900.0}
    with patch.object(dgs, "get_clip_duration", side_effect=lambda p: durs[p.name]):
        results = slice_day_gpx(day, d, utc_offset_hours=-4)
    starts = {r.sidecar.name: r.start_utc for r in results}
    assert starts[Path(c1).stem + ".gpx"] == A_START
    assert starts[Path(c2).stem + ".gpx"] == A_START + timedelta(seconds=1800)
    t2 = _times(d / (Path(c2).stem + ".gpx"))
    assert t2[0] == A_START + timedelta(seconds=1800 - 15)
    assert t2[-1] == A_START + timedelta(seconds=1800 + 900 + 120)


def test_chapter_with_a_missing_predecessor_refused(tmp_path, day):
    d = tmp_path / "chap"
    d.mkdir()
    for name in ("VID_20240101_080000_00_030.mp4", "VID_20240101_080000_00_032.mp4"):
        (d / name).write_bytes(b"x")
    with patch.object(dgs, "get_clip_duration", return_value=600.0):
        with pytest.raises(DayGpxSliceError, match="chapter"):
            slice_day_gpx(day, d, utc_offset_hours=-4)
    assert not list(d.glob("*.gpx*"))


def test_wrong_offset_refused_when_no_points(src, day):
    # +10 puts the clips at 22:00Z the previous day, outside the 4 h day GPX (10:00-14:00Z).
    with pytest.raises(DayGpxSliceError, match="no trackpoints"):
        slice_day_gpx(day, src, utc_offset_hours=10)
    assert not list(src.glob("*.gpx"))


def test_foreign_sidecar_not_overwritten(src, day):
    foreign = src / (Path(A).stem + ".gpx")
    foreign.write_text("<gpx>operator</gpx>")
    with pytest.raises(DayGpxSliceError, match="not written by"):
        slice_day_gpx(day, src, utc_offset_hours=-4)
    assert foreign.read_text() == "<gpx>operator</gpx>"


def test_own_sidecar_is_rewritten(src, day):
    slice_day_gpx(day, src, utc_offset_hours=-4)
    slice_day_gpx(day, src, utc_offset_hours=-4)  # marker present -> idempotent


def test_short_clip_refused_by_unchanged_span_gate(src, day):
    (src / SHORT).write_bytes(b"x")  # 300 s: 435 s window is 135 s over, gate allows 120
    with pytest.raises(DayGpxSliceError, match="Phase-A gate"):
        slice_day_gpx(day, src, utc_offset_hours=-4)
    assert not list(src.glob("*.gpx*"))


def test_unparseable_filename_refused(src, day):
    (src / "clip.mp4").write_bytes(b"x")
    with pytest.raises(DayGpxSliceError, match="VID_"):
        slice_day_gpx(day, src, utc_offset_hours=-4)


def test_sliced_sidecar_passes_existing_gates_unchanged(src, day):
    from gomoveshift.analysis.gps_parser import parse_gpx_file
    from gomoveshift.pipeline.gopro_gps import assert_gpx_video_correspondence

    slice_day_gpx(day, src, utc_offset_hours=-4)
    gps = parse_gpx_file(src / (Path(A).stem + ".gpx"))
    assert_gpx_video_correspondence(gps, A_START, 600.0)


def test_drive_requires_offset_with_day_gpx():
    from gomoveshift.cli.handlers.drive import cmd_drive

    args = argparse.Namespace(
        day_gpx="/x/day.gpx",
        camera_utc_offset_hours=None,
        publish="sun",
        lf_day=None,
        list_lfs=False,
        label_dry_run=False,
        skip_stop_verify=False,
    )
    with pytest.raises(SystemExit, match="--camera-utc-offset-hours"):
        cmd_drive(args, MagicMock())


def test_lrv_proxy_skipped_not_refused(src, day):
    (src / "LRV_20240101_080000_01_001.mp4").write_bytes(b"x")
    results = slice_day_gpx(day, src, utc_offset_hours=-4)
    assert len(results) == 2
    assert not (src / "LRV_20240101_080000_01_001.gpx").exists()


def test_other_unparseable_mp4_still_refused_with_lrv_present(src, day):
    (src / "LRV_20240101_080000_01_001.mp4").write_bytes(b"x")
    (src / "renamed.mp4").write_bytes(b"x")
    with pytest.raises(DayGpxSliceError, match="renamed.mp4"):
        slice_day_gpx(day, src, utc_offset_hours=-4)


def test_only_lrv_proxies_is_refused(tmp_path, day):
    d = tmp_path / "lrvonly"
    d.mkdir()
    (d / "LRV_20240101_080000_01_001.mp4").write_bytes(b"x")
    with pytest.raises(DayGpxSliceError, match="no .mp4"):
        slice_day_gpx(day, d, utc_offset_hours=-4)


def test_creation_time_disagreement_warns(src, day, caplog):
    ct = A_START + timedelta(seconds=61)
    with (
        patch.object(
            dgs, "_ffprobe_creation_time", side_effect=lambda p: ct if p.name == A else None
        ),
        caplog.at_level("WARNING", logger=dgs.logger.name),
    ):
        slice_day_gpx(day, src, utc_offset_hours=-4)
    assert any("creation_time" in r.message and A in r.message for r in caplog.records)


def test_creation_time_within_tolerance_silent(src, day, caplog):
    ct = A_START + timedelta(seconds=60)
    with (
        patch.object(dgs, "_ffprobe_creation_time", return_value=ct),
        caplog.at_level("WARNING", logger=dgs.logger.name),
    ):
        slice_day_gpx(day, src, utc_offset_hours=-4)
    # B's filename start is about 1 h from ct, so only assert A did not warn.
    assert not any(A in r.message and "creation_time" in r.message for r in caplog.records)


def test_zero_duration_refused(src, day):
    with patch.object(dgs, "get_clip_duration", return_value=0.0):
        with pytest.raises(DayGpxSliceError, match="duration"):
            slice_day_gpx(day, src, utc_offset_hours=-4)
    assert not list(src.glob("*.gpx*"))


def test_missing_day_gpx_refused(src, tmp_path):
    with pytest.raises(DayGpxSliceError, match="day GPX not found"):
        slice_day_gpx(tmp_path / "nope.gpx", src, utc_offset_hours=-4)


def _drive_args(**over):
    base = dict(
        day_gpx="/x/day.gpx",
        camera_utc_offset_hours=-4.0,
        directory="/x/src",
        publish="sun",
        lf_day=None,
        list_lfs=False,
        dry_run=False,
        label_dry_run=False,
        skip_stop_verify=False,
        no_tmux=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


class _ReachedError(Exception):
    """Raised by a patched configure_logging, which runs right after the slicer."""


def test_cmd_drive_calls_slicer_with_args(tmp_path):
    from gomoveshift.cli.handlers.drive import cmd_drive

    args = _drive_args(output_dir=str(tmp_path / "out"))
    with (
        patch.object(dgs, "slice_day_gpx", return_value=[]) as sl,
        patch("gomoveshift.cli.handlers.drive.configure_logging", side_effect=_ReachedError),
        pytest.raises(_ReachedError),
    ):
        cmd_drive(args, MagicMock())
    sl.assert_called_once_with(Path("/x/day.gpx"), Path("/x/src"), -4.0)


def test_cmd_drive_slice_error_returns_2(capsys):
    from gomoveshift.cli.handlers.drive import cmd_drive

    with patch.object(dgs, "slice_day_gpx", side_effect=DayGpxSliceError("boom")):
        assert cmd_drive(_drive_args(), MagicMock()) == 2
    assert "boom" in capsys.readouterr().err


def test_cmd_drive_dry_run_does_not_write_sidecars(tmp_path):
    from gomoveshift.cli.handlers.drive import cmd_drive

    args = _drive_args(dry_run=True, output_dir=str(tmp_path / "out"))
    with (
        patch.object(dgs, "slice_day_gpx") as sl,
        patch("gomoveshift.cli.handlers.drive.configure_logging", side_effect=_ReachedError),
        pytest.raises(_ReachedError),  # proves cmd_drive got past the slicer's slot
    ):
        cmd_drive(args, MagicMock())
    sl.assert_not_called()


def test_parser_registers_flags():
    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        ns = _build_parser().parse_args(
            [
                "drive",
                "/s",
                "--publish",
                "sun",
                "--day-gpx",
                "/d.gpx",
                "--camera-utc-offset-hours",
                "-4",
            ]
        )
    assert ns.day_gpx == "/d.gpx" and ns.camera_utc_offset_hours == -4.0
