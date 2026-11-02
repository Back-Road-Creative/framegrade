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


def test_drive_no_longer_requires_an_offset_with_day_gpx(tmp_path):
    from gomoveshift.cli.handlers.drive import cmd_drive

    args = _drive_args(camera_utc_offset_hours=None, output_dir=str(tmp_path / "out"))
    with (
        patch.object(dgs, "slice_day_gpx", return_value=[]) as sl,
        patch("gomoveshift.cli.handlers.drive.configure_logging", side_effect=_ReachedError),
        pytest.raises(_ReachedError),
    ):
        cmd_drive(args, MagicMock())
    sl.assert_called_once_with(Path("/x/day.gpx"), Path("/x/src"), None)


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


def _ct(**by_name):
    return patch.object(dgs, "_ffprobe_creation_time", side_effect=lambda p: by_name.get(p.name))


def test_operator_offset_disagreeing_with_creation_time_is_refused(src, day):
    with _ct(**{A: A_START + timedelta(seconds=61)}):
        with pytest.raises(DayGpxSliceError, match=rf"{A}.*measured -4"):
            slice_day_gpx(day, src, utc_offset_hours=-4)
    assert not list(src.glob("*.gpx"))


def test_operator_offset_within_tolerance_is_accepted(src, day):
    with _ct(**{A: A_START + timedelta(seconds=60)}):
        assert slice_day_gpx(day, src, utc_offset_hours=-4)


def test_offset_is_measured_from_creation_time_when_not_given(src, day, caplog):
    with _ct(**{A: A_START, B: B_START}), caplog.at_level("INFO", logger=dgs.logger.name):
        results = slice_day_gpx(day, src)
    assert {r.start_utc for r in results} == {A_START, B_START}
    assert any("measured camera UTC offset -4" in r.message for r in caplog.records)


def test_measured_offset_tolerates_container_second_jitter(src, day):
    with _ct(**{A: A_START + timedelta(seconds=30), B: B_START - timedelta(seconds=20)}):
        assert slice_day_gpx(day, src)


def test_chapters_after_the_first_are_not_measured(tmp_path, day):
    d = tmp_path / "chap"
    d.mkdir()
    c1, c2 = "VID_20240101_080000_00_030.mp4", "VID_20240101_080000_00_031.mp4"
    for name in (c1, c2):
        (d / name).write_bytes(b"x")
    durs = {c1: 1800.0, c2: 900.0}
    with (
        patch.object(dgs, "get_clip_duration", side_effect=lambda p: durs[p.name]),
        _ct(**{c1: A_START, c2: A_START + timedelta(seconds=1800)}),
    ):
        assert len(slice_day_gpx(day, d)) == 2


def test_no_offset_and_no_creation_time_is_refused_naming_the_clips(src, day):
    with _ct():
        with pytest.raises(DayGpxSliceError, match=rf"creation_time.*{A}"):
            slice_day_gpx(day, src)
    assert not list(src.glob("*.gpx"))


def test_first_chapter_clips_that_disagree_are_refused_with_values(src, day):
    with _ct(**{A: A_START, B: B_START + timedelta(hours=5)}):
        with pytest.raises(DayGpxSliceError, match=r"disagree.*-4.*-9"):
            slice_day_gpx(day, src)


@pytest.mark.parametrize("skew", [timedelta(minutes=7), timedelta(minutes=-7)])
def test_an_offset_off_the_15_minute_grid_is_refused(src, day, skew):
    with _ct(**{A: A_START + skew, B: B_START + skew}):
        with pytest.raises(DayGpxSliceError, match="15 min"):
            slice_day_gpx(day, src)


@pytest.mark.parametrize("hours", [-13, 15])
def test_an_offset_outside_minus12_to_plus14_is_refused(src, day, hours):
    # offset = local - creation_time, so creation_time = local - offset.
    ct = {A: datetime(2024, 1, 1, 8, tzinfo=UTC) - timedelta(hours=hours),
          B: datetime(2024, 1, 1, 9, tzinfo=UTC) - timedelta(hours=hours)}  # fmt: skip
    with _ct(**ct):
        with pytest.raises(DayGpxSliceError, match="-12"):
            slice_day_gpx(day, src)


def test_a_measured_offset_still_has_to_pass_the_gpx_gate(tmp_path, src):
    # creation_time says -4, but the day GPX is another day: no trackpoints.
    other = _day_gpx(tmp_path / "other.gpx", datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC))
    with _ct(**{A: A_START, B: B_START}):
        with pytest.raises(DayGpxSliceError, match="no trackpoints"):
            slice_day_gpx(other, src)


def test_a_zero_offset_names_the_local_as_z_possibility_when_the_gate_fails(src, day):
    local_a, local_b = datetime(2024, 1, 1, 8, tzinfo=UTC), datetime(2024, 1, 1, 9, tzinfo=UTC)
    with _ct(**{A: local_a, B: local_b}):
        with pytest.raises(DayGpxSliceError, match="local-as-Z"):
            slice_day_gpx(day, src)  # day GPX starts 10:00Z: 08:00Z has no points


def test_a_zero_offset_is_allowed_when_the_gpx_gate_passes(tmp_path, src):
    gpx = _day_gpx(tmp_path / "z.gpx", datetime(2024, 1, 1, 7, 0, 0, tzinfo=UTC))
    local_a, local_b = datetime(2024, 1, 1, 8, tzinfo=UTC), datetime(2024, 1, 1, 9, tzinfo=UTC)
    with _ct(**{A: local_a, B: local_b}):
        assert slice_day_gpx(gpx, src)


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
