"""Inserts in the drive run: --insert-photos / --insert-clip flow into the
plan, the preflight print and the OST audit sidecar; the drive splice always
passes a real InsertGate. ffmpeg, the geocoder and the track are injected."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import inserts as ins_mod
from gomoveshift.pipeline.inserts import (
    DriveInserts,
    Insert,
    InsertError,
    InsertGate,
    apply_insert_flags,
    describe_drive_inserts,
    insert_records,
    master_to_lf_seconds,
    parse_insert_clip,
    place_day_inserts,
    splice_lf_inserts,
)
from gomoveshift.pipeline.ost_audit_sidecar import compute_ost_audit_hash, format_ost_audit
from gomoveshift.pipeline.preflight_plan import _render_plan_lines, build_plan


def _parse_drive(*argv: str):
    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    return parser.parse_args(["drive", "/tmp/src", *argv])


def _config():
    from gomoveshift.config import Config

    return Config()


def test_apply_insert_flags_ignores_unset_mock_args():
    """A MagicMock ``args`` (handler unit tests) carries no insert flags: its
    auto-attributes are not paths, so the config stays at its defaults."""
    config = _config()
    apply_insert_flags(MagicMock(), config)
    assert config.drive.insert_photos == ""
    assert config.drive.insert_clips == _config().drive.insert_clips


def test_drive_insert_flags_land_in_plan_and_audit(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "IMG_0001.jpg").write_bytes(b"x")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    args = _parse_drive("--insert-photos", str(photos), "--insert-clip", f"{clip}@01:30")
    config = _config()
    apply_insert_flags(args, config)
    assert config.drive.insert_photos == str(photos)
    assert config.drive.insert_clips == [f"{clip}@01:30"]

    # preflight print names every insert
    lines = describe_drive_inserts(config.drive)
    plan = build_plan(SimpleNamespace(), tmp_path, calibration=None, insert_lines=lines)
    printed = "\n".join(_render_plan_lines(plan))
    assert "IMG_0001.jpg" in printed
    assert "clip.mp4" in printed and "01:30" in printed

    # the audit sidecar lists them per LF and they move the confirmation hash
    di = DriveInserts.from_drive(config.drive, clip_date="2026-10-03")
    placed = {1: [Insert(90.0, clip, style="full"), Insert(12.5, photos / "IMG_0001.jpg")]}
    state = {"per_lf_ost_text": {1: "Colleton County, SC"}}
    base = format_ost_audit("2026-10-03", state, [1])
    audit = format_ost_audit("2026-10-03", state, [1], inserts=insert_records(placed))
    entry = audit["lfs"][0]["inserts"]
    assert [(e["kind"], e["media"], e["at_s"]) for e in entry] == [
        ("clip", "clip.mp4", 90.0),
        ("still", "IMG_0001.jpg", 12.5),
    ]
    assert "inserts" not in base["lfs"][0]
    assert compute_ost_audit_hash(audit) != compute_ost_audit_hash(base)
    assert di is not None and di.clip_date == "2026-10-03"

    # the plan the drive splice builds always carries a real gate
    seen = {}

    def spy_plan(lf, inserts, **kw):
        seen.update(kw)
        return []

    lf = tmp_path / "lf_01.mp4"
    lf.write_bytes(b"x")
    out_calls = []

    def spy_apply(lf_path, plans, out, **kw):
        out_calls.append(out)
        Path(out).write_bytes(b"spliced")
        return ins_mod.InsertResult(output=Path(out))

    splice_lf_inserts(
        lf,
        placed[1],
        config=config,
        label="Colleton County, SC",
        profile=None,
        work_dir=tmp_path / "w",
        plan=spy_plan,
        apply=spy_apply,
    )
    assert isinstance(seen["gate"], InsertGate)
    assert lf.read_bytes() == b"spliced"


def test_splice_without_config_still_passes_a_closed_gate(tmp_path):
    seen = {}
    lf = tmp_path / "lf.mp4"
    lf.write_bytes(b"x")
    splice_lf_inserts(
        lf,
        [Insert(1.0, tmp_path / "c.mp4")],
        config=None,
        label="A County, SC",
        profile=None,
        work_dir=tmp_path / "w",
        plan=lambda *a, **kw: seen.update(kw) or [],
        apply=lambda lf, p, o, **kw: (Path(o).write_bytes(b"y"), ins_mod.InsertResult(Path(o)))[1],
    )
    gate = seen["gate"]
    assert isinstance(gate, InsertGate)
    assert gate.clip_blurred(tmp_path / "c.mp4") is False  # fail closed


def test_parse_insert_clip():
    assert parse_insert_clip("/a/b@c.mp4@02:05") == (Path("/a/b@c.mp4"), 125.0)
    for bad in ("/a/clip.mp4", "/a/clip.mp4@1:5", "@01:00", "/a.mp4@01:75", "/a.mp4@xx"):
        with pytest.raises(InsertError):
            parse_insert_clip(bad)


def test_bad_clip_flag_is_a_parse_error(capsys):
    with pytest.raises(SystemExit):
        _parse_drive("--insert-clip", "no-timestamp.mp4")
    assert "PATH@MM:SS" in capsys.readouterr().err


def test_describe_refuses_missing_media(tmp_path):
    drive = SimpleNamespace(insert_photos=str(tmp_path / "nope"), insert_clips=[])
    with pytest.raises(InsertError, match="not a directory"):
        describe_drive_inserts(drive)
    drive = SimpleNamespace(insert_photos="", insert_clips=[f"{tmp_path / 'gone.mp4'}@00:10"])
    with pytest.raises(InsertError, match="gone.mp4"):
        describe_drive_inserts(drive)


def test_master_to_lf_seconds_walks_ranges_after_the_montage():
    ranges = [(100.0, 160.0), (400.0, 430.0)]
    assert master_to_lf_seconds(110.0, ranges, 15.0) == 25.0
    assert master_to_lf_seconds(410.0, ranges, 15.0) == 85.0
    assert master_to_lf_seconds(200.0, ranges, 15.0) is None


class _Ep:
    def __init__(self, ct, dur):
        self.creation_time, self.total_duration = ct, dur


def _day(photo: Path, clip: Path, tmp_path, *, captured, gps):
    di = DriveInserts(photos=(photo,), clips=((clip, 30.0),), clip_date="2026-10-03")
    episodes = [_Ep(datetime(2026, 10, 3, 9, 0, 0), 600.0)]
    return dict(
        di=di,
        date="2026-10-03",
        episodes=episodes,
        boundaries=[(0.0, 600.0)],
        gps_points=gps,
        lf_ranges={1: [(100.0, 400.0)]},
        lead_s=15.0,
        capture_of=lambda p: captured,
        gps_of=lambda p: None,
    )


def test_place_day_inserts_anchors_photo_and_puts_clip_in_first_lf(tmp_path):
    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    gps = [(float(t), 32.9 + t * 1e-5, -80.6) for t in range(0, 601, 5)]
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 3, 9, 3, 20), gps=gps)
    placed = place_day_inserts(**kw)  # 200 s into the master -> 100 s into the LF body
    by_media = {i.media: i for i in placed[1]}
    assert by_media[photo].at_s == pytest.approx(115.0)
    assert by_media[clip].at_s == 30.0


def test_place_day_inserts_refuses_a_photo_in_no_long_form(tmp_path):
    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    gps = [(float(t), 32.9 + t * 1e-5, -80.6) for t in range(0, 601, 5)]
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 3, 9, 1, 0), gps=gps)
    with pytest.raises(InsertError, match="no long-form"):
        place_day_inserts(**kw)  # 60 s: before the LF's first range


def test_place_day_inserts_refuses_a_photo_with_no_track(tmp_path):
    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 3, 9, 3, 20), gps=None)
    with pytest.raises(InsertError):
        place_day_inserts(**kw)


def test_photo_on_another_day_is_left_for_that_day(tmp_path):
    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 9, 9, 3, 20), gps=None)
    placed = place_day_inserts(**kw)
    assert [i.media for i in placed[1]] == [clip]


def test_clip_goes_only_to_its_date(tmp_path):
    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 9, 9, 3, 20), gps=None)
    kw["date"] = "2026-10-04"
    assert place_day_inserts(**kw) == {}


def test_clip_past_the_end_of_the_lf_is_refused(tmp_path):
    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 9, 9, 3, 20), gps=None)
    kw["di"] = DriveInserts(clips=((clip, 9999.0),), clip_date="2026-10-03")
    with pytest.raises(InsertError, match="past the end"):
        place_day_inserts(**kw)


# ── the drive path itself: clip gate, early gate, run_all_days, Phase D ─────────


def test_drive_placed_clip_carries_a_track_position_and_passes_the_real_gate(tmp_path):
    from gomoveshift.pipeline.inserts import InsertGate, gate_inserts, label_counties

    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    gps = [(float(t), 32.9 + t * 1e-5, -80.6) for t in range(0, 601, 5)]
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 9, 9, 0, 0), gps=gps)
    (placed,) = place_day_inserts(**kw).values()
    (c,) = placed
    assert c.capture == pytest.approx((32.9 + 115e-5, -80.6))  # LF 30 s -> master 115 s
    gate = InsertGate(
        label_counties("Colleton County, SC"),
        county_of=lambda la, lo: ("Colleton", "SC"),
        clip_blurred=lambda p: True,
    )
    assert gate_inserts([c], gate, work_dir=tmp_path / "w") == [c]


def test_drive_clip_with_no_nearby_fix_is_refused(tmp_path):
    from gomoveshift.pipeline.inserts import gate_inserts

    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    kw = _day(photo, clip, tmp_path, captured=datetime(2026, 10, 9, 9, 0, 0), gps=[(500.0, 1, 2)])
    (placed,) = place_day_inserts(**kw).values()
    assert placed[0].capture is None
    gate = InsertGate(
        frozenset({("colleton", "SC")}),
        county_of=lambda la, lo: ("Colleton", "SC"),
        clip_blurred=lambda p: True,
    )
    with pytest.raises(InsertError, match="no capture position"):
        gate_inserts(placed, gate, work_dir=tmp_path / "w")


def _drive_cfg(tmp_path, clip):
    config = _config()
    config.drive.insert_clips = [f"{clip}@00:30"]
    return config


def _batch():
    return SimpleNamespace(date="2026-10-03", episodes=[])


def test_plan_drive_inserts_runs_the_whole_gate_before_any_encode(tmp_path):
    from gomoveshift.pipeline.inserts import plan_drive_inserts

    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    config = _drive_cfg(tmp_path, clip)
    placed = {1: [Insert(30.0, clip, capture=(32.9, -80.6))]}

    def run(county, blurred=True):
        with patch.object(ins_mod, "master_blur_proven", return_value=blurred):
            return plan_drive_inserts(
                config,
                [_batch()],
                lf_plans_by_date={},
                gps_points_by_date={},
                label_of=lambda d, i: "Colleton County, SC",
                place=lambda *a, **k: placed,
                gate_kw=dict(county_of=lambda la, lo: county),
            )

    assert run(("Colleton", "SC")) == {"2026-10-03": placed}
    with pytest.raises(InsertError, match="outside the LF label"):
        run(("Beaufort", "SC"))
    with pytest.raises(ins_mod.BlurGateError):
        run(("Colleton", "SC"), blurred=False)


def test_insert_label_prefers_the_county_label_over_the_ost_text():
    from gomoveshift.pipeline.inserts import insert_label

    assert insert_label({1: "Colleton County, SC"}, 1, "A to B via R1, SC") == "Colleton County, SC"
    assert insert_label({}, 1, "Colleton County, SC") == "Colleton County, SC"


# ── the real-batch glue and the all-days planning (resume-safe) ──────────────────


def _two_ep_batch(date="2026-10-03", start=datetime(2026, 10, 3, 9, 0, 0)):
    eps = [_Ep(start, 600.0), _Ep(start.replace(minute=30), 600.0)]
    return SimpleNamespace(date=date, episodes=eps)


def _seg(ep, a, b):
    return SimpleNamespace(source_file_index=ep, start_time=a, end_time=b)


def test_plan_day_from_batch_builds_boundaries_and_lf_master_ranges(tmp_path):
    from gomoveshift.pipeline.inserts import plan_day_from_batch

    photo, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    gps = [(float(t), 32.9 + t * 1e-5, -80.6) for t in range(0, 1201, 5)]
    lf_plans = [
        SimpleNamespace(index=1, segments=[]),  # no segments: not an LF the insert can land in
        SimpleNamespace(index=2, segments=[_seg(0, 100.0, 200.0), _seg(1, 50.0, 80.0)]),
    ]
    di = DriveInserts(photos=(photo,), clips=((clip, 20.0),), clip_date="2026-10-03")
    # photo 60 s into the 2nd episode = master 660 s = LF2 range 2 (650-680), after 100 s of range 1
    placed = plan_day_from_batch(
        di,
        _two_ep_batch(),
        lf_plans,
        gps,
        lead_s=15.0,
        capture_of=lambda p: datetime(2026, 10, 3, 9, 31, 0),
        gps_of=lambda p: None,
    )
    assert list(placed) == [2]
    by_media = {i.media: i for i in placed[2]}
    assert by_media[photo].at_s == pytest.approx(15.0 + 100.0 + 10.0)
    assert by_media[clip].at_s == 20.0  # the first LF that has a range


def test_photos_unplaced_names_a_photo_no_day_holds(tmp_path):
    from gomoveshift.pipeline.inserts import photos_unplaced

    inside, outside = tmp_path / "in.jpg", tmp_path / "out.jpg"
    when = {inside: datetime(2026, 10, 3, 9, 5, 0), outside: datetime(2026, 11, 1, 9, 5, 0)}
    di = DriveInserts(photos=(inside, outside))
    assert photos_unplaced(di, [_two_ep_batch()], capture_of=when.get) == ["out.jpg"]
    assert photos_unplaced(DriveInserts(photos=(inside,)), [], capture_of=when.get) == ["in.jpg"]


def test_photos_unplaced_sees_a_completed_day_only_when_given_every_day(tmp_path):
    """The resume bug: judged against the remaining days alone, a photo of a finished day is 'lost'."""
    from gomoveshift.pipeline.inserts import photos_unplaced

    photo = tmp_path / "day1.jpg"
    day1 = _two_ep_batch("2026-10-03")
    day2 = _two_ep_batch("2026-10-04", datetime(2026, 10, 4, 9, 0, 0))
    di = DriveInserts(photos=(photo,))
    when = lambda p: datetime(2026, 10, 3, 9, 5, 0)  # noqa: E731
    assert photos_unplaced(di, [day2], capture_of=when) == ["day1.jpg"]
    assert photos_unplaced(di, [day1, day2], capture_of=when) == []


def _plan_cfg(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    return clip, _drive_cfg(tmp_path, clip)


def test_plan_drive_inserts_clip_date_is_the_drives_first_day_not_the_first_rendered(tmp_path):
    from gomoveshift.pipeline.inserts import plan_drive_inserts

    clip, config = _plan_cfg(tmp_path)
    day1, day2 = _batch(), SimpleNamespace(date="2026-10-04", episodes=[])
    seen = []

    def place(di, batch, *a, **k):
        seen.append((di.clip_date, batch.date))
        return {}

    plan_drive_inserts(
        config,
        [day1, day2],
        lf_plans_by_date={},
        gps_points_by_date={},
        label_of=lambda d, i: "Colleton County, SC",
        place=place,
        render_dates={"2026-10-04"},
    )
    assert seen == [("2026-10-03", "2026-10-03"), ("2026-10-03", "2026-10-04")]


def test_plan_drive_inserts_gates_only_the_days_being_rendered(tmp_path):
    from gomoveshift.pipeline.inserts import plan_drive_inserts

    clip, config = _plan_cfg(tmp_path)
    done, todo = _batch(), SimpleNamespace(date="2026-10-04", episodes=[])
    stray = Insert(30.0, clip, capture=(1.0, 2.0))  # outside the label: the gate refuses it

    def run(render_dates):
        return plan_drive_inserts(
            config,
            [done, todo],
            lf_plans_by_date={},
            gps_points_by_date={},
            label_of=lambda d, i: "Colleton County, SC",
            place=lambda di, b, *a, **k: {1: [stray]} if b is done else {},
            gate_kw=dict(county_of=lambda la, lo: ("Beaufort", "SC")),
            render_dates=render_dates,
        )

    with pytest.raises(InsertError, match="outside the LF label"):
        run(None)  # no render set: every day is gated, as a fresh run does
    placed = run({"2026-10-04"})  # the finished day already carries its insert
    assert placed["2026-10-03"] == {1: [stray]}


def test_drive_refuses_inserts_until_the_splice_is_wired(tmp_path, capsys):
    """Inserts are planning only: a run that asked for inserts must not render without them."""
    from test_drive_skip_flags_rejected import _drive_args

    from gomoveshift.cli.handlers.drive import cmd_drive

    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "a.jpg").write_bytes(b"x")
    args = _drive_args(insert_photos=str(photos), insert_clip=None)
    assert cmd_drive(args, _config()) == 2
    assert "not spliced yet" in capsys.readouterr().err
