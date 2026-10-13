"""Inserts are GPS-anchored and gated: capture clock -> at_s, county gate,
blur + EXIF/GPS-strip gates, all before any encode. ffmpeg and the geocoder are
injected; the still privacy gate runs for real (blur off, so no model)."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.blur_gate import BlurGateError
from gomoveshift.pipeline.encoding_constants import RENDITIONS
from gomoveshift.pipeline.inserts import (
    Insert,
    InsertError,
    InsertGate,
    MediaInfo,
    anchor_photo_insert,
    apply_inserts,
    label_counties,
    plan_inserts,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.publishing.brand_policy import MediaProfile

LF = Path("/work/lf.mp4")
KEYFRAMES = [float(t) for t in range(0, 100, 2)]
RENDITION = replace(RENDITIONS["youtube_longform"], width=3840, height=2160)
T0 = datetime(2026, 10, 3, 14, 0, 0, tzinfo=timezone.utc)  # the GPX source clock start
GPS_IFD_TAG = 0x8825
COLLETON = "Colleton County, SC"


def _probe(path: Path) -> MediaInfo:
    if Path(path) == LF:
        return MediaInfo(width=3840, height=2160, fps=30.0, duration=100.0)
    return MediaInfo(width=1920, height=1080, fps=30.0, duration=6.0)


def _track():
    """One 5-minute segment, a fix every 10 s, heading north along a road."""
    return [[RoutePoint(float(t), 32.90 + t * 1e-5, -80.60) for t in range(0, 301, 10)]]


def _photo(capture_s: float | None, gps=None, name="a.jpg"):
    return dict(
        photo=Path(f"/m/{name}"),
        captured_at_utc=None if capture_s is None else T0 + timedelta(seconds=capture_s),
        track_started_at_utc=T0,
        segments=_track(),
        lf_start_s=100.0,
        lf_duration_s=100.0,
        gps=gps,
    )


def _county(name, state="South Carolina"):
    return lambda lat, lon: (name, state)


def _gate(label=COLLETON, county=("Colleton County", "South Carolina"), **kw):
    return InsertGate(label_counties=label_counties(label), county_of=_county(*county), **kw)


def _still_file(tmp_path: Path, name="a.jpg", gps=True) -> Path:
    exif = Image.Exif()
    if gps:
        exif[GPS_IFD_TAG] = {1: "N", 2: (32.0, 54.0, 0.0), 3: "W", 4: (80.0, 36.0, 0.0)}
    path = tmp_path / name
    Image.fromarray(np.full((60, 80, 3), 90, dtype=np.uint8)).save(path, "JPEG", exif=exif)
    return path


def _plan(tmp_path, inserts, gate=None, loaded=None):
    def load(path):
        if loaded is not None:
            loaded.append(Path(path))
        return Image.new("RGB", (800, 600), (255, 255, 255))

    return plan_inserts(
        LF,
        inserts,
        profile=MediaProfile(upscale="resample"),
        rendition=RENDITION,
        probe=_probe,
        keyframes=lambda p: KEYFRAMES,
        load_still=load,
        work_dir=tmp_path / "work",
        gate=gate,
    )


# --- GPS anchoring --------------------------------------------------------


def test_insert_photo_anchored_by_capture_clock():
    ins = anchor_photo_insert(**_photo(capture_s=130.0))
    assert ins.at_s == pytest.approx(30.0)  # 130 s on the GPX clock - LF start at 100 s
    assert ins.media == Path("/m/a.jpg")
    lat, lon = ins.capture
    assert lat == pytest.approx(32.90 + 130 * 1e-5) and lon == pytest.approx(-80.60)


def test_insert_photo_anchor_refuses_what_it_cannot_place():
    with pytest.raises(InsertError, match="unanchored_outside_track"):
        anchor_photo_insert(**_photo(capture_s=900.0))  # after the recording
    with pytest.raises(InsertError, match="no capture"):
        anchor_photo_insert(**_photo(capture_s=None))
    with pytest.raises(InsertError, match="outside this LF"):
        anchor_photo_insert(**_photo(capture_s=250.0))  # on the track, after the LF ends


def test_insert_photo_with_impossible_gps_takes_the_clock_position():
    ins = anchor_photo_insert(**_photo(capture_s=130.0, gps=(33.5, -79.0)))  # 70 km off the track
    assert ins.at_s == pytest.approx(30.0)
    assert ins.capture[0] == pytest.approx(32.90 + 130 * 1e-5)  # not the rejected fix


# --- county gate ----------------------------------------------------------


def test_label_counties_parses_every_label_form():
    assert label_counties(COLLETON) == {("colleton", "SC")}
    assert label_counties("Colleton & Beaufort County, SC") == {
        ("colleton", "SC"),
        ("beaufort", "SC"),
    }
    assert label_counties("Pickens County, SC & Rabun County, GA") == {
        ("pickens", "SC"),
        ("rabun", "GA"),
    }
    assert label_counties("Dingle Peninsula, Ireland") == frozenset()


def test_insert_refused_outside_label_counties(tmp_path):
    def must_not_run(*a, **k):
        raise AssertionError("privacy gate ran for an insert the county gate refuses")

    gate = _gate(county=("Beaufort County", "South Carolina"), privacy_gate=must_not_run)
    ins = Insert(10.5, _still_file(tmp_path), capture=(32.43, -80.67))
    with pytest.raises(InsertError, match="Beaufort County.*outside.*Colleton"):
        _plan(tmp_path, [ins], gate)
    # same county name in another state is another county
    nc = _gate(county=("Colleton County", "North Carolina"), privacy_gate=must_not_run)
    with pytest.raises(InsertError, match="outside"):
        _plan(tmp_path, [ins], nc)


def test_insert_refused_when_state_unknown_but_label_names_one(tmp_path):
    # the label names SC; an insert whose geocoded state is unknown (or not a state
    # we can name) passes on the county name alone otherwise
    ins = Insert(10.5, _still_file(tmp_path), capture=(32.9, -80.6))
    for unknown in (None, "", "Atlantis"):
        with pytest.raises(InsertError, match="outside"):
            _plan(tmp_path, [ins], _gate(county=("Colleton County", unknown)))
    # a label with no state still matches a lone county on its name alone
    plans = _plan(
        tmp_path,
        [ins],
        _gate(label="Colleton County", county=("Colleton County", None), blur=False),
    )
    assert plans[0].kind == "still"
    # ... but not when it names several state-less counties (which one is it?)
    with pytest.raises(InsertError, match="outside"):
        _plan(
            tmp_path,
            [ins],
            _gate(label="Colleton & Beaufort County", county=("Colleton County", None)),
        )


def test_insert_refused_same_county_other_state(tmp_path):
    ins = Insert(10.5, _still_file(tmp_path), capture=(32.9, -80.6))
    two_states = "Pickens County, SC & Rabun County, GA"
    for state in ("Georgia", "GA"):
        with pytest.raises(InsertError, match="outside"):
            _plan(tmp_path, [ins], _gate(label="Rabun County, SC", county=("Rabun County", state)))
    plans = _plan(
        tmp_path, [ins], _gate(label=two_states, county=("Rabun County", "Georgia"), blur=False)
    )
    assert plans[0].kind == "still"
    with pytest.raises(InsertError, match="outside"):
        _plan(tmp_path, [ins], _gate(label=two_states, county=("Pickens County", "Georgia")))


def test_insert_county_gate_fails_closed(tmp_path):
    src = _still_file(tmp_path, gps=False)
    with pytest.raises(InsertError, match="no capture position"):
        _plan(tmp_path, [Insert(10.5, src)], _gate())
    with pytest.raises(InsertError, match="county unresolved"):
        _plan(tmp_path, [Insert(10.5, src, capture=(1, 1))], _gate(county=(None, None)))
    with pytest.raises(InsertError, match="no county"):
        _plan(tmp_path, [Insert(10.5, src, capture=(1, 1))], _gate(label="Dingle Peninsula"))


def test_insert_in_label_county_passes_and_exif_fix_is_the_fallback(tmp_path):
    src = _still_file(tmp_path)  # carries a GPS EXIF fix, no explicit capture
    plans = _plan(tmp_path, [Insert(10.5, src)], _gate(blur=False))
    assert plans[0].kind == "still"


# --- blur and GPS-strip gates ----------------------------------------------


def test_still_is_gated_copy_with_no_gps_before_the_card_is_drawn(tmp_path):
    src = _still_file(tmp_path)
    loaded: list[Path] = []
    _plan(tmp_path, [Insert(10.5, src, capture=(32.9, -80.6))], _gate(blur=False), loaded)
    assert len(loaded) == 1 and loaded[0] != src  # the card reads the working copy
    with Image.open(loaded[0]) as img:
        assert not img.getexif().get_ifd(GPS_IFD_TAG)
    with Image.open(src) as img:
        assert img.getexif().get_ifd(GPS_IFD_TAG)  # the original is untouched


def test_clip_must_be_provably_blurred_when_blur_is_required(tmp_path):
    clip = tmp_path / "phone.mp4"
    clip.write_bytes(b"x")
    cfg = SimpleNamespace(drive=SimpleNamespace(blur_enabled=True, skip_blur=False))
    gate = InsertGate.from_config(cfg, COLLETON, county_of=_county("Colleton County"))
    ins = Insert(20.0, clip, capture=(32.9, -80.6))
    with pytest.raises(BlurGateError, match="blur"):
        _plan(tmp_path, [ins], gate)
    Path(f"{clip}.blurred").write_text("blurred\n")  # blur_gate.mark_applied's marker
    assert _plan(tmp_path, [ins], gate)[0].kind == "clip"
    skip = SimpleNamespace(drive=SimpleNamespace(blur_enabled=True, skip_blur=True))
    Path(f"{clip}.blurred").unlink()
    gate = InsertGate.from_config(skip, COLLETON, county_of=_county("Colleton County"))
    assert _plan(tmp_path, [ins], gate)[0].kind == "clip"  # --skip-blur is the named escape


def test_clip_splice_carries_only_the_lf_metadata(tmp_path):
    plans = _plan(tmp_path, [Insert(20.0, Path("/m/phone.mp4"))])
    calls: list[list[str]] = []

    def run(argv, **kw):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    apply_inserts(LF, plans, tmp_path / "out.mp4", run=run)
    enc = next(c for c in calls if "-filter_complex" in c)
    assert enc[enc.index("-map_metadata") + 1] == "0"  # never the clip's location tags


def test_plan_without_a_gate_is_unchanged(tmp_path):
    assert _plan(tmp_path, [Insert(10.5, Path("/m/a.jpg"))])[0].kind == "still"
